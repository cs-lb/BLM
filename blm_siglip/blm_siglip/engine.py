# -*- coding: utf-8 -*-
"""
训练引擎与评估器（SigLIP 版）
================================
与 blm_clip 的差异：
- 损失换 SigLIP（逐对 sigmoid，各卡只算本地 query 行，无需全局归一）；
- 模型 forward 返回 4 元组（img_z, txt_z, logit_scale, logit_bias）；
- 温度日志同时打印 τ = exp(logit_scale) 与 b = logit_bias。
"""

import json
import math
import os
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import SiglipLoss


def cosine_warmup_lambda(step: int, warmup: int, total: int) -> float:
    """linear warmup + cosine decay。"""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


class Evaluator:
    """检索评估：i2t/t2i 的 R@1/5/10，hit@5 = i2t R@5（业务口径）。"""

    def __init__(self, model, dataloader: DataLoader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    @torch.no_grad()
    def run(self) -> dict:
        from blm_clip.data import move_batch_to_device
        self.model.eval()
        img_embeds, txt_embeds = [], []
        for batch in self.dataloader:
            batch = move_batch_to_device(batch, self.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=self.device.type == "cuda"):
                iz, tz = self.model.encode_image(batch["image_inputs"]), \
                         self.model.encode_text(batch["input_ids"], batch["attention_mask"])
            img_embeds.append(iz.float().cpu())
            txt_embeds.append(tz.float().cpu())
        self.model.train()

        sim = torch.cat(img_embeds) @ torch.cat(txt_embeds).t()
        n = sim.shape[0]
        metrics = {}
        for k in (1, 5, 10):
            k = min(k, n)
            i2t = (sim.topk(k, dim=1).indices == torch.arange(n).unsqueeze(1)).any(1).float().mean().item()
            t2i = (sim.topk(k, dim=0).indices == torch.arange(n).unsqueeze(0)).any(0).float().mean().item()
            metrics[f"i2t_R@{k}"], metrics[f"t2i_R@{k}"] = round(i2t, 4), round(t2i, 4)
        metrics["hit@5"] = metrics.get("i2t_R@5", metrics["i2t_R@1"])
        return metrics


class Trainer:
    def __init__(self, model, cfg: dict, train_loader, eval_loader, device,
                 is_dist=False, rank=0, start_step: int = 0):
        self.model = model
        self.cfg = cfg
        self.tcfg = cfg["train"]
        self.train_loader = train_loader
        self.device = device
        self.is_dist = is_dist
        self.rank = rank

        self.loss_fn = SiglipLoss(use_dist=is_dist)
        self.evaluator = Evaluator(model, eval_loader, device) if eval_loader is not None else None

        # 24GB 级显卡（双卡 4090）：fp32 AdamW 的 m+v 状态 ~7.3GB（0.91B 参数），
        # 叠加 fp32 权重+梯度与 bf16 激活后 OOM。8-bit AdamW 把状态压到 1/4（~1.9GB），
        # 精度基本无损（bitsandbytes 官方口径，LLM 微调标配）。
        # 无 bitsandbytes 的环境（如 Mac MPS）自动回退原版 AdamW，可用 optim_8bit: false 关闭。
        optim_cls = torch.optim.AdamW
        if self.tcfg.get("optim_8bit", True) and device.type == "cuda":
            try:
                import bitsandbytes as bnb
                optim_cls = bnb.optim.AdamW8bit
            except ImportError:
                if rank == 0:
                    print("[env] 未安装 bitsandbytes，回退 fp32 AdamW"
                          "（24GB 显卡建议 pip install bitsandbytes）")
        if rank == 0:
            print(f"[env] optimizer = {optim_cls.__name__}")

        self.optimizer = optim_cls(
            model.param_groups(self.tcfg["lr_vision"], self.tcfg["lr_text"],
                               self.tcfg["lr_proj"], self.tcfg["weight_decay"]),
            betas=tuple(self.tcfg["betas"]), eps=self.tcfg["eps"],
        )
        # 断点续训：start_step>0 时调度器直接快进到对应位置（LR 不重走 warmup）
        self.start_step = start_step
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda s: cosine_warmup_lambda(s, self.tcfg["warmup_steps"], self.tcfg["max_steps"]),
            last_epoch=start_step - 1,
        )
        self.output_dir = self.tcfg["output_dir"]
        if rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)
            # 训练指标持久化：loss 曲线/超参轨迹/评估结果，便于事后画曲线与排查
            self.metrics_path = os.path.join(self.output_dir, "metrics.jsonl")
        self.use_bf16 = self.tcfg["bf16"] and device.type == "cuda"

    def _log_metrics(self, record: dict):
        """追加一行 JSON 到 metrics.jsonl（仅主进程调用）。"""
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_checkpoint(self, step: int):
        """原子写入 + 失败容错：磁盘满等 IO 故障只告警跳过，不中断训练。

        教训（2026-08-26 双 4090 事故）：直接 torch.save 到已满磁盘，
        留下半截 ckpt 且训练进程崩溃（损失 2000 步进度）。
        """
        path = os.path.join(self.output_dir, f"ckpt_step{step}.pt")
        tmp = path + ".tmp"
        try:
            torch.save({"step": step, "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(), "config": self.cfg}, tmp)
            os.replace(tmp, path)          # 原子替换：要么完整新文件，要么保留旧文件
        except (RuntimeError, OSError) as e:
            for p in (tmp, path):
                if os.path.exists(p):
                    try:
                        os.remove(p)       # 清掉半截文件，避免误当有效 ckpt resume
                    except OSError:
                        pass
            print(f"[warn] checkpoint 保存失败（step {step}）：{e}，已跳过本次保存", flush=True)
            return None
        # 按步数数值排序清理旧 ckpt（字典序会把 step1000 排在 step20 前面，是错的）
        ckpts = [f for f in os.listdir(self.output_dir)
                 if f.startswith("ckpt_step") and f.endswith(".pt")]
        ckpts.sort(key=lambda f: int(f[len("ckpt_step"):-len(".pt")]))
        while len(ckpts) > self.tcfg["max_checkpoints"]:
            os.remove(os.path.join(self.output_dir, ckpts.pop(0)))
        return path

    def train(self):
        from blm_clip.data import move_batch_to_device
        max_steps = self.tcfg["max_steps"]
        accum = self.tcfg["accum_steps"]

        self.model.train()
        step, micro = self.start_step, 0       # 续训从断点步数继续（LR 已在对应位置）
        running_loss, running_n = 0.0, 0
        t0 = time.time()
        pbar = tqdm(total=max_steps, initial=self.start_step,
                    desc="train", disable=self.rank != 0)

        while step < max_steps:
            for batch in self.train_loader:
                if step >= max_steps:
                    break
                batch = move_batch_to_device(batch, self.device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=self.use_bf16):
                    img_z, txt_z, logit_scale, logit_bias = self.model(batch)
                    loss = self.loss_fn(img_z, txt_z, logit_scale, logit_bias) / accum
                loss.backward()
                running_loss += loss.item() * accum
                running_n += 1
                micro += 1

                if micro % accum == 0:
                    # clip_grad_norm_ 返回裁剪前的总梯度范数，顺手记录用于异常监控
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.tcfg["grad_clip"]).item()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    step += 1

                if step > 0 and step % self.tcfg["log_every"] == 0 and running_n > 0:
                    avg = running_loss / running_n
                    if self.is_dist:
                        t = torch.tensor([avg], device=self.device)
                        dist.all_reduce(t)
                        avg = (t / dist.get_world_size()).item()
                    if self.rank == 0:
                        pbar.set_postfix(
                            loss=f"{avg:.4f}",
                            lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                            tau=f"{self.model.logit_scale.exp().item():.2f}",
                            b=f"{self.model.logit_bias.item():.2f}",
                            it_s=f"{running_n / max(1e-9, time.time() - t0):.2f}")
                        self._log_metrics({
                            "step": step, "loss": round(avg, 5),
                            "lr": self.scheduler.get_last_lr()[0],
                            "tau": round(self.model.logit_scale.exp().item(), 3),
                            "bias": round(self.model.logit_bias.item(), 3),
                            "grad_norm": round(grad_norm, 3),
                            "samples_per_s": round(running_n / max(1e-9, time.time() - t0), 2),
                        })
                    running_loss, running_n, t0 = 0.0, 0, time.time()

                if self.rank == 0 and step > 0 and step % self.tcfg["eval_every"] == 0 and self.evaluator:
                    metrics = self.evaluator.run()
                    tqdm.write(f"[step {step}] eval: {json.dumps(metrics, ensure_ascii=False)}")
                    self._log_metrics({"step": step, "eval": metrics})
                if self.rank == 0 and step > 0 and step % self.tcfg["save_every"] == 0:
                    self.save_checkpoint(step)
                pbar.update(1 if step <= max_steps else 0)
                if step >= max_steps:
                    break

        pbar.close()
        if self.rank == 0:
            self.save_checkpoint(step)
            if self.evaluator:
                final_metrics = self.evaluator.run()
                print(f"[final] eval: {json.dumps(final_metrics, ensure_ascii=False)}")
                self._log_metrics({"step": step, "final_eval": final_metrics})
