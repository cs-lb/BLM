# -*- coding: utf-8 -*-
"""
训练引擎与评估器
====================
Trainer:
  - bf16 混合精度（autocast，不断层主权重保持 fp32）
  - 判别式学习率：视觉塔 1e-5 / 文本塔 5e-5 / 温度与投影 5e-5
  - warmup + cosine 学习率调度
  - 两种大 batch 模式：梯度累积（accum_steps）或 GradCache（推荐）
  - 周期性检索评估 + checkpoint 轮转保存

Evaluator:
  - 在评估集上计算 图搜文 / 文搜图 的 R@1、R@5、R@10
  - hit@5 = 图搜文 R@5（对齐 BLM 业务口径）
"""

import json
import math
import os
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import move_batch_to_device
from .gradcache import GradCache
from .losses import ClipLoss


# ---------------------------------------------------------------
# 学习率调度：linear warmup + cosine decay
# ---------------------------------------------------------------
def cosine_warmup_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


# ---------------------------------------------------------------
# 评估器
# ---------------------------------------------------------------
class Evaluator:
    def __init__(self, model, dataloader: DataLoader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    @torch.no_grad()
    def run(self) -> dict:
        self.model.eval()
        img_embeds, txt_embeds = [], []
        for batch in self.dataloader:
            batch = move_batch_to_device(batch, self.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                iz, tz, _ = self.model(batch)
            img_embeds.append(iz.float().cpu())
            txt_embeds.append(tz.float().cpu())
        self.model.train()

        sim = torch.cat(img_embeds) @ torch.cat(txt_embeds).t()   # [Ni, Nt]
        n = sim.shape[0]
        metrics = {}
        for k in (1, 5, 10):
            k = min(k, n)
            # 图搜文：每张图的配对文本是否进入 Top-K
            i2t_topk = sim.topk(k, dim=1).indices
            i2t_hit = (i2t_topk == torch.arange(n).unsqueeze(1)).any(dim=1).float().mean().item()
            # 文搜图：对称方向
            t2i_topk = sim.topk(k, dim=0).indices
            t2i_hit = (t2i_topk == torch.arange(n).unsqueeze(0)).any(dim=0).float().mean().item()
            metrics[f"i2t_R@{k}"] = round(i2t_hit, 4)
            metrics[f"t2i_R@{k}"] = round(t2i_hit, 4)
        metrics["hit@5"] = metrics.get("i2t_R@5", metrics.get("i2t_R@1"))  # 业务口径
        return metrics


# ---------------------------------------------------------------
# 训练器
# ---------------------------------------------------------------
class Trainer:
    def __init__(self, model, cfg: dict, train_loader, eval_loader, device, is_dist=False, rank=0):
        self.model = model
        self.cfg = cfg
        self.tcfg = cfg["train"]
        self.train_loader = train_loader
        self.device = device
        self.is_dist = is_dist
        self.rank = rank

        self.loss_fn = ClipLoss(use_dist=is_dist)
        self.grad_cache = GradCache(model, self.loss_fn) if self.tcfg["use_gradcache"] else None
        self.evaluator = Evaluator(model, eval_loader, device) if eval_loader is not None else None

        self.optimizer = torch.optim.AdamW(
            model.param_groups(
                self.tcfg["lr_vision"], self.tcfg["lr_text"],
                self.tcfg["lr_proj"], self.tcfg["weight_decay"],
            ),
            betas=tuple(self.tcfg["betas"]), eps=self.tcfg["eps"],
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda s: cosine_warmup_lambda(s, self.tcfg["warmup_steps"], self.tcfg["max_steps"]),
        )

        self.output_dir = self.tcfg["output_dir"]
        if rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)

        self.use_bf16 = self.tcfg["bf16"] and device.type == "cuda"

    # ---------------- checkpoint ----------------
    def save_checkpoint(self, step: int):
        state = {
            "step": step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.cfg,
        }
        path = os.path.join(self.output_dir, f"ckpt_step{step}.pt")
        torch.save(state, path)

        # 轮转：只保留最近 max_checkpoints 个
        ckpts = sorted(f for f in os.listdir(self.output_dir) if f.startswith("ckpt_step"))
        while len(ckpts) > self.tcfg["max_checkpoints"]:
            os.remove(os.path.join(self.output_dir, ckpts.pop(0)))
        return path

    # ---------------- 主训练循环 ----------------
    def train(self):
        max_steps = self.tcfg["max_steps"]
        accum = self.tcfg["accum_steps"]
        use_gc = self.grad_cache is not None

        self.model.train()
        step = 0
        micro_count = 0
        running_loss, running_n = 0.0, 0
        t0 = time.time()

        pbar = tqdm(total=max_steps, desc="train", disable=self.rank != 0)
        while step < max_steps:
            for batch in self.train_loader:
                if step >= max_steps:
                    break
                batch = move_batch_to_device(batch, self.device)

                if use_gc:
                    # GradCache：内部完成缓存+反传，一次调用 = 一次完整参数更新
                    self.optimizer.zero_grad(set_to_none=True)
                    loss_val = self.grad_cache.step(batch, self.tcfg["gc_chunk_size"])
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg["grad_clip"])
                    self.optimizer.step()
                    self.scheduler.step()
                    step += 1
                    running_loss += loss_val
                    running_n += 1
                else:
                    # 常规梯度累积
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                        iz, tz, scale = self.model(batch)
                        loss = self.loss_fn(iz, tz, scale) / accum
                    loss.backward()
                    running_loss += loss.item() * accum
                    running_n += 1
                    micro_count += 1
                    if micro_count % accum == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg["grad_clip"])
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        step += 1

                # ---------- 日志 / 评估 / 保存（仅主进程）----------
                if step > 0 and step % self.tcfg["log_every"] == 0 and running_n > 0:
                    if self.is_dist:  # 多卡 loss 取均值
                        t = torch.tensor([running_loss / running_n], device=self.device)
                        dist.all_reduce(t)
                        avg_loss = (t / dist.get_world_size()).item()
                    else:
                        avg_loss = running_loss / running_n
                    if self.rank == 0:
                        lr = self.scheduler.get_last_lr()[0]
                        tau = self.model.clamped_logit_scale().exp().item()
                        ips = running_n / max(1e-9, time.time() - t0)
                        pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}",
                                         tau=f"{tau:.1f}", it_s=f"{ips:.2f}")
                    running_loss, running_n, t0 = 0.0, 0, time.time()

                if self.rank == 0 and step > 0 and step % self.tcfg["eval_every"] == 0 and self.evaluator:
                    metrics = self.evaluator.run()
                    tqdm.write(f"[step {step}] eval: {json.dumps(metrics, ensure_ascii=False)}")

                if self.rank == 0 and step > 0 and step % self.tcfg["save_every"] == 0:
                    self.save_checkpoint(step)

                pbar.update(1 if step <= max_steps else 0)
                if step >= max_steps:
                    break

        pbar.close()
        if self.rank == 0:
            self.save_checkpoint(step)
            if self.evaluator:
                metrics = self.evaluator.run()
                print(f"[final] eval: {json.dumps(metrics, ensure_ascii=False)}")
