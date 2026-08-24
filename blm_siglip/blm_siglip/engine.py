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
                 is_dist=False, rank=0):
        self.model = model
        self.cfg = cfg
        self.tcfg = cfg["train"]
        self.train_loader = train_loader
        self.device = device
        self.is_dist = is_dist
        self.rank = rank

        self.loss_fn = SiglipLoss(use_dist=is_dist)
        self.evaluator = Evaluator(model, eval_loader, device) if eval_loader is not None else None

        self.optimizer = torch.optim.AdamW(
            model.param_groups(self.tcfg["lr_vision"], self.tcfg["lr_text"],
                               self.tcfg["lr_proj"], self.tcfg["weight_decay"]),
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

    def save_checkpoint(self, step: int):
        path = os.path.join(self.output_dir, f"ckpt_step{step}.pt")
        torch.save({"step": step, "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "config": self.cfg}, path)
        ckpts = sorted(f for f in os.listdir(self.output_dir) if f.startswith("ckpt_step"))
        while len(ckpts) > self.tcfg["max_checkpoints"]:
            os.remove(os.path.join(self.output_dir, ckpts.pop(0)))
        return path

    def train(self):
        from blm_clip.data import move_batch_to_device
        max_steps = self.tcfg["max_steps"]
        accum = self.tcfg["accum_steps"]

        self.model.train()
        step, micro = 0, 0
        running_loss, running_n = 0.0, 0
        t0 = time.time()
        pbar = tqdm(total=max_steps, desc="train", disable=self.rank != 0)

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
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg["grad_clip"])
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
                    running_loss, running_n, t0 = 0.0, 0, time.time()

                if self.rank == 0 and step > 0 and step % self.tcfg["eval_every"] == 0 and self.evaluator:
                    tqdm.write(f"[step {step}] eval: {json.dumps(self.evaluator.run(), ensure_ascii=False)}")
                if self.rank == 0 and step > 0 and step % self.tcfg["save_every"] == 0:
                    self.save_checkpoint(step)
                pbar.update(1 if step <= max_steps else 0)
                if step >= max_steps:
                    break

        pbar.close()
        if self.rank == 0:
            self.save_checkpoint(step)
            if self.evaluator:
                print(f"[final] eval: {json.dumps(self.evaluator.run(), ensure_ascii=False)}")
