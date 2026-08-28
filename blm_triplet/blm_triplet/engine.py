# -*- coding: utf-8 -*-
"""
三元损失 + 蒸馏训练引擎
==========================
复用 blm_siglip 的双塔模型（BLMSiglipModel，从 v1 ckpt 继续训练）；
前向：图 anchor [A, D] × 候选文本 [A, K, D] → 候选级相似度 [A, K]
损失：L = λ_t·Triplet(batch-hard) + λ_d·DistillMSE(σ(α·cos+β) vs teacher)
"""

import json
import math
import os
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .losses import TripletDistillLoss


def cosine_warmup_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


class TripletTrainer:
    def __init__(self, model, cfg: dict, train_loader, eval_fn=None, device=None):
        self.model = model
        self.cfg = cfg
        self.tcfg = cfg["train"]
        self.train_loader = train_loader
        self.eval_fn = eval_fn
        self.device = device

        # 组合损失（α、β 可学习，并入优化器）
        self.loss_fn = TripletDistillLoss(
            margin=self.tcfg["margin"],
            lambda_triplet=self.tcfg["lambda_triplet"],
            lambda_distill=self.tcfg["lambda_distill"],
        ).to(device)

        # 优化器：模型参数分组 + 损失的 α/β 并入 proj 组
        groups = model.param_groups(self.tcfg["lr_vision"], self.tcfg["lr_text"],
                                    self.tcfg["lr_proj"], self.tcfg["weight_decay"])
        groups.append({"params": list(self.loss_fn.aligner.parameters()),
                       "lr": self.tcfg["lr_proj"], "weight_decay": 0.0, "name": "aligner"})
        self.optimizer = torch.optim.AdamW(groups, betas=tuple(self.tcfg["betas"]),
                                           eps=self.tcfg["eps"])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda s: cosine_warmup_lambda(s, self.tcfg["warmup_steps"], self.tcfg["max_steps"]))

        self.output_dir = self.tcfg["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        self.metrics_path = os.path.join(self.output_dir, "metrics.jsonl")
        self.use_bf16 = self.tcfg["bf16"] and device.type == "cuda"

    def _log(self, record: dict):
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_checkpoint(self, step: int):
        path = os.path.join(self.output_dir, f"ckpt_step{step}.pt")
        torch.save({"step": step, "model": self.model.state_dict(),
                    "aligner": self.loss_fn.aligner.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "config": self.cfg}, path)
        ckpts = sorted(f for f in os.listdir(self.output_dir) if f.startswith("ckpt_step"))
        while len(ckpts) > self.tcfg["max_checkpoints"]:
            os.remove(os.path.join(self.output_dir, ckpts.pop(0)))
        return path

    def train(self):
        from .data import move_to_device
        max_steps = self.tcfg["max_steps"]
        self.model.train()
        step = 0
        buf, t0 = {"t": 0.0, "d": 0.0, "n": 0}, time.time()
        pbar = tqdm(total=max_steps, desc="triplet-train")

        while step < max_steps:
            for batch in self.train_loader:
                if step >= max_steps:
                    break
                batch = move_to_device(batch, self.device)
                A, K, L = batch["cand_ids"].shape

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=self.use_bf16):
                    # anchor 图编码 [A, D]
                    img_z = self.model.encode_image(batch["image_inputs"])
                    # K 个候选文本编码 [A*K, D] → [A, K, D]
                    cand_z = self.model.encode_text(
                        batch["cand_ids"].view(A * K, L), batch["cand_mask"].view(A * K, L)
                    ).view(A, K, -1)
                    # 候选级余弦相似度 [A, K]
                    sims = (img_z.unsqueeze(1) * cand_z).sum(-1)

                    loss, parts = self.loss_fn(sims, batch["labels"],
                                               batch["teacher_scores"], batch["conf_weights"])

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.loss_fn.parameters()),
                    self.tcfg["grad_clip"]).item()
                self.optimizer.step()
                self.scheduler.step()
                step += 1
                pbar.update(1)

                buf["t"] += parts["l_triplet"]; buf["d"] += parts["l_distill"]; buf["n"] += 1
                if step % self.tcfg["log_every"] == 0:
                    rec = {"step": step,
                           "l_triplet": round(buf["t"] / buf["n"], 4),
                           "l_distill": round(buf["d"] / buf["n"], 4),
                           "alpha": round(parts["alpha"], 3), "beta": round(parts["beta"], 3),
                           "lr": self.scheduler.get_last_lr()[0],
                           "grad_norm": round(grad_norm, 3),
                           "steps_per_s": round(buf["n"] / max(1e-9, time.time() - t0), 2)}
                    pbar.set_postfix(**{k: v for k, v in rec.items() if k != "step"})
                    self._log(rec)
                    buf, t0 = {"t": 0.0, "d": 0.0, "n": 0}, time.time()

                if self.eval_fn and step % self.tcfg["eval_every"] == 0:
                    metrics = self.eval_fn(self.model, self.loss_fn.aligner)
                    tqdm.write(f"[step {step}] eval: {json.dumps(metrics, ensure_ascii=False)}")
                    self._log({"step": step, "eval": metrics})
                if step % self.tcfg["save_every"] == 0:
                    self.save_checkpoint(step)

        pbar.close()
        self.save_checkpoint(step)
