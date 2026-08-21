# -*- coding: utf-8 -*-
"""
GradCache：单卡实现超大有效 batch 的对比学习
================================================
问题：InfoNCE 的效果强依赖 batch 内负例数（CLIP 用 32k），
     但单卡显存一次只能放下几十~几百对（动态分辨率下更少）。

思路（分块缓存梯度，三倍显存换 batch）：
    Step 1 缓存：逐 chunk 无梯度前向，得到全部 embedding（不带计算图）；
    Step 2 损失：拼接全量 embedding 计算 InfoNCE，backward，
             得到「每个 embedding 的梯度」dL/dz（温度参数梯度也在此步累积）；
    Step 3 重算：再逐 chunk 带梯度前向，用 torch.autograd.backward(z, dL/dz)
             把缓存的梯度"注入"各 chunk 的计算图。

显存占用 = 1 个 chunk 的计算图 + 全量 embedding（很小），
等效 batch = 全部 chunk 之和。代价是约 2 倍前向计算。
"""

from typing import Dict, List

import torch


class GradCache:
    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn

    def _split_batch(self, batch: Dict, chunk_size: int) -> List[Dict]:
        """把 batch 按样本维切成若干 chunk。"""
        n = batch["input_ids"].shape[0]
        chunks = []
        for s in range(0, n, chunk_size):
            e = min(s + chunk_size, n)
            img_inputs = batch["image_inputs"]
            if isinstance(img_inputs, (list, tuple)):
                # qwenvit 路径：pixel_values 是 packing 的，需按 grid_thw 的
                # patch 数切分，保持每 chunk 内 pixel 与 grid 对齐
                pixel, grid = img_inputs
                counts = (grid[:, 0] * grid[:, 1] * grid[:, 2]).tolist()
                offsets = [0]
                for c in counts:
                    offsets.append(offsets[-1] + c)
                p0, p1 = offsets[s], offsets[e]
                img_chunk = (pixel[p0:p1], grid[s:e])
            else:
                img_chunk = img_inputs[s:e]
            chunks.append({
                "image_inputs": img_chunk,
                "input_ids": batch["input_ids"][s:e],
                "attention_mask": batch["attention_mask"][s:e],
            })
        return chunks

    @torch.no_grad()
    def _forward_no_grad(self, chunk: Dict):
        self.model.eval()   # 缓存阶段：关 dropout 等随机性，保证与重算阶段一致
        return self.model(chunk)

    def step(self, batch: Dict, chunk_size: int) -> float:
        """执行一次 GradCache 更新（调用方需已 zero_grad）。"""
        chunks = self._split_batch(batch, chunk_size)

        # ---- Step 1：无梯度缓存全部 embedding ----
        img_cache, txt_cache = [], []
        logit_scale = None
        for c in chunks:
            iz, tz, logit_scale = self._forward_no_grad(c)
            # detach 后作为叶子节点，Step 2 的 backward 会把梯度写进 .grad
            img_cache.append(iz.detach().requires_grad_(True))
            txt_cache.append(tz.detach().requires_grad_(True))

        self.model.train()

        # ---- Step 2：全量 embedding 上算损失并取 embedding 梯度 ----
        img_all = torch.cat(img_cache, dim=0)
        txt_all = torch.cat(txt_cache, dim=0)
        loss = self.loss_fn(img_all, txt_all, logit_scale)
        loss.backward()   # 填充各 cache 的 .grad 与 logit_scale 的梯度

        img_grads = [c.grad for c in img_cache]
        txt_grads = [c.grad for c in txt_cache]

        # ---- Step 3：逐 chunk 带梯度重算，注入缓存梯度 ----
        for c, ig, tg in zip(chunks, img_grads, txt_grads):
            iz, tz, _ = self.model(c)
            torch.autograd.backward([iz, tz], grad_tensors=[ig, tg])

        return loss.item()
