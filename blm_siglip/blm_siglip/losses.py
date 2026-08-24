# -*- coding: utf-8 -*-
"""
SigLIP 损失 + 多卡负例同步
================================
SigLIP（Sigmoid Loss for Language-Image Pre-training）：
把 N×N 相似度矩阵的**每个格子当作独立二分类**：

    logit_ij = τ · cos(f_i, g_j) + b
    L = -1/N Σ_ij log σ(z_ij · logit_ij)     z_ij = +1(配对) / −1(非配对)

τ（init 10，存 ln10）与 b（init −10）均可学习。负偏置 b 的作用：
矩阵中负对远多于正对，b=−10 让初始 logits 偏负，正对需要"爬坡"，
避免负对在初期被轻易压到 σ≈1 后梯度消失（SigLIP 论文口径）。

与 InfoNCE 的工程差异（选 SigLIP 的核心原因）：
- InfoNCE 的分母是全 batch softmax 归一，多卡必须全局归一；
- SigLIP 逐对独立，loss 可按 (i,j) 分解 → 每卡只算【本地 query 行 ×
  全局 key 列】，无需全局归一，负例同步实现大幅简化。
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class _GatherWithGrad(torch.autograd.Function):
    """跨卡收集 embedding 且保留梯度回传（梯度只取回本卡那一份）。

    前向：all_gather 各卡 embedding 后，把【本卡位置】替换回带计算图的
    原 tensor（他卡份是无梯度数值副本，只当常量 key 用）。
    反向：只把属于本卡样本的梯度切回来——SigLIP 逐对独立，本卡参数
    相关的梯度恰好全在这一块里，数学上与全 batch 单卡严格等价。
    """

    @staticmethod
    def forward(ctx, local_tensor: torch.Tensor) -> torch.Tensor:
        ctx.world_size = dist.get_world_size()
        ctx.rank = dist.get_rank()
        gathered = [torch.zeros_like(local_tensor) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, local_tensor.contiguous())
        gathered[ctx.rank] = local_tensor
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        chunk = grad_output.shape[0] // ctx.world_size
        return grad_output[ctx.rank * chunk:(ctx.rank + 1) * chunk]


def gather_keys(t: torch.Tensor, use_dist: bool) -> torch.Tensor:
    """key 侧（被检索方）跨卡 gather；单卡原样返回。"""
    if use_dist and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        return _GatherWithGrad.apply(t)
    return t


class SiglipLoss(nn.Module):
    """SigLIP 成对 sigmoid 损失。

    调用约定：
        img_z / txt_z 均为【本卡】query/key（[n, D]，已归一化）；
        内部对两侧做 key 维 gather，然后每卡只计算本地 query 行：
            行 = 本地 img query × 全局 txt key（图搜文方向）
            列 = 本地 txt query × 全局 img key（文搜图方向）
        总损失 = 两方向平均。各卡 loss 语义不重叠，DDP 梯度天然正确。
    """

    def __init__(self, use_dist: bool = False):
        super().__init__()
        self.use_dist = use_dist

    def _pairwise_loss(self, query_z: torch.Tensor, key_all: torch.Tensor,
                       logit_scale: torch.Tensor, logit_bias: torch.Tensor,
                       rank: int, chunk: int) -> torch.Tensor:
        """query 是本地 [n, D]；key_all 是全局 [Ng, D]（本卡份带梯度）。"""
        logits = logit_scale.exp() * query_z @ key_all.t() + logit_bias  # [n, Ng]

        n = logits.shape[0]
        labels = -torch.ones_like(logits)                     # 默认 z_ij = −1（负对）
        # 本卡第 i 个 query 的正例位于全局第 rank*chunk+i 列
        pos_cols = torch.arange(n, device=logits.device) + rank * chunk
        labels[torch.arange(n, device=logits.device), pos_cols] = 1.0

        # -log σ(z·logit)，按本卡 query 的 pair 总数平均（SigLIP 论文口径 /N）
        return -F.logsigmoid(labels * logits).sum() / n

    def forward(
        self,
        img_z: torch.Tensor,        # [n, D] 本卡图 embedding（已归一化）
        txt_z: torch.Tensor,        # [n, D] 本卡文 embedding（已归一化）
        logit_scale: torch.Tensor,  # 标量参数（存 ln τ）
        logit_bias: torch.Tensor,   # 标量参数（b，init −10）
    ) -> torch.Tensor:
        n = img_z.shape[0]
        if self.use_dist and dist.is_initialized() and dist.get_world_size() > 1:
            rank, world = dist.get_rank(), dist.get_world_size()
            assert n * world == gather_keys(img_z, True).shape[0], "各卡 batch 不一致"
            chunk = n
        else:
            rank, chunk = 0, n

        # key 维 gather：全局负例（本卡份保留梯度）
        img_all = gather_keys(img_z, self.use_dist)   # [Ng, D]
        txt_all = gather_keys(txt_z, self.use_dist)   # [Ng, D]

        # 两个方向：本地 query × 全局 key（逐对独立，无需全局归一）
        loss_i2t = self._pairwise_loss(img_z, txt_all, logit_scale, logit_bias, rank, chunk)
        loss_t2i = self._pairwise_loss(txt_z, img_all, logit_scale, logit_bias, rank, chunk)
        return (loss_i2t + loss_t2i) / 2.0
