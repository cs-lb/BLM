# -*- coding: utf-8 -*-
"""
损失模块：对称 InfoNCE（CLIP 损失）+ 多卡负例同步
=====================================================
对比学习公式（batch 内 N 对图文，相似度 s_ij = cos(f_i, g_j)）：

    L_i2t = -1/N Σ_i log [ exp(s_ii/τ) / Σ_j exp(s_ij/τ) ]
    L     = (L_i2t + L_t2i) / 2

负例越多对比学习效果越好。多卡训练时通过 all_gather 把各卡 embedding
汇集后再算相似度矩阵，使有效负例数 = 全局 batch（对应 open_clip 的
--gather-with-grad；工业版的 bidir 通信是对这一步的通信优化）。
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class _GatherWithGrad(torch.autograd.Function):
    """跨卡收集 embedding 且保留梯度回传（梯度只取回本卡那一份）。"""

    @staticmethod
    def forward(ctx, local_tensor: torch.Tensor) -> torch.Tensor:
        ctx.world_size = dist.get_world_size()
        ctx.rank = dist.get_rank()
        gathered = [torch.zeros_like(local_tensor) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, local_tensor.contiguous())
        # 本卡位置保留原 tensor（带计算图），其余为无梯度的副本
        gathered[ctx.rank] = local_tensor
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # 只把属于本卡样本的梯度切回来（其余卡的梯度由它们各自回传）
        chunk = grad_output.shape[0] // ctx.world_size
        return grad_output[ctx.rank * chunk:(ctx.rank + 1) * chunk]


def maybe_gather(t: torch.Tensor, use_dist: bool) -> torch.Tensor:
    """单卡原样返回；多卡则 gather 全局负例。"""
    if use_dist and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        return _GatherWithGrad.apply(t)
    return t


class ClipLoss(nn.Module):
    """对称 InfoNCE。

    Args:
        use_dist: 是否启用多卡负例同步（torchrun 启动时置 True）。
    """

    def __init__(self, use_dist: bool = False):
        super().__init__()
        self.use_dist = use_dist

    def forward(
        self,
        image_embeds: torch.Tensor,   # [N, D]，已 L2 归一化
        text_embeds: torch.Tensor,    # [N, D]，已 L2 归一化
        logit_scale: torch.Tensor,    # 标量 = exp(log τ_inv)，最大 100
    ) -> torch.Tensor:
        img_all = maybe_gather(image_embeds, self.use_dist)
        txt_all = maybe_gather(text_embeds, self.use_dist)

        # 相似度矩阵 [N_global, N_global]，对角线为正样本
        logits = logit_scale * img_all @ txt_all.t()
        labels = torch.arange(logits.shape[0], device=logits.device)

        loss_i2t = F.cross_entropy(logits, labels)        # 图搜文
        loss_t2i = F.cross_entropy(logits.t(), labels)    # 文搜图
        return (loss_i2t + loss_t2i) / 2.0
