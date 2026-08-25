# -*- coding: utf-8 -*-
"""
SigLIP 损失 + 多卡负例同步（支持各卡 bin 样本数不同）
==========================================================
SigLIP：N×N 相似度矩阵的每个格子是独立二分类：
    logit_ij = τ·cos(f_i, g_j) + b
    L = -Σ log σ(z_ij · logit_ij) / N_query     z_ij = +1(配对) / −1(非配对)
τ（init 10）与 b（init −10）可学习。逐对独立 → 每卡只算【本地 query 行 ×
全局 key 列】，无需全局归一。

多卡负例同步的关键实现（gather-with-grad，变长版）：
- binpack 模式下每个 bin 的图片数不同（如 rank0 有 10 张、rank1 有 9 张），
  NCCL all_gather 要求各卡形状一致 → 先交换各卡样本数 counts，
  零填充到 max_n 再汇集，无效 key 列用 mask 剔除出损失；
- 前向汇集后把本卡位置换回带计算图的真身，反向只切回本卡那份梯度。
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def gather_counts(n_local: int, device) -> list[int]:
    """交换各卡样本数（world 个 int32，通信量可忽略）。"""
    world = dist.get_world_size()
    count_t = torch.tensor([n_local], device=device, dtype=torch.int32)
    count_list = [torch.zeros_like(count_t) for _ in range(world)]
    dist.all_gather(count_list, count_t)
    return [int(c.item()) for c in count_list]


class _GatherWithGrad(torch.autograd.Function):
    """跨卡收集 embedding 且保留梯度（counts 由调用方预先交换并传入）。

    前向：本卡 embedding 尾部零填充到 max_n → all_gather →
          本卡位置替换回带计算图的 padded tensor（他卡份为无梯度副本）。
    反向：只切回本卡那 n_local 行（填充行梯度为零，丢弃）。
    """

    @staticmethod
    def forward(ctx, local_tensor: torch.Tensor, counts: list):
        ctx.rank = dist.get_rank()
        ctx.n_local = local_tensor.shape[0]
        ctx.max_n = max(counts)

        if ctx.n_local < ctx.max_n:
            pad = torch.zeros(ctx.max_n - ctx.n_local, *local_tensor.shape[1:],
                              dtype=local_tensor.dtype, device=local_tensor.device)
            padded = torch.cat([local_tensor, pad], dim=0)
        else:
            padded = local_tensor

        gathered = [torch.zeros_like(padded) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, padded)
        gathered[ctx.rank] = padded
        return torch.cat(gathered, dim=0)          # [world*max_n, D]

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        start = ctx.rank * ctx.max_n
        return grad_output[start:start + ctx.n_local], None   # counts 无需梯度


def _key_mask(counts: list[int], max_n: int, device) -> torch.Tensor:
    """按各卡样本数生成全局 key 有效性掩码 [world*max_n]（1=有效，0=填充）。"""
    mask = torch.zeros(len(counts) * max_n, device=device)
    for r, c in enumerate(counts):
        mask[r * max_n: r * max_n + c] = 1.0
    return mask


class SiglipLoss(nn.Module):
    """SigLIP 成对 sigmoid 损失（各卡只算本地 query 行）。

    forward(img_z, txt_z, logit_scale, logit_bias)：
      - 单卡：本地 N×N 全量；
      - 多卡：gather 全局 key（变长兼容），本地 n 行 × 全局 Ng 列，
        无效 key 列经 mask 剔除，正例位于第 rank*max_n+i 列。
    """

    def __init__(self, use_dist: bool = False):
        super().__init__()
        self.use_dist = use_dist

    def _pairwise_loss(self, query_z, key_all, key_mask, logit_scale, logit_bias,
                       rank, max_n) -> torch.Tensor:
        logits = logit_scale.exp() * query_z @ key_all.t() + logit_bias   # [n, Ng]
        n = logits.shape[0]
        labels = -torch.ones_like(logits)                                 # z=−1 默认负对
        pos_cols = torch.arange(n, device=logits.device) + rank * max_n   # 本卡正例列
        labels[torch.arange(n, device=logits.device), pos_cols] = 1.0

        pair_loss = -F.logsigmoid(labels * logits)                        # [n, Ng]
        pair_loss = pair_loss * key_mask.unsqueeze(0)                     # 剔除填充 key 列
        return pair_loss.sum() / n                                        # 按本地 query 数平均

    def forward(self, img_z, txt_z, logit_scale, logit_bias):
        n = img_z.shape[0]
        if self.use_dist and dist.is_available() and dist.is_initialized() \
                and dist.get_world_size() > 1:
            rank = dist.get_rank()
            counts = gather_counts(n, img_z.device)      # 图文两侧样本数必然一致
            max_n = max(counts)
            # 图/文两侧各自做一次 gather：全局负例（本卡份保留梯度）
            img_all = _GatherWithGrad.apply(img_z, counts)   # [Ng, D]
            txt_all = _GatherWithGrad.apply(txt_z, counts)   # [Ng, D]
            key_mask = _key_mask(counts, max_n, img_z.device)
            loss_i2t = self._pairwise_loss(img_z, txt_all, key_mask,
                                           logit_scale, logit_bias, rank, max_n)
            loss_t2i = self._pairwise_loss(txt_z, img_all, key_mask,
                                           logit_scale, logit_bias, rank, max_n)
        else:
            ones = torch.ones(n, device=img_z.device)
            loss_i2t = self._pairwise_loss(img_z, txt_z, ones, logit_scale, logit_bias, 0, n)
            loss_t2i = self._pairwise_loss(txt_z, img_z, ones, logit_scale, logit_bias, 0, n)
        return (loss_i2t + loss_t2i) / 2.0
