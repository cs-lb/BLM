# -*- coding: utf-8 -*-
"""
区域特征提取与细粒度损失（方案第 4、5 章）
==============================================
区域特征（方案 4.1）：
    不引入额外检测主干，直接在 ViT 的 patch token 上做网格映射 + 均值池化：
    bbox（归一化）→ 映射到 (gh, gw) patch 网格 → 覆盖 patch 取均值 → 区域投影层。
    与全局 embedding 共用同一嵌入空间（独立的 region_proj，merger 前 hidden 维）。

损失（FG-CLIP 论文版，arXiv 2505.05071）：
    L = L_global + α * L_region + β * L_hard，α=0.1，β=0.5
    - L_region：区域-表达式的批内对称 InfoNCE（式 3）；
    - L_hard：每区域 M 路单向 softmax 分类（式 4，1 正 + M-1 改写难负例）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------
# 区域特征提取
# ---------------------------------------------------------------
def _pool_bbox(feat_map: torch.Tensor, bbox: list[float]) -> torch.Tensor:
    """在 [gh, gw, D] 的 patch 特征图上按归一化 bbox 做矩形均值池化。"""
    gh, gw = feat_map.shape[0], feat_map.shape[1]
    x1 = max(0, min(int(bbox[0] * gw), gw - 1))
    y1 = max(0, min(int(bbox[1] * gh), gh - 1))
    x2 = max(x1 + 1, min(int(bbox[2] * gw) + 1, gw))
    y2 = max(y1 + 1, min(int(bbox[3] * gh) + 1, gh))
    return feat_map[y1:y2, x1:x2].mean(dim=(0, 1))          # [D]


def encode_regions(model, image_inputs, regions_per_image: list[list[list[float]]],
                   region_proj: nn.Module) -> torch.Tensor:
    """统一入口：对 batch 内每图的若干 bbox 提取区域特征。

    参数：
        image_inputs:       qwenvit -> (pixel_values, grid_thw)；timm -> [B,3,H,W]
        regions_per_image:  [B][R_i][4]，每图的归一化 bbox 列表
    返回：[Σ R_i, embed_dim]（未归一化），顺序与 regions_per_image 展平一致
    """
    if isinstance(image_inputs, (list, tuple)):   # qwenvit 路径
        pixel_values, grid_thw = image_inputs
        tower = model.visual.visual
        out = tower(pixel_values, grid_thw=grid_thw)
        tokens = out.last_hidden_state if not torch.is_tensor(out) else out
        patch_counts = (grid_thw[:, 1] * grid_thw[:, 2]).tolist()
        per_image = torch.split(tokens, patch_counts, dim=0)

        feats = []
        for tokens_i, (_, gh, gw), bboxes in zip(per_image, grid_thw.tolist(), regions_per_image):
            feat_map = tokens_i.view(gh, gw, -1)
            feats.extend(_pool_bbox(feat_map, b) for b in bboxes)
    else:                                          # timm 路径（固定 224，14x14 网格）
        feats_tokens = model.visual.vit.forward_features(image_inputs)  # [B, 1+N, D]
        patch = feats_tokens[:, 1:, :]                                # 去 CLS
        gh = gw = int(patch.shape[1] ** 0.5)
        feats = []
        for tokens_i, bboxes in zip(patch, regions_per_image):
            feat_map = tokens_i.view(gh, gw, -1)
            feats.extend(_pool_bbox(feat_map, b) for b in bboxes)

    stacked = torch.stack(feats, dim=0)
    return region_proj(stacked.to(region_proj.weight.dtype))


# ---------------------------------------------------------------
# 细粒度损失（方案 5.2 / 5.3）
# ---------------------------------------------------------------
def region_infonce_loss(region_z: torch.Tensor, expr_z: torch.Tensor,
                        logit_scale: torch.Tensor) -> torch.Tensor:
    """区域-表达式批内对称 InfoNCE（输入需已 L2 归一化）。"""
    logits = logit_scale.exp() * region_z @ expr_z.t()          # [M, M]
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def fgclip_hard_loss(region_z: torch.Tensor, cand_z: torch.Tensor,
                     cand_counts: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """难负例 M 路 softmax 分类损失（FG-CLIP 论文式 4，单向）：
        L = -(1/K) Σ_i log [ exp(s(r_i, l_i,1)/τ) / Σ_j exp(s(r_i, l_i,j)/τ) ]

    每个区域的 M_i 条候选描述（第 0 条正例 + M_i-1 条属性改写难负例）放进
    同一个 softmax 竞争。相对 margin hinge 的优势：
      - 正例与所有难负例持续竞争，每条难负例都贡献梯度（hinge 分对即零梯度）；
      - 分数经同一 τ 标定，跨样本可比（缓解"比烂"/校准问题）。

    参数：
        region_z:    [K, D] 区域特征（已归一化）
        cand_z:      [ΣM_i, D] 候选描述特征，按区域连续排布，每区第 0 条为正例
        cand_counts: [K] 每个区域的候选数 M_i
        logit_scale: 共享温度（与全局/区域 InfoNCE 同一个 logit_scale）
    """
    tau = logit_scale.exp()
    losses, offset = [], 0
    for i, m in enumerate(cand_counts.tolist()):
        logits = tau * (cand_z[offset:offset + m] @ region_z[i])      # [M_i]
        target = torch.zeros(1, dtype=torch.long, device=logits.device)
        losses.append(F.cross_entropy(logits.unsqueeze(0), target))
        offset += m
    return torch.stack(losses).mean()
