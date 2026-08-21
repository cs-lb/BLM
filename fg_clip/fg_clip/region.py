# -*- coding: utf-8 -*-
"""
区域特征提取与细粒度损失（方案第 4、5 章）
==============================================
区域特征（方案 4.1）：
    不引入额外检测主干，直接在 ViT 的 patch token 上做网格映射 + 均值池化：
    bbox（归一化）→ 映射到 (gh, gw) patch 网格 → 覆盖 patch 取均值 → 区域投影层。
    与全局 embedding 共用同一嵌入空间（独立的 region_proj，merger 前 hidden 维）。

损失（方案第 5 章）：
    L = L_global + λ1 * L_region + λ2 * L_hard
    - L_region：区域-表达式的批内 InfoNCE；
    - L_hard：难负例 margin 排序损失
      L_hard = mean( max(0, m + cos(r, e_hard) - cos(r, e_pos)) )，m=0.2。
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


def hard_negative_margin_loss(region_z: torch.Tensor, pos_z: torch.Tensor,
                              hard_z: torch.Tensor, hard_region_idx: torch.Tensor,
                              margin: float = 0.2) -> torch.Tensor:
    """难负例 margin 排序损失（方案公式）：
        L = mean( max(0, m + cos(r, e_hard) - cos(r, e_pos)) )

    参数：
        region_z:        [M, D] 区域特征（已归一化）
        pos_z:           [M, D] 正例表达式特征（已归一化）
        hard_z:          [ΣK, D] 所有难负例表达式特征（已归一化）
        hard_region_idx: [ΣK] 每条难负例所属的区域下标
    """
    s_pos = (region_z * pos_z).sum(dim=-1)                        # [M]
    s_hard = (hard_z * region_z[hard_region_idx]).sum(dim=-1)     # [ΣK]
    losses = F.relu(margin + s_hard - s_pos[hard_region_idx])
    return losses.mean()
