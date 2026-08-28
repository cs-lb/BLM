# -*- coding: utf-8 -*-
"""
三元损失 + Reranker 蒸馏损失
================================
设计（对应 v1→v2 阶段）：
  L = λ_triplet · L_triplet + λ_distill · L_distill

- L_triplet（相对排序）：batch-hard 三元损失
    L = mean( relu(margin + cos(a, n_hard) − cos(a, p)) )
  每个正例配当前最难负例（相似度最高者），只约束相对距离；
- L_distill（绝对标定）：修"比烂"漏洞
    学生分 σ(α·cos + β)（α、β 可学习仿射，解决量纲错配）；
    教师分 = 分类 MLLM 离线预计算的 relevance ∈ [0,1]（烘进数据）；
    L = mean( w_conf · (student − teacher)² )，w_conf 为教师置信度权重。

注意（评审结论）：
- 教师必须是多模态模型（Judged MLLM / 分类 MLLM）。BGE-Reranker-V2-M3
  是纯文本 cross-encoder，无法直接给图-文对打分（见 README 评审节）。
- 两个损失缺一不可：三元管相对排序，蒸馏管绝对标定。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """batch-hard 三元损失。

    输入按 anchor 分组的候选相似度：
        sims:    [n_anchor, K]  anchor 与各候选文本的余弦相似度（已归一化后点积）
        pos_mask: [n_anchor, K]  1=该候选是正例（相关），0=负例（不相关）
    对每个 anchor：s_pos = 正例均值相似度；s_hard = 负例中的最大相似度（最难负例）
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, sims: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
        neg_mask = 1.0 - pos_mask
        # 正例均值（多正例时取平均，稳定梯度）
        pos_cnt = pos_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        s_pos = (sims * pos_mask).sum(dim=1) / pos_cnt.squeeze(1)          # [A]
        # 最难负例：负例位置取最大值，正例位置置 -inf 排除
        s_hard = (sims + (1.0 - neg_mask) * (-1e9)).max(dim=1).values      # [A]
        loss = F.relu(self.margin + s_hard - s_pos)
        # 有 anchor 可能全正/全负（无有效对），用掩码剔除
        valid = (pos_mask.sum(1) > 0) & (neg_mask.sum(1) > 0)
        return loss[valid].mean() if valid.any() else sims.sum() * 0.0


class ScoreAligner(nn.Module):
    """学生分数的仿射标定层：σ(α·cos + β)。

    解决教师分（[0,1] 相关性）与学生分（[-1,1] 余弦）的量纲错配。
    α init 5.0（把余弦拉开）、β init −1.0（对齐初始先验偏低），均可学习。
    """

    def __init__(self, alpha: float = 5.0, beta: float = -1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.beta = nn.Parameter(torch.tensor(float(beta)))

    def forward(self, cos_sim: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.alpha * cos_sim + self.beta)             # ∈ (0,1)


class DistillMSELoss(nn.Module):
    """Reranker 蒸馏：学生分对齐教师分（MSE，置信度加权）。

        teacher_scores: [A, K] ∈ [0,1]（分类 MLLM 离线预计算）
        conf_weights:   [A, K]（教师置信度；低置信的样本降权，防错误传染）
    """

    def __init__(self):
        super().__init__()

    def forward(self, student_scores: torch.Tensor, teacher_scores: torch.Tensor,
                conf_weights: torch.Tensor = None) -> torch.Tensor:
        se = (student_scores - teacher_scores) ** 2
        if conf_weights is not None:
            se = se * conf_weights
            return se.sum() / conf_weights.sum().clamp(min=1e-6)
        return se.mean()


class TripletDistillLoss(nn.Module):
    """组合损失：L = λ_t · Triplet + λ_d · Distill（输入为候选级相似度矩阵）。"""

    def __init__(self, margin=0.2, lambda_triplet=1.0, lambda_distill=0.5):
        super().__init__()
        self.triplet = TripletLoss(margin)
        self.aligner = ScoreAligner()
        self.distill = DistillMSELoss()
        self.lambda_triplet = lambda_triplet
        self.lambda_distill = lambda_distill

    def forward(self, sims, pos_mask, teacher_scores, conf_weights=None):
        l_t = self.triplet(sims, pos_mask)
        student = self.aligner(sims)                       # σ(α·cos+β) ∈ (0,1)
        l_d = self.distill(student, teacher_scores, conf_weights)
        total = self.lambda_triplet * l_t + self.lambda_distill * l_d
        return total, {"l_triplet": l_t.item(), "l_distill": l_d.item(),
                       "alpha": self.aligner.alpha.item(), "beta": self.aligner.beta.item()}
