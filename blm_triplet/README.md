# BLM Triplet：v1→v2 三元损失 + Reranker 蒸馏

对应流程图的 v1→v2 阶段：在 BLM ViT v1（SigLIP 预训练）基础上，
用**人工/裁判标注的正负样本对 + 难负例挖掘 + 三元损失 + Reranker 蒸馏**
精修图-文相似度能力（目标 hit@5 97% 口径）。

## 方案设计（评审修正版）

### 双损失互补

```
L = λ_t · L_triplet(batch-hard, margin=0.2)  +  λ_d · L_distill(MSE)
    └─ 相对排序：正例比最难负例近 margin      └─ 绝对标定：学生分对齐教师分
```

| 损失 | 解决什么 | 缺了会怎样 |
|---|---|---|
| 三元损失 | 难负例之间的精细相对距离 | 候选间排序粗糙，"像但不对"拉不开 |
| 蒸馏（σ(α·cos+β) MSE） | "比烂"漏洞：绝对相似度无锚点 | 正负整体漂移到低分区，阈值召回失效 |

### 评审修正的两个坑

1. **量纲错配**：教师分（[0,1]）与学生余弦（[−1,1]）不能直接 MSE →
   学生侧加可学习仿射 `σ(α·cos+β)`（α init 5、β init −1，见 `losses.py::ScoreAligner`）；
2. **教师选型**：BGE-Reranker-V2-M3 是**纯文本** cross-encoder，无法给图-文对打分 →
   教师必须是多模态模型（Judged MLLM v2 / 分类 MLLM），且**离线预计算**烘进数据
   （`teacher_score` 字段），训练零在线推理成本。

## 数据链路

```
Judged MLLM 打标（相关/不相关 + teacher_score + confidence）
  → scripts/build_triplets.py（按图聚合、难负例优先 top-K）
  → triplets.jsonl（anchor-candidates 结构）
  → train.py（从 v1 ckpt 继续训练）
```

```bash
# 1. 构建三元组（每图 2 正 + 4 难负例）
python scripts/build_triplets.py --input judged.jsonl --out data/triplets.jsonl \
    --pos-per-image 2 --neg-per-image 4

# 2. 从 v1 ckpt 继续训练
python train.py --config configs/default.yaml --v1-ckpt <blm_siglip的ckpt>
```

## 项目结构

```
blm_triplet/
├── blm_triplet/
│   ├── losses.py      # TripletLoss(batch-hard) + ScoreAligner(σ(αcos+β)) + DistillMSE
│   ├── data.py        # anchor-candidates 数据集与 collator（复用图像 packing）
│   └── engine.py      # 训练循环（双损失、α/β 轨迹入 metrics.jsonl）
├── scripts/build_triplets.py
├── configs/default.yaml
└── train.py
```

## 监控要点（metrics.jsonl）

- `l_triplet` → 应降到 margin 以下（相对约束逐渐满足）；
- `l_distill` → 降到 0.05 以下（绝对分对齐教师）；
- `alpha`/`beta` 轨迹 → α 应增大（学生余弦被拉开），β 向教师先验收敛；
- 评估：hit@5（图搜理由）+ **召回阈值稳定性**（蒸馏的直接收益，v1 时阈值会漂）。
