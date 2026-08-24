# BLM SigLIP：风控视觉基座预训练（8×H800 正式版）

架构：**Qwen2.5-VL-7B 视觉塔 + BGE-M3 文本塔 + SigLIP 损失**，
数据侧 binpack（在/离线），训练侧 8 卡 DDP + 多卡负例同步。
配套方案文档：`outputs/视觉基座预训练完整方案.md`。

## 与 blm_clip 的关系

`blm_clip` 是学习版（InfoNCE + 从零文本塔 + GradCache）；本工程是**正式版**。
数据层（smart_resize / packing / binpack / collator）经 `sys.path` 复用 blm_clip，
本工程只承载差异部分：SigLIP 损失、BGE-M3 文本塔、7B 视觉权重抽取、DDP 训练。

## 项目结构

```
blm_siglip/
├── configs/default.yaml         # 8×H800 正式配置
├── blm_siglip/
│   ├── models.py                # QwenViT-7B 视觉塔 + BGE-M3 文本塔 + SigLIP 双标量
│   ├── losses.py                # SigLIP 损失 + gather-with-grad 负例同步
│   └── engine.py                # Trainer（DDP/bf16/判别式LR）+ Evaluator（hit@5）
├── scripts/
│   └── extract_qwenvit.py       # 从 7B 全模抽取视觉塔权重（15GB → 1.4GB，秒级加载）
├── train.py                     # 训练入口（torchrun --nproc_per_node=8）
└── eval.py                      # 独立评估入口
```

## 使用流程

```bash
pip install -r ../blm_clip/requirements.txt   # 依赖与 blm_clip 相同

# 0. 一次性：抽取视觉塔权重（之后训练秒级加载）
python scripts/extract_qwenvit.py --model Qwen/Qwen2.5-VL-7B-Instruct \
    --out assets/qwenvit_7b_visual.pt

# 1. 数据（复用 blm_clip 管线）：公共:业务 = 8:2 混合 shuffle 后放 data/
#    下载/清洗/去重：python ../blm_clip/scripts/prepare_data.py ...

# 2. 冒烟（单卡 + timm 备用塔：把 yaml 的 vision_type 改为 timm）
python train.py --config configs/default.yaml --max-steps 200

# 3. 8 卡正式训练（binpack 已在 yaml 开启）
torchrun --nproc_per_node=8 train.py --config configs/default.yaml

# 4. 评估（i2t_R@5 即业务口径 hit@5）
python eval.py --config configs/default.yaml --ckpt outputs/blm_vit_v1/ckpt_step20000.pt
```

## 关键实现说明

| 机制 | 位置 | 说明 |
|---|---|---|
| SigLIP 损失 | `losses.py::SiglipLoss` | 逐对 sigmoid，每卡只算本地 query 行 × 全局 key 列，无需全局归一 |
| 负例同步 | `losses.py::_GatherWithGrad` | all_gather 全局 embedding，梯度只回本卡份，数学等价全 batch 单卡 |
| SigLIP 双标量 | `models.py::BLMSiglipModel` | τ init 10 / b init −10（负偏置防初期梯度消失），免 weight_decay |
| 视觉权重抽取 | `scripts/extract_qwenvit.py` | 避免每次加载 15GB 全模 |
| BGE-M3 池化 | `models.py::BgeM3TextTower` | CLS 位（dense 检索标准取法），SentencePiece 词表勿与 Qwen 混用 |
| merger 边界 | `models.py::QwenViTVisionTower` | pooler_output（merger 后）切分池化；区域特征需求才用 merger 前 |
| binpack | 复用 `blm_clip/binpack.py` | 在线（业务）/离线（开源）两策略，DDP 各 rank 交错取 bin |
