# BLM：风控视觉基座复现（学习版）

基于《风控大模型 BLM 预训练》方案的学习版复现工程：
以 QwenViT 为初始化、单阶段图文对比学习（InfoNCE）训练风控视觉基座，
含动态分辨率 packing、数据侧 binpack、细粒度图文对齐（FG-CLIP 式）完整实现。

## 工程组成

| 目录 | 内容 |
|---|---|
| `blm_clip/` | 主工程：QwenViT 双塔 CLIP 训练（动态分辨率 packing、binpack、GradCache、多卡负例同步、检索评估） |
| `fg_clip/` | 细粒度图文对齐：五步数据生产线（dense caption → 引用表达式 → YOLO-World grounding → 难负例）+ 区域级损失 |
| `outputs/文档` | 方案文档（不入库，见 .gitignore） |

## 快速开始

```bash
# 1. 主工程
cd blm_clip && pip install -r requirements.txt
python scripts/prepare_data.py --mode url --input data/raw/train-00000-of-00032.parquet \
    --url-col url --text-col caption --num-samples 100000 --workers 64 --out-dir data
python train.py --config configs/default.yaml          # GPU 上正式训练
python scripts/test_qwenvit_binpack.py                 # 无卡链路验证

# 2. 细粒度对齐
cd fg_clip && pip install -r requirements.txt
python -m fg_data.dense_caption --limit 100
python -m fg_data.expressions && python -m fg_data.grounding && python -m fg_data.hard_negatives
python train_fg.py
```

## 数据来源

- [wanng/wukong100m](https://huggingface.co/datasets/wanng/wukong100m)（CC BY-NC-SA 4.0，仅用于学习研究）
