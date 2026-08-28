# FG-CLIP：细粒度图文对齐（学习版实现）

实现《细粒度图文对齐详细方案.md》的完整链路：**五步数据生产线 + 区域级三损失联合训练**，
复用 `blm_clip` 工程的基座模型（QwenViT + 文本塔）。

## 项目结构

```
fg_clip/
├── configs/fg_config.yaml     # 全部配置（数据生产线 + 训练）
├── fg_data/                   # 数据生产线（方案第3章，五步）
│   ├── dense_caption.py       # Step1: Qwen2.5-VL 生成详细描述（local/dashscope 双后端）
│   ├── expressions.py         # Step2: SpaCy 引用表达式抽取（含正则兜底）
│   ├── grounding.py           # Step3: YOLO-World 定位 + NMS + 置信度>0.4
│   └── hard_negatives.py      # Step4: 颜色/动作/数量/材质 优先级改写难负例
├── fg_clip/
│   └── region.py              # 区域特征（bbox→patch网格池化）+ 区域InfoNCE + 难负例margin损失
├── train_fg.py                # 训练入口：L = L_global + λ1·L_region + λ2·L_hard
└── requirements.txt
```

## 使用流程

```bash
pip install -r requirements.txt
python -m spacy download zh_core_web_sm    # 中文引用表达式抽取模型

# ---- 数据生产线（逐步执行，中间产物都在 data/fg/ 可检查）----
python -m fg_data.dense_caption --limit 100   # 先小批量验证（约1~2分钟/10张，CPU更慢）
python -m fg_data.expressions
python -m fg_data.grounding                   # 首次自动下载 YOLO-World 权重
python -m fg_data.hard_negatives              # 产出最终 fg_train.jsonl

# ---- 训练 ----
python train_fg.py                            # 可从 blm_clip 的 ckpt 继续（改 base_ckpt）
```

## 数据格式（fg_train.jsonl）

```json
{
  "image": "<sha1>.jpg",
  "caption": "一个穿着蓝色T恤的男人在绿色草地上玩红色的球",
  "regions": [
    {"expr": "穿着蓝色T恤的男人", "bbox": [0.31, 0.12, 0.55, 0.86], "conf": 0.62,
     "hard": ["穿着绿色T恤的男人", "穿着蓝色T恤的女人"]}
  ]
}
```

## 与方案文档的对应

| 方案章节 | 实现 | 说明 |
|---|---|---|
| 3.1 详细描述 | `fg_data/dense_caption.py` | cogvlm2 → Qwen2.5-VL-3B（本地）/ qwen-vl-max（API） |
| 3.2 引用表达式 | `fg_data/expressions.py` | SpaCy noun_chunks + 修饰语过滤 + 代词丢弃 |
| 3.3 边界框+NMS+0.4 | `fg_data/grounding.py` | YOLO-World，每表达式保留最高分框，无框丢弃 |
| 3.4 难负例 | `fg_data/hard_negatives.py` | LLM 改写 + 编辑相似度校验（规则词表兜底），每正例 1~3 条 |
| 4.1 区域特征 | `fg_clip/region.py::encode_regions` | merger 前 patch token 网格池化 + region_proj |
| 5.2 区域 InfoNCE | `region_infonce_loss` | λ1=0.5 |
| 5.3 难负例 margin | `hard_negative_margin_loss` | λ2=1.0，margin=0.2 |
| 7 评估回退检查 | `train_fg.py` 周期 eval | 监控全局 hit@5 不低于 FG 训练前 |

## 已知边界（学习版简化）

- 区域特征与全局前向分开执行（多一次视觉前向），`max_regions_per_image` 控制开销；
- 难负例默认 LLM 改写（`hard_neg_backend: llm`，复用 Qwen2.5-VL-3B text-only）+ 结构/长度/编辑相似度三重校验，产出不足自动回退规则词表；工业版可再加 judge 模型做语义级校验；
- grounding 中文表达式需 `translate_to_en: true`（YOLO-World 文本编码器为英文 CLIP）；实测带属性长短语置信度系统性偏低（0.1~0.25），conf 阈值需按数据校准，方案口径 0.4 会筛掉全部正确框；
- 消融实验（去难负例/去区域损失）需自行跑三个配置对比；
- 人工抽查 grounding 质量：`python -m fg_data.visualize --input fg_train.jsonl` 输出标注图到 `data/fg/vis/`。
