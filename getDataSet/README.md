# getDataSet：细粒度图文对齐数据生产线（批量生产版）

一个流程对应一个脚本，PyCharm 里打开文件**直接点运行按钮**即可（无相对导入、路径自解析，与工作目录无关）。LLM 推理全部使用 **vLLM** 批量加速。

## 目录结构

```
getDataSet/
├── config.py                  # 全部配置（模型/阈值/路径），只改这一个文件
├── common.py                  # jsonl 读写 + 图片扫描
├── step1_dense_caption.py     # Qwen2.5-VL-3B + vLLM 批量生成详细描述
├── step2_expressions.py       # 引用表达式抽取（spaCy 探测 + 正则兜底）
├── step3_grounding.py         # YOLO-World 定位（自动中译英送检，conf=0.1）
├── step4_hard_negatives.py    # LLM 批量改写难负例 + 三重校验 + 规则兜底
├── step5_visualize.py         # （可选）画标注框人工抽查
└── data/
    ├── images/                # ★ 把原始图片放这里（jpg/png/jpeg）
    └── out/                   # 各步骤产物（captions/expressions/regions/fg_train.jsonl/vis）
```

## 环境与依赖

```bash
pip install vllm transformers ultralytics spacy pillow tqdm
python -m spacy download zh_core_web_sm   # 可选；不装会自动降级为正则抽取
```

**注意**：vLLM 需要 **Linux + NVIDIA GPU**。首次运行自动下载 Qwen2.5-VL-3B-Instruct（约 7GB）、yolov8s-worldv2.pt、Helsinki 翻译模型（约 300MB）。

## 使用流程（按顺序点运行）

1. 把原始图片放入 `data/images/`
2. `step1_dense_caption.py` → `data/out/captions.jsonl`
3. `step2_expressions.py` → `expressions.jsonl`
4. `step3_grounding.py` → `regions.jsonl`
5. `step4_hard_negatives.py` → **`fg_train.jsonl`（最终训练数据）**
6. （可选）`step5_visualize.py` → `out/vis/` 标注图抽查

## 最终数据格式

```json
{"image": "xxx.jpg", "caption": "详细描述",
 "regions": [{"expr": "一朵粉色的花", "bbox": [0.53, 0.19, 0.83, 0.68], "conf": 0.18,
              "hard": ["一朵黑色的花", "一朵绿色的花"]}]}
```

## 关键口径（来自 fg_clip 链路实测，勿轻易改回）

- `CONF_THRESHOLD = 0.1`：YOLO-World 对带属性长短语打分系统性偏低（正确框常 0.1~0.25），0.4 会筛掉全部正确框；
- `TRANSLATE_TO_EN = True`：YOLO-World 文本编码器是英文 CLIP，中文直检几乎零检出；
- 难负例校验：`SequenceMatcher ≥ 0.5` 拦截 LLM 整句重写；产出不足用规则词表补齐。
