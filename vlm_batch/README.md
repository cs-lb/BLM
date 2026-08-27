# vlm_batch：通用批量 VLM 推理工具

候选生成、裁判打标、飞轮 caption 共用的批推理基建：
**vLLM 离线批量 / DashScope API / transformers 三后端 + 缓存断点 + 失败重试**。

## 三个内置任务

| 任务 | 输入 | 输出 | 推荐后端 |
|---|---|---|---|
| `caption` | {"image"} | {"image", "caption"} | vllm 3B/7B（飞轮/FG-CLIP） |
| `candidate` | {"image"} | {"image", "candidates":[{reason_id, prior}]} | dashscope 72B |
| `judge` | {"image", "reason_id", "reason_text"} | {"image", "label", "teacher_score", "confidence", "analysis"} | vllm 7B / 32B-LoRA |

## 用法

```bash
pip install vllm   # 或 dashscope（API 模式）

# 候选生成（72B API）
python run_batch.py --task candidate --backend dashscope --model qwen-vl-max \
    --input risk_images.jsonl --output candidates.jsonl

# 裁判打标（双卡 4090 vLLM）
python run_batch.py --task judge --backend vllm \
    --model Qwen/Qwen2.5-VL-7B-Instruct --tp 2 \
    --input pairs.jsonl --output judged.jsonl

# 中断重跑：同一条命令直接再执行（自动跳过已完成）
# 只重跑失败条目：
python run_batch.py --task judge --backend vllm --model ... --tp 2 \
    --input judged.jsonl.failures.jsonl --output judged.jsonl --retry-failures
```

## 设计要点

- **缓存 key** = 图片 + 任务 + prompt_version（judge 任务含 reason_id）；
  改 prompt 时在 `vlm_batch/tasks.py` 把对应 `prompt_version` +1，旧缓存自动失效；
- **解析校验**：输出非合法 JSON / 缺字段 / judge 无 analysis 一律进
  `<output>.failures.jsonl`（含原始输出前 500 字，便于诊断）；
- **像素预算分级**：caption 用 224×28²（快），候选/裁判用 448×28²（看清细节），
  大图在推理前等比缩小（视觉 token 是成本大头）；
- **产出接下游**：`candidate` 的 pairs 展开格式可直接喂 `judge`；
  `judge` 的输出可直接喂 `blm_triplet/scripts/build_triplets.py` 构建三元组。
