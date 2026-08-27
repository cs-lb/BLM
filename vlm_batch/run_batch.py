# -*- coding: utf-8 -*-
"""
通用批量 VLM 推理工具：缓存 + 断点 + 失败重试
================================================
用法示例：

  # 1) 候选拒绝理由生成（72B API）
  python run_batch.py --task candidate --backend dashscope --model qwen-vl-max \
      --input risk_images.jsonl --output candidates.jsonl

  # 2) 裁判打标（本地 7B vLLM 双卡）
  python run_batch.py --task judge --backend vllm \
      --model Qwen/Qwen2.5-VL-7B-Instruct --tp 2 \
      --input pairs.jsonl --output judged.jsonl

  # 3) 飞轮 caption（本地 3B vLLM）
  python run_batch.py --task caption --backend vllm \
      --model Qwen/Qwen2.5-VL-3B-Instruct \
      --input biz_images.jsonl --output captions.jsonl

输入 jsonl 通用格式（每行一条）：
    caption:   {"image": "图片绝对路径"}
    candidate: {"image": "图片绝对路径"}
    judge:     {"image": "...", "reason_id": "R03", "reason_text": "..."}

断点与缓存：
  - 输出文件 append 写入；重启自动跳过已完成条目（key=图+任务+prompt版本）；
  - 解析失败进 <output>.failures.jsonl，--retry-failures 可只重跑失败条目。
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vlm_batch.backends import build_backend
from vlm_batch.cache import ResultCache, log_failure
from vlm_batch.tasks import TASKS


def read_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    p = argparse.ArgumentParser(description="批量 VLM 推理（缓存+断点）")
    p.add_argument("--task", choices=list(TASKS.keys()), required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--backend", choices=["vllm", "dashscope", "hf"], default="vllm")
    p.add_argument("--model", required=True)
    p.add_argument("--tp", type=int, default=1, help="vLLM 张量并行卡数")
    p.add_argument("--workers", type=int, default=8, help="API 并发数")
    p.add_argument("--batch-size", type=int, default=64, help="每批推理条数")
    p.add_argument("--limit", type=int, default=None, help="只处理前 N 条（调试）")
    p.add_argument("--retry-failures", action="store_true",
                   help="input 传 failures.jsonl，只重跑失败条目")
    args = p.parse_args()

    task = TASKS[args.task]
    backend = build_backend(args.backend, args.model, tp=args.tp, workers=args.workers)
    cache = ResultCache(args.output, args.task, task["prompt_version"])
    failures_path = args.output + ".failures.jsonl"

    # ---- 读取输入（retry 模式从 failures 提取原 record）----
    records = read_jsonl(args.input)
    if args.retry_failures:
        records = [r["record"] for r in records]
    if args.limit:
        records = records[: args.limit]

    todo = [r for r in records if not cache.is_done(r)]
    print(f"[plan] 总数 {len(records)}，已完成 {len(records) - len(todo)}，"
          f"待推理 {len(todo)}")

    # ---- 分批推理 ----
    done, failed = 0, 0
    for i in range(0, len(todo), args.batch_size):
        chunk = todo[i:i + args.batch_size]
        items = [{"image": r["image"], "prompt": task["build"](r)} for r in chunk]
        try:
            raws = backend.infer(items, task["max_tokens"], task["temperature"],
                                 task["max_pixels"])
        except Exception as e:
            for r in chunk:
                log_failure(failures_path, r, "", f"backend_error: {e}")
            failed += len(chunk)
            continue

        for rec, raw in zip(chunk, raws):
            result = task["parse"](rec, raw) if raw else None
            if result is None:
                log_failure(failures_path, rec, raw, "parse_failed")
                failed += 1
            else:
                cache.append(rec, result)
                done += 1
        print(f"[progress] {min(i + args.batch_size, len(todo))}/{len(todo)}"
              f"  成功 {done}  失败 {failed}")

    print(f"[done] 输出 {args.output}（成功 {done}）；"
          f"失败 {failed} 条见 {failures_path}")


if __name__ == "__main__":
    main()
