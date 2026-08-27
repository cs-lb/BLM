# -*- coding: utf-8 -*-
"""
缓存与断点：以 (image, task, prompt_version) 为 key 的结果缓存
================================================================
- 输出文件 append 写入，启动时先扫描已完成 key，跳过不重复推理；
- 解析失败的记录进 failures.jsonl（带原始输出，便于诊断与重跑）；
- 换 prompt 时把 tasks.py 里对应 prompt_version +1，旧缓存自动失效。
"""

import hashlib
import json
import os


def make_key(record: dict, task: str, prompt_version: str) -> str:
    """稳定 key：图片标识 + 任务 + prompt 版本 + 记录级区分字段。"""
    base = f"{record.get('image','')}|{task}|{prompt_version}"
    # judge 任务同一图有多条理由，需把理由纳入 key
    if "reason_id" in record:
        base += f"|{record['reason_id']}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


class ResultCache:
    """基于输出 jsonl 的断点缓存。"""

    def __init__(self, output_path: str, task: str, prompt_version: str):
        self.output_path = output_path
        self.task = task
        self.prompt_version = prompt_version
        self.done: set[str] = set()
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if "_key" in rec:
                            self.done.add(rec["_key"])
                    except json.JSONDecodeError:
                        continue
        if self.done:
            print(f"[cache] 命中已完成 {len(self.done)} 条，跳过重算")

    def is_done(self, record: dict) -> bool:
        return make_key(record, self.task, self.prompt_version) in self.done

    def append(self, record: dict, result: dict):
        key = make_key(record, self.task, self.prompt_version)
        row = {**result, "_key": key, "_task": self.task,
               "_pv": self.prompt_version}
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.done.add(key)


def log_failure(failures_path: str, record: dict, raw: str, reason: str):
    row = {"record": record, "raw_output": raw[:500], "reason": reason}
    with open(failures_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
