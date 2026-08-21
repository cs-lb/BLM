# -*- coding: utf-8 -*-
"""数据生产线公共工具：jsonl 读写、配置加载、图片路径解析"""

import json
import os

import yaml


def load_config(path: str = "configs/fg_config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: str, items: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def resolve_image(image_field: str, image_root: str) -> str:
    """把 jsonl 中的 image 字段解析为绝对路径。"""
    return image_field if os.path.isabs(image_field) else os.path.join(image_root, image_field)
