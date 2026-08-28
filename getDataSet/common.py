# -*- coding: utf-8 -*-
"""
公共工具：jsonl 读写 + 图片扫描
==================================
被各 step 脚本以「同目录普通导入」方式使用（import common），
不依赖包结构，因此每个 step 都能在 PyCharm 里单独点运行。
"""

import json
from pathlib import Path

from config import IMAGE_DIR, OUT_DIR

# 支持的图片后缀（小写比较）
_IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(image_dir: Path = IMAGE_DIR) -> list[Path]:
    """扫描图片目录，返回按文件名排序的图片路径列表。

    排序是为了让多次运行结果可复现（文件系统遍历顺序不稳定）。
    """
    if not image_dir.exists():
        raise FileNotFoundError(
            f"图片目录不存在：{image_dir}\n"
            f"请把原始图片放进 getDataSet/data/images/ 再运行")
    files = [p for p in image_dir.iterdir()
             if p.is_file() and p.suffix.lower() in _IMG_SUFFIXES]
    if not files:
        raise FileNotFoundError(f"图片目录为空：{image_dir}，请先放入图片")
    return sorted(files, key=lambda p: p.name)


def read_jsonl(path: Path) -> list[dict]:
    """读取 jsonl：每行一个 JSON 对象，返回 dict 列表。"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: Path, items: list[dict]):
    """写 jsonl：自动创建父目录；ensure_ascii=False 保留中文原文。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def out_path(filename: str) -> Path:
    """约定各步骤产物文件名，统一放在 data/out/ 下。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR / filename
