# -*- coding: utf-8 -*-
"""
数据准备脚本：从原始来源 -> blm_clip 训练格式
==================================================
产出：
    <out_dir>/images/          所有图片（sha1 命名，天然去重）
    <out_dir>/train.jsonl      {"image": "<sha1>.jpg", "text": "..."}
    <out_dir>/eval.jsonl       同上（按比例切分）

两种来源模式：
    1) hf     HuggingFace 数据集（图片内嵌在数据集中，如部分中文图文集）
    2) url    parquet/csv 文件，含图片 URL + 文本列，并发下载（LAION 类）

用法示例：
    # HuggingFace 内嵌图片的数据集
    python scripts/prepare_data.py --mode hf --hf-name <dataset> \
        --image-col image --text-col caption --num-samples 50000

    # URL 列表（LAION-zh 子集 parquet）
    python scripts/prepare_data.py --mode url --input laion_zh.parquet \
        --url-col url --text-col text --num-samples 50000
"""

import argparse
import hashlib
import io
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image

# ---------------- 清洗规则 ----------------
MIN_TEXT_LEN = 5          # caption 过短无监督价值（如纯文件名）
MAX_TEXT_LEN = 200        # 过长截断（文本塔 max_len=96 token ≈ 100+ 汉字）
MIN_IMAGE_SIDE = 64       # 过滤小图/图标
MIN_IMAGE_BYTES = 5 * 1024  # 过滤损坏/占位图


def clean_text(text: str) -> str | None:
    """caption 清洗：去空白、长度过滤。不合格返回 None。"""
    if not text:
        return None
    text = " ".join(str(text).split())  # 压缩连续空白/换行
    if not (MIN_TEXT_LEN <= len(text) <= MAX_TEXT_LEN):
        return None
    return text


def process_image(content: bytes) -> tuple[bytes, str] | None:
    """校验并转存为 RGB JPEG 字节；失败返回 None。"""
    if len(content) < MIN_IMAGE_BYTES:
        return None
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        return None
    if min(img.size) < MIN_IMAGE_SIDE:
        return None
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue(), "jpg"


def download_one(url: str, timeout: int = 10) -> bytes | None:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "blm-clip/0.1"})
        return r.content if r.status_code == 200 else None
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description="blm_clip 数据准备")
    p.add_argument("--mode", choices=["hf", "url"], required=True)
    p.add_argument("--out-dir", type=str, default="data")
    p.add_argument("--num-samples", type=int, default=50000)
    p.add_argument("--eval-ratio", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    # hf 模式
    p.add_argument("--hf-name", type=str, default=None)
    p.add_argument("--hf-split", type=str, default="train")
    p.add_argument("--image-col", type=str, default="image")
    p.add_argument("--text-col", type=str, default="caption")
    # url 模式
    p.add_argument("--input", type=str, default=None, help="parquet/csv 路径")
    p.add_argument("--url-col", type=str, default="url")
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args()

    image_dir = os.path.join(args.out_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    # ---------- 1. 拉取原始 (图片字节, 文本) 流 ----------
    def iter_raw():
        if args.mode == "hf":
            from datasets import load_dataset
            ds = load_dataset(args.hf_name, split=args.hf_split, streaming=True)
            for item in ds:
                text = clean_text(item.get(args.text_col))
                if text is None:
                    continue
                try:
                    img = item[args.image_col]
                    if not isinstance(img, Image.Image):
                        img = Image.open(io.BytesIO(img["bytes"]))
                    buf = io.BytesIO()
                    img.convert("RGB").save(buf, format="JPEG", quality=95)
                    yield buf.getvalue(), text
                except Exception:
                    continue
        else:  # url 模式：并发下载（分块提交）
            # 注意：不能一次性把百万级 URL 全部 submit——调用方收够 num_samples 后
            # 会 break，若 future 已全部提交，with 退出时会 shutdown(wait=True)
            # 傻等全部下载完。分块提交保证提前退出时最多等多一个 chunk。
            import pandas as pd
            df = pd.read_parquet(args.input) if args.input.endswith(".parquet") \
                else pd.read_csv(args.input, sep=None, engine="python")
            rows = [(u, clean_text(t)) for u, t in zip(df[args.url_col], df[args.text_col])]
            rows = [(u, t) for u, t in rows if t is not None]
            chunk = args.workers * 20  # 每批提交的 URL 数
            for i in range(0, len(rows), chunk):
                batch = rows[i:i + chunk]
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    futs = {pool.submit(download_one, u): t for u, t in batch}
                    for fut in as_completed(futs):
                        content = fut.result()
                        if content:
                            yield content, futs[fut]

    # ---------- 2. 校验 + 去重（sha1）+ 落盘 ----------
    # 断点续跑：磁盘已有图片的文件名即内容 sha1，预加载后——
    # 重复内容跳过写盘（省 I/O）但仍计入 records（保证 jsonl 完整覆盖已有图片）
    on_disk = set()
    if os.path.isdir(image_dir):
        on_disk = {os.path.splitext(fn)[0] for fn in os.listdir(image_dir)}
        if on_disk:
            print(f"[resume] 复用磁盘已有 {len(on_disk)} 张图片")

    seen = set()
    records = []
    for content, text in iter_raw():
        if len(records) >= args.num_samples:
            break
        digest = hashlib.sha1(content).hexdigest()
        if digest in seen:          # 字节级去重（同一图片不同 URL 只留一份）
            continue
        result = process_image(content)
        if result is None:
            continue
        data, ext = result
        fname = f"{digest}.{ext}"
        if digest not in on_disk:   # 磁盘已有则跳过写盘
            with open(os.path.join(image_dir, fname), "wb") as f:
                f.write(data)
        seen.add(digest)
        records.append({"image": fname, "text": text})
        if len(records) % 5000 == 0:
            print(f"[prepare] 已收集 {len(records)} 条")

    # ---------- 3. 先 shuffle 再切分（评估集与训练集无近邻泄漏）----------
    random.Random(args.seed).shuffle(records)
    n_eval = max(1, int(len(records) * args.eval_ratio))
    eval_recs, train_recs = records[:n_eval], records[n_eval:]

    for name, recs in [("train", train_recs), ("eval", eval_recs)]:
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[done] {path}: {len(recs)} 条1   ")

    print(f"[done] 图片目录: {image_dir}（{len(seen)} 张，已按 sha1 去重）")
    print("提示：首次训练会自动统计 token 数并缓存到 train.jsonl.tokencache.json")


if __name__ == "__main__":
    main()
