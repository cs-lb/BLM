# -*- coding: utf-8 -*-
"""
Step 3：获取边界框（Grounding）
==================================
方案要点：用 YOLO-World（开放词汇检测）为每个引用表达式在图中定位边界框；
NMS 去重（模型内置，iou=0.5）；只保留置信度 > 0.4 的框；
表达式无框检出时**丢弃该表达式**（不退化用全图框，避免噪声区域监督）。

用法：
    python -m fg_data.grounding --config configs/fg_config.yaml [--limit 100]

输出：data/fg/regions.jsonl
    {"image": ..., "caption": ...,
     "regions": [{"expr": "...", "bbox": [x1,y1,x2,y2], "conf": 0.62}, ...]}
    （bbox 为归一化 [0,1] 坐标，训练时按实际缩放尺寸换算）
"""

import argparse
import os

from PIL import Image
from tqdm import tqdm

from .common import load_config, read_jsonl, resolve_image, write_jsonl


def ground_one_image(model, image_path: str, expressions: list[str],
                     conf_thres: float, nms_iou: float) -> list[dict]:
    """对一张图的所有表达式做开放词汇检测，返回 [{expr, bbox, conf}]。

    YOLO-World 的 set_classes 设置开放词表后，predict 的输出里
    每个检测框的 cls 对应词表下标；模型内置 NMS（iou 参数）。
    每个表达式只保留置信度最高的一个框（方案：每物体一个最准的框）。
    """
    model.set_classes(expressions)
    results = model.predict(image_path, conf=conf_thres, iou=nms_iou, verbose=False)[0]

    W, H = results.orig_shape[1], results.orig_shape[0]
    best: dict[int, dict] = {}   # cls_idx -> 最高分框
    for box in results.boxes:
        cls_idx = int(box.cls.item())
        conf = float(box.conf.item())
        if cls_idx not in best or conf > best[cls_idx]["conf"]:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            best[cls_idx] = {
                "expr": expressions[cls_idx],
                "bbox": [x1 / W, y1 / H, x2 / W, y2 / H],   # 归一化坐标
                "conf": round(conf, 4),
            }
    return list(best.values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/fg_config.yaml")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    pc = cfg["pipeline"]
    samples = read_jsonl(os.path.join(pc["out_dir"], "expressions.jsonl"))
    if args.limit:
        samples = samples[: args.limit]

    from ultralytics import YOLO
    model = YOLO(pc["yolo_model"])   # 首次自动下载 yolov8s-worldv2.pt

    results, n_regions = [], 0
    for s in tqdm(samples, desc="grounding"):
        path = resolve_image(s["image"], pc["image_root"])
        try:
            regions = ground_one_image(
                model, path, s["expressions"], pc["conf_threshold"], pc["nms_iou"])
        except Exception:
            continue
        if not regions:   # 所有表达式均无框检出 -> 丢弃该图（方案规则）
            continue
        n_regions += len(regions)
        results.append({"image": s["image"], "caption": s["caption"], "regions": regions})

    out_path = os.path.join(pc["out_dir"], "regions.jsonl")
    write_jsonl(out_path, results)
    print(f"[done] {out_path}: {len(results)} 张图，{n_regions} 个区域"
          f"（平均每图 {n_regions / max(1, len(results)):.1f} 个）")


if __name__ == "__main__":
    main()
