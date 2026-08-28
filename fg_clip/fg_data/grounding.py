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
                     conf_thres: float, nms_iou: float,
                     detect_expressions: list[str] | None = None) -> list[dict]:
    """对一张图的所有表达式做开放词汇检测，返回 [{expr, bbox, conf}]。

    YOLO-World 的 set_classes 设置开放词表后，predict 的输出里
    每个检测框的 cls 对应词表下标；模型内置 NMS（iou 参数）。
    每个表达式只保留置信度最高的一个框（方案：每物体一个最准的框）。

    detect_expressions：实际送检词表（如英文翻译），与 expressions 等长一一对应；
    输出 expr 字段始终保留原始中文表达式。
    """
    detect_list = detect_expressions if detect_expressions else expressions
    model.set_classes(detect_list)
    results = model.predict(image_path, conf=conf_thres, iou=nms_iou, verbose=False)[0]

    W, H = results.orig_shape[1], results.orig_shape[0]
    best: dict[int, dict] = {}   # cls_idx -> 最高分框
    for box in results.boxes:
        cls_idx = int(box.cls.item())
        conf = float(box.conf.item())
        if cls_idx not in best or conf > best[cls_idx]["conf"]:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            best[cls_idx] = {
                "expr": expressions[cls_idx],   # 保留原始（中文）表达式
                "bbox": [x1 / W, y1 / H, x2 / W, y2 / H],   # 归一化坐标
                "conf": round(conf, 4),
            }
    return list(best.values())


# ---------------------------------------------------------------
# 可选：表达式翻译（YOLO-World 文本编码器是英文 CLIP，中文检出率极低）
# ---------------------------------------------------------------
class ZhEnTranslator:
    """Helsinki-NLP opus-mt-zh-en 轻量翻译器（首次自动下载，约 300MB）。

    注：transformers v5 已移除 "translation" pipeline，这里直接用
    AutoModelForSeq2SeqLM + generate，兼容性更好。
    """

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).eval()

    def __call__(self, text: str) -> str:
        inputs = self.tok(text, return_tensors="pt", truncation=True, max_length=128)
        with self.torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=64)
        return self.tok.decode(ids[0], skip_special_tokens=True).strip()


def translate_expressions(translator, expressions: list[str]) -> list[str]:
    """逐条翻译；翻译失败时回退为原表达式（检出率差但不中断流水线）。"""
    out = []
    for expr in expressions:
        try:
            out.append(translator(expr))
        except Exception:
            out.append(expr)
    return out


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

    # 可选翻译：YOLO-World 文本编码器是英文 CLIP，中文表达式直接检出率极低
    translator = None
    if pc.get("translate_to_en"):
        translator = ZhEnTranslator(pc.get("translate_model", "Helsinki-NLP/opus-mt-zh-en"))
        print("[info] 表达式将翻译为英文后送检（expr 字段仍保留中文）")

    results, n_regions = [], 0
    for s in tqdm(samples, desc="grounding"):
        path = resolve_image(s["image"], pc["image_root"])
        try:
            detect_list = (translate_expressions(translator, s["expressions"])
                           if translator else None)
            regions = ground_one_image(
                model, path, s["expressions"], pc["conf_threshold"], pc["nms_iou"],
                detect_expressions=detect_list)
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
