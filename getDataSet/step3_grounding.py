# -*- coding: utf-8 -*-
"""
Step 3：边界框定位（Grounding）
==================================
用 YOLO-World（开放词汇检测器）为每个引用表达式在图中定位边界框。
YOLO-World 的 set_classes 支持任意文本词表：把本图表达式设成词表后，
检测头对每个词表项独立打分，每个检测框的 cls 即词表下标。

三条质控规则（共同原则：宁可丢数据，不要噪声监督）：
  1. NMS 去重（模型内置，iou=0.5）；
  2. 只保留置信度 > CONF_THRESHOLD 的框；
  3. 表达式无框检出时丢弃该表达式——不退化用全图框
     （全图框等于把区域监督退化成全局监督，还不如没有）。

两个实测结论（来自 fg_clip 链路测试，勿轻易改回旧口径）：
  - YOLO-World 的文本编码器是英文 CLIP，中文表达式直检几乎全零检出，
    因此默认开启 TRANSLATE_TO_EN，先翻译成英文送检，输出的 expr 仍保留中文；
  - 带属性的长短语置信度系统性偏低（正确框常只有 0.1~0.25），
    方案的 0.4 口径会把全部正确框筛掉，默认阈值已校准为 0.1。

输入：data/out/expressions.jsonl（Step2 产物）+ data/images/ 原图
输出：data/out/regions.jsonl
      {"image", "caption", "regions": [{"expr", "bbox", "conf"}]}
      bbox 为归一化 [x1,y1,x2,y2]，与图片分辨率解耦，训练时按实际尺寸换算

运行：PyCharm 直接点运行（首次自动下载 YOLO-World 与翻译模型权重）。
依赖：pip install ultralytics transformers torch pillow tqdm
"""

import config as C
from common import list_images, out_path, read_jsonl, write_jsonl


class ZhEnTranslator:
    """中译英轻量翻译器（Helsinki opus-mt-zh-en，约 300MB）。

    注：transformers v5 已移除 "translation" pipeline 接口，
    这里直接用 AutoModelForSeq2SeqLM + generate，兼容性更好。
    翻译失败时回退为原中文表达式（检出率低但不中断流水线）。
    """

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).eval()

    def __call__(self, text: str) -> str:
        inputs = self.tok(text, return_tensors="pt", truncation=True,
                          max_length=128)
        with self.torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=64)
        return self.tok.decode(ids[0], skip_special_tokens=True).strip()


def ground_one_image(model, image_path, expressions: list[str],
                     detect_expressions: list[str] | None = None) -> list[dict]:
    """对一张图的所有表达式做开放词汇检测，返回 [{expr, bbox, conf}]。

    detect_expressions：实际送检词表（英文翻译），与 expressions 等长对应；
    输出的 expr 字段始终保留原始中文表达式。
    每个表达式只保留置信度最高的一个框（每个物体取一个最准的框）。
    """
    detect_list = detect_expressions if detect_expressions else expressions
    model.set_classes(detect_list)
    results = model.predict(str(image_path), conf=C.CONF_THRESHOLD,
                            iou=C.NMS_IOU, verbose=False)[0]

    W, H = results.orig_shape[1], results.orig_shape[0]
    best: dict[int, dict] = {}     # 词表下标 -> 最高分框
    for box in results.boxes:
        cls_idx = int(box.cls.item())
        conf = float(box.conf.item())
        if cls_idx not in best or conf > best[cls_idx]["conf"]:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            best[cls_idx] = {
                "expr": expressions[cls_idx],          # 保留中文原文
                "bbox": [x1 / W, y1 / H, x2 / W, y2 / H],  # 归一化坐标
                "conf": round(conf, 4),
            }
    return list(best.values())


def main():
    samples = read_jsonl(out_path("expressions.jsonl"))
    image_index = {p.name: p for p in list_images()}   # 文件名 -> 完整路径

    from ultralytics import YOLO
    model = YOLO(C.YOLO_MODEL)   # 首次自动下载 yolov8s-worldv2.pt

    translator = None
    if C.TRANSLATE_TO_EN:
        translator = ZhEnTranslator(C.TRANSLATE_MODEL)
        print("[info] 表达式将翻译为英文后送检（输出 expr 仍保留中文）")

    results, n_regions, dropped = [], 0, 0
    for s in samples:
        path = image_index.get(s["image"])
        if path is None:
            print(f"[skip] 图片缺失: {s['image']}")
            continue
        try:
            detect_list = ([translator(e) for e in s["expressions"]]
                           if translator else None)
            regions = ground_one_image(model, path, s["expressions"],
                                       detect_list)
        except Exception as e:
            print(f"[skip] 检测失败: {s['image']} ({type(e).__name__})")
            continue
        if not regions:      # 所有表达式均无框检出 -> 丢图（方案规则）
            dropped += 1
            continue
        n_regions += len(regions)
        results.append({"image": s["image"], "caption": s["caption"],
                        "regions": regions})

    dst = out_path("regions.jsonl")
    write_jsonl(dst, results)
    print(f"[done] {dst}: {len(results)} 张图，{n_regions} 个区域"
          f"（平均每图 {n_regions / max(1, len(results)):.1f} 个，丢弃 {dropped} 张）")
    print(f"[next] 运行 step4_hard_negatives.py 构造难负例")


if __name__ == "__main__":
    main()
