# -*- coding: utf-8 -*-
"""
可视化：把 regions.jsonl / fg_train.jsonl 中的 bbox 画到原图上
==================================================================
生产线各步骤只输出坐标（jsonl），不产出图片；本工具用于人工抽查
grounding 质量——这是数据生产线最重要的质控手段之一。

用法：
    python -m fg_data.visualize --config configs/test_config.yaml
    python -m fg_data.visualize --config configs/test_config.yaml --input fg_train.jsonl

输出：{out_dir}/vis/<image>（与原图同名的标注图）
"""

import argparse
import os

from PIL import Image, ImageDraw, ImageFont

from .common import load_config, read_jsonl, resolve_image

# 每个区域一个颜色（区分多个框）
_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """优先用系统中文字体（macOS PingFang），找不到则退回 PIL 默认位图字体。"""
    for path in ("/System/Library/Fonts/PingFang.ttc",
                 "/System/Library/Fonts/STHeiti Light.ttc",
                 "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_regions(image_path: str, regions: list[dict], out_path: str):
    """在一张图上画出所有区域框 + 表达式标签 + 置信度。"""
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    font = _load_font(max(14, W // 50))

    for i, r in enumerate(regions):
        color = _COLORS[i % len(_COLORS)]
        x1, y1, x2, y2 = (r["bbox"][0] * W, r["bbox"][1] * H,
                          r["bbox"][2] * W, r["bbox"][3] * H)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = f'{r["expr"]} {r.get("conf", 0):.2f}'
        # 标签底色块，保证文字可读
        tb = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=color)
        draw.text((x1, y1), label, fill="white", font=font)

        # 难负例以副标题形式列在框下方（如有）
        for j, h in enumerate(r.get("hard", [])):
            hy = min(y2 + 4 + j * (font.size + 6), H - font.size - 6)
            draw.text((x1, hy), f"× {h}", fill=color, font=font)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/fg_config.yaml")
    p.add_argument("--input", default="regions.jsonl",
                   help="out_dir 下的文件名：regions.jsonl 或 fg_train.jsonl（含难负例）")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    pc = cfg["pipeline"]
    samples = read_jsonl(os.path.join(pc["out_dir"], args.input))
    if args.limit:
        samples = samples[: args.limit]

    vis_dir = os.path.join(pc["out_dir"], "vis")
    n = 0
    for s in samples:
        src = resolve_image(s["image"], pc["image_root"])
        if not os.path.exists(src):
            print(f"[skip] 图片不存在: {src}")
            continue
        draw_regions(src, s["regions"], os.path.join(vis_dir, os.path.basename(s["image"])))
        n += 1
    print(f"[done] {vis_dir}: {n} 张标注图")


if __name__ == "__main__":
    main()
