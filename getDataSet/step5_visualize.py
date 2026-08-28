# -*- coding: utf-8 -*-
"""
Step 5（可选）：可视化抽查
==============================
把 fg_train.jsonl 里的 bbox 画回原图，框上标注「表达式 + 置信度」，
框下方列出难负例。这是数据生产线性价比最高的质控手段：
跑完一批先抽查十几张，重点看——
  1. 框是否套住了表达式描述的物体（而不是全图或错位）；
  2. 是否有框面积占比接近全图的噪声框（考虑加面积过滤规则）；
  3. 难负例是否确实与正例只差一个属性。

输入：data/out/fg_train.jsonl（Step4 产物）+ data/images/ 原图
输出：data/out/vis/ 下的同名标注图

运行：PyCharm 直接点运行。依赖：pillow
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import config as C
from common import list_images, out_path, read_jsonl

# 每个区域轮换一个颜色，便于区分同图多个框
_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]


def _load_font(size: int):
    """优先系统中文字体（macOS PingFang / Linux Noto），找不到退回 PIL 默认字体。"""
    for path in ("/System/Library/Fonts/PingFang.ttc",
                 "/System/Library/Fonts/STHeiti Light.ttc",
                 "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_regions(image_path: Path, regions: list[dict], out: Path):
    """在一张图上画出所有区域框 + 表达式标签 + 置信度 + 难负例副标题。"""
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    font = _load_font(max(14, W // 50))

    for i, r in enumerate(regions):
        color = _COLORS[i % len(_COLORS)]
        # 归一化坐标换算回像素坐标
        x1, y1, x2, y2 = (r["bbox"][0] * W, r["bbox"][1] * H,
                          r["bbox"][2] * W, r["bbox"][3] * H)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 标签带底色块，保证在任何图片上可读
        label = f'{r["expr"]} {r.get("conf", 0):.2f}'
        tb = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=color)
        draw.text((x1, y1), label, fill="white", font=font)

        # 难负例以 × 前缀列在框下方（与正例共用同一个框）
        for j, h in enumerate(r.get("hard", [])):
            hy = min(y2 + 4 + j * (font.size + 6), H - font.size - 6)
            draw.text((x1, hy), f"× {h}", fill=color, font=font)

    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def main():
    samples = read_jsonl(out_path("fg_train.jsonl"))[: C.VIS_MAX_IMAGES]
    image_index = {p.name: p for p in list_images()}
    vis_dir = out_path("vis")

    n = 0
    for s in samples:
        src = image_index.get(s["image"])
        if src is None:
            print(f"[skip] 图片缺失: {s['image']}")
            continue
        draw_regions(src, s["regions"], vis_dir / s["image"])
        n += 1
    print(f"[done] {vis_dir}: {n} 张标注图，请人工抽查框的质量")


if __name__ == "__main__":
    main()
