# -*- coding: utf-8 -*-
"""
Step 1（Mac 版）：生成详细图像描述（dense caption）
======================================================
与 step1_dense_caption.py 功能相同，但用 transformers 直接推理，
适配无 NVIDIA GPU 的 macOS（vLLM 仅支持 Linux + CUDA）。

设备选择（自动）：
  - Apple Silicon 优先走 MPS（Metal GPU 加速，比 CPU 快数倍）；
  - 不可用则退回 CPU（Qwen2.5-VL-3B 约 2~3 分钟/张，做好按天跑的准备）。

针对 Mac 低速场景的两个生产级设计：
  1. 断点续跑：已写入 captions.jsonl 的图片自动跳过，重复运行安全；
  2. 逐条落盘：每生成一条立即 append 写文件，中途崩溃/Ctrl-C 不丢进度。

输入：data/images/ 下的图片
输出：data/out/captions.jsonl   {"image": "文件名", "caption": "详细描述..."}

运行：PyCharm 直接点运行；也可命令行传参限制数量：
      python dense_caption_mac.py            # 全量（断点续跑）
      python dense_caption_mac.py --limit 5  # 只跑前 5 张（试跑验证）
依赖：pip install transformers torch pillow tqdm
"""

import argparse
import json

import torch
from PIL import Image
from tqdm import tqdm

import config as C
from common import list_images, out_path

# 与 GPU 版完全相同的 prompt——四点要求逼模型输出属性密集的描述，
# 输出连贯文字（不分点），便于 Step2 句法/正则抽取
PROMPT = (
    "请详细描述这张图片，要求：\n"
    "1. 列出图中所有主要物体，并说明其颜色、材质、状态等属性；\n"
    "2. 描述物体的动作和相互关系；\n"
    "3. 描述场景环境；\n"
    "4. 输出一段连贯文字，不要列表，不要分点。"
)


def pick_device() -> str:
    """Apple Silicon 用 MPS 加速；其余退回 CPU。"""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class MacCaptioner:
    """transformers 版 Qwen2.5-VL 描述生成器（逐张推理）。"""

    def __init__(self):
        from transformers import (AutoProcessor,
                                  Qwen2_5_VLForConditionalGeneration)
        self.device = pick_device()
        # MPS 上用 float16 显著快于 float32；CPU 用 float32 更稳
        dtype = torch.float16 if self.device == "mps" else torch.float32
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            C.VLM_MODEL, torch_dtype=dtype, low_cpu_mem_usage=True,
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(C.VLM_MODEL)
        print(f"[info] 模型已加载到 {self.device}（dtype={dtype}）")

    def caption(self, image_path) -> str:
        """对单张图生成描述；贪心解码保证可复现（数据生产不要采样随机性）。"""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        img = Image.open(image_path).convert("RGB")   # 兜底 RGBA/灰度
        inputs = self.processor(text=[text], images=[img],
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=C.CAPTION_MAX_NEW_TOKENS,
                do_sample=False)
        # 切掉 prompt 部分，只保留模型新生成的 token
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(gen_ids, skip_special_tokens=True).strip()


def load_done_images(dst) -> set[str]:
    """读取已有产物，返回已完成图片名集合（断点续跑的关键）。"""
    if not dst.exists():
        return set()
    done = set()
    with open(dst, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["image"])
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="只处理前 N 张（试跑验证用）")
    args = parser.parse_args()

    dst = out_path("captions.jsonl")
    done = load_done_images(dst)

    images = list_images()
    todo = [p for p in images if p.name not in done]   # 跳过已完成
    if args.limit:
        todo = todo[: args.limit]

    print(f"[info] 共 {len(images)} 张图片，已完成 {len(done)} 张，"
          f"本次待处理 {len(todo)} 张")
    if not todo:
        print("[done] 没有需要处理的图片")
        return

    captioner = MacCaptioner()

    # 逐条生成 + 立即落盘：任何时刻中断，已完成的进度都在文件里
    n_ok, n_skip = 0, 0
    with open(dst, "a", encoding="utf-8") as f:
        for p in tqdm(todo, desc="dense caption (mac)"):
            try:
                caption = captioner.caption(p)
            except Exception as e:
                print(f"\n[skip] 处理失败: {p.name} ({type(e).__name__})")
                n_skip += 1
                continue
            # 质控：过滤过短或模板化失败的描述，宁缺毋滥
            if not caption or len(caption) < C.CAPTION_MIN_LEN:
                n_skip += 1
                continue
            f.write(json.dumps({"image": p.name, "caption": caption},
                               ensure_ascii=False) + "\n")
            f.flush()          # 立即刷盘，不等缓冲区
            n_ok += 1
            # MPS 长时运行会累积显存碎片，定期清理
            if captioner.device == "mps" and n_ok % 50 == 0:
                torch.mps.empty_cache()

    print(f"[done] {dst}: 本次新增 {n_ok} 条（跳过 {n_skip} 条），"
          f"累计 {len(load_done_images(dst))} 条")
    print(f"[next] 运行 step2_expressions.py 抽取引用表达式")


if __name__ == "__main__":
    main()
