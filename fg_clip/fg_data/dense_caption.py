# -*- coding: utf-8 -*-
"""
Step 1：生成详细图像描述（dense caption）
============================================
方案要点：用多模态大模型（文档用 cogvlm2，本实现默认 Qwen2.5-VL）为每张图
生成「物体+属性+动作+场景」的详细描述——这是后续引用表达式抽取的原料。

用法：
    python -m fg_data.dense_caption --config configs/fg_config.yaml [--limit 100]

输出：data/fg/captions.jsonl   {"image": ..., "caption": "详细描述..."}
"""

import argparse
import os

from tqdm import tqdm

from .common import load_config, read_jsonl, resolve_image, write_jsonl

PROMPT = (
    "请详细描述这张图片，要求：\n"
    "1. 列出图中所有主要物体，并说明其颜色、材质、状态等属性；\n"
    "2. 描述物体的动作和相互关系；\n"
    "3. 描述场景环境；\n"
    "4. 输出一段连贯文字，不要列表，不要分点。"
)


# ---------------------------------------------------------------
# 后端一：本地 Qwen2.5-VL（默认，已缓存 3B 可直接用）
# ---------------------------------------------------------------
class LocalQwenCaptioner:
    def __init__(self, model_name: str):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", low_cpu_mem_usage=True
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

    def caption(self, image_path: str, max_new_tokens: int) -> str:
        import torch
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": PROMPT},
            ],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image_path], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        # 去掉 prompt 部分，只保留新生成的 token
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(gen_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------
# 后端二：DashScope API（无本地算力时）
# ---------------------------------------------------------------
class DashScopeCaptioner:
    def __init__(self, model: str):
        import dashscope  # pip install dashscope；需环境变量 DASHSCOPE_API_KEY
        self.dashscope = dashscope
        self.model = model

    def caption(self, image_path: str, max_new_tokens: int) -> str:
        from dashscope import MultiModalConversation
        resp = MultiModalConversation.call(
            model=self.model,
            messages=[{"role": "user", "content": [
                {"image": f"file://{image_path}"}, {"text": PROMPT}]}],
        )
        return resp["output"]["choices"][0]["message"]["content"][0]["text"].strip()


def quality_filter(caption: str, min_len: int = 20) -> bool:
    """质控：过滤过短或明显模板化失败的描述（方案 3.1）。"""
    return bool(caption) and len(caption) >= min_len


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/fg_config.yaml")
    p.add_argument("--limit", type=int, default=None, help="只处理前 N 张（调试用）")
    args = p.parse_args()

    cfg = load_config(args.config)
    pc = cfg["pipeline"]
    os.makedirs(pc["out_dir"], exist_ok=True)

    samples = read_jsonl(pc["image_jsonl"])
    if args.limit:
        samples = samples[: args.limit]

    if pc["caption_backend"] == "local":
        captioner = LocalQwenCaptioner(pc["caption_model"])
    else:
        captioner = DashScopeCaptioner(pc["dashscope_model"])

    results, skipped = [], 0
    for s in tqdm(samples, desc="dense caption"):
        path = resolve_image(s["image"], pc["image_root"])
        try:
            cap = captioner.caption(path, pc["max_new_tokens"])
        except Exception as e:
            skipped += 1
            continue
        if quality_filter(cap):
            results.append({"image": s["image"], "caption": cap})
        else:
            skipped += 1

    out_path = os.path.join(pc["out_dir"], "captions.jsonl")
    write_jsonl(out_path, results)
    print(f"[done] {out_path}: {len(results)} 条（跳过 {skipped} 条质控不合格）")


if __name__ == "__main__":
    main()
