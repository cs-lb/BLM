# -*- coding: utf-8 -*-
"""
Step 1：批量生成详细图像描述（dense caption）
==================================================
作用：用 Qwen2.5-VL-3B 为每张图生成「物体+属性+动作+场景」的详细描述。
原始 caption 往往太短（"一个男人在玩球"），抽不出细粒度表达式，
所以要用多模态大模型把图片重新描述一遍——这是后续所有步骤的原料。

加速方式：vLLM 离线批量推理（continuous batching + PagedAttention），
把全部图片一次性送入，由 vLLM 内部调度拼批，吞吐远高于逐张 generate。

输入：data/images/ 下的图片
输出：data/out/captions.jsonl   {"image": "文件名", "caption": "详细描述..."}

运行：PyCharm 里打开本文件，直接点运行按钮即可（无任何必须参数）。
依赖：pip install vllm transformers pillow tqdm
      （vLLM 需要 Linux + NVIDIA GPU；Mac/CPU 环境请改用 transformers 逐张生成）
"""

from pathlib import Path

from PIL import Image
from tqdm import tqdm

import config as C
from common import list_images, out_path, write_jsonl

# Prompt 设计是整个 Step1 的关键：四点要求逼模型输出属性密集的描述，
# 并要求"连贯文字、不分点"——分点列表会给 Step2 的句法抽取制造噪音。
PROMPT = (
    "请详细描述这张图片，要求：\n"
    "1. 列出图中所有主要物体，并说明其颜色、材质、状态等属性；\n"
    "2. 描述物体的动作和相互关系；\n"
    "3. 描述场景环境；\n"
    "4. 输出一段连贯文字，不要列表，不要分点。"
)


def build_prompt_text(processor) -> str:
    """用模型的 chat template 把对话消息拼成 vLLM 需要的纯文本 prompt。

    vLLM 的离线 generate 接口接受 {prompt, multi_modal_data} 结构：
    图像走 multi_modal_data 通道，文本侧的 <|image_pad|> 占位符由
    chat template 自动插入，两者由 vLLM 内部对齐。
    """
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},          # 占位符，实际图像在 multi_modal_data 里传
            {"type": "text", "text": PROMPT},
        ],
    }]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


def quality_filter(caption: str, min_len: int = C.CAPTION_MIN_LEN) -> bool:
    """质控：过滤过短或明显模板化失败的描述。

    宁缺毋滥——这一步放水的噪声会在「抽取→定位→难负例→训练」四级放大。
    """
    return bool(caption) and len(caption.strip()) >= min_len


def main():
    images = list_images()
    print(f"[info] 共发现 {len(images)} 张图片，开始批量生成描述")

    # ---- 加载 vLLM 引擎（首次会从 HuggingFace 下载模型权重）----
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    llm = LLM(
        model=C.VLM_MODEL,
        max_model_len=C.VLM_MAX_MODEL_LEN,
        gpu_memory_utilization=C.GPU_MEM_UTIL,
        limit_mm_per_prompt={"image": 1},   # 每条 prompt 只带一张图
    )
    processor = AutoProcessor.from_pretrained(C.VLM_MODEL)
    prompt_text = build_prompt_text(processor)

    # 贪心解码：数据生产要稳定可复现，不要采样随机性
    sampling = SamplingParams(temperature=0.0,
                              max_tokens=C.CAPTION_MAX_NEW_TOKENS)

    # ---- 组装批量请求：全部图片一次送入，vLLM 内部连续拼批 ----
    requests = []
    valid_images: list[Path] = []
    for p in tqdm(images, desc="读取图片"):
        try:
            # convert("RGB") 兜底 RGBA/灰度等格式；读入后即关闭文件句柄
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"[skip] 图片损坏无法打开: {p.name} ({type(e).__name__})")
            continue
        requests.append({
            "prompt": prompt_text,
            "multi_modal_data": {"image": img},
        })
        valid_images.append(p)

    print(f"[info] 有效图片 {len(requests)} 张，进入 vLLM 批量推理...")
    outputs = llm.generate(requests, sampling)   # 阻塞直至全部完成

    # ---- 收集结果 + 质控 ----
    results, skipped = [], 0
    for p, out in zip(valid_images, outputs):
        caption = out.outputs[0].text.strip()
        if quality_filter(caption):
            results.append({"image": p.name, "caption": caption})
        else:
            skipped += 1

    dst = out_path("captions.jsonl")
    write_jsonl(dst, results)
    print(f"[done] {dst}: {len(results)} 条描述（质控丢弃 {skipped} 条）")
    print(f"[next] 运行 step2_expressions.py 抽取引用表达式")


if __name__ == "__main__":
    main()
