# -*- coding: utf-8 -*-
"""
推理后端：vLLM 离线批量（推荐）/ DashScope API / transformers 兜底
====================================================================
统一接口：infer(items) -> list[str]
    items: [{"image": 图片路径, "prompt": 文本}, ...]
    返回与输入等长的原始输出文本列表（顺序对齐）。
"""

import base64
import os


# ---------------------------------------------------------------
# 后端一：vLLM 离线批推理（本地 GPU，吞吐最高）
# ---------------------------------------------------------------
class VLLMBackend:
    def __init__(self, model: str, tp: int = 1, max_model_len: int = 4096,
                 gpu_mem: float = 0.9):
        from vllm import LLM
        self.llm = LLM(model=model, tensor_parallel_size=tp,
                       max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_mem,
                       limit_mm_per_prompt={"image": 1})
        self.model = model

    def infer(self, items: list[dict], max_tokens: int, temperature: float,
              max_pixels: int) -> list[str]:
        from vllm import SamplingParams
        from PIL import Image

        requests = []
        for it in items:
            img = Image.open(it["image"]).convert("RGB")
            # 像素预算控制：过大图先等比缩小（视觉 token 是成本大头）
            if img.width * img.height > max_pixels:
                scale = (max_pixels / (img.width * img.height)) ** 0.5
                img = img.resize((max(28, int(img.width * scale)),
                                  max(28, int(img.height * scale))))
            requests.append({
                "prompt": f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                          f"{it['prompt']}<|im_end|>\n<|im_start|>assistant\n",
                "multi_modal_data": {"image": img},
            })
        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        outputs = self.llm.generate(requests, sp)
        return [o.outputs[0].text.strip() for o in outputs]


# ---------------------------------------------------------------
# 后端二：DashScope API（72B 等大模型，按量付费）
# ---------------------------------------------------------------
class DashScopeBackend:
    def __init__(self, model: str = "qwen-vl-max", workers: int = 8):
        self.model = model
        self.workers = workers

    def infer(self, items: list[dict], max_tokens: int, temperature: float,
              max_pixels: int) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor
        from dashscope import MultiModalConversation

        def call(item):
            for _ in range(3):                       # API 抖动重试
                try:
                    resp = MultiModalConversation.call(
                        model=self.model,
                        messages=[{"role": "user", "content": [
                            {"image": f"file://{item['image']}"},
                            {"text": item["prompt"]}]}],
                        temperature=temperature, max_tokens=max_tokens)
                    return resp["output"]["choices"][0]["message"]["content"][0]["text"].strip()
                except Exception:
                    continue
            return ""                                  # 连续失败返回空串（进失败日志）

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(call, items))


# ---------------------------------------------------------------
# 后端三：transformers（兜底调试用，逐条，慢）
# ---------------------------------------------------------------
class HFBackend:
    def __init__(self, model: str):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model, torch_dtype="auto", low_cpu_mem_usage=True).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model)

    def infer(self, items: list[dict], max_tokens: int, temperature: float,
              max_pixels: int) -> list[str]:
        import torch
        results = []
        for it in items:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": f"file://{it['image']}"},
                {"type": "text", "text": it["prompt"]}]}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[it["image"]],
                                    return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs, max_new_tokens=max_tokens,
                    do_sample=temperature > 0, temperature=max(temperature, 1e-4))
            gen = out[0][inputs["input_ids"].shape[1]:]
            results.append(self.processor.decode(gen, skip_special_tokens=True).strip())
        return results


def build_backend(name: str, model: str, **kwargs):
    if name == "vllm":
        return VLLMBackend(model, tp=kwargs.get("tp", 1))
    if name == "dashscope":
        return DashScopeBackend(model, workers=kwargs.get("workers", 8))
    if name == "hf":
        return HFBackend(model)
    raise ValueError(f"未知后端: {name}（可选 vllm / dashscope / hf）")
