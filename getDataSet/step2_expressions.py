# -*- coding: utf-8 -*-
"""
Step 2：从详细描述中抽取引用表达式（Referring Expressions）
==============================================================
引用表达式 = 描述中指代图中特定物体/区域的短语，
必须带细粒度修饰（颜色/材质/动作/数量），光秃秃的"猫""狗"不算——
没有修饰语就无法和图中具体区域建立唯一对应，grounding 也没有意义。

例："一个穿着蓝色T恤的男人在一片绿色的草地上玩一个红色的球"
  → ["穿着蓝色T恤的男人", "一片绿色的草地", "一个红色的球"]

三级抽取路径（自动选择，前者优先）：
  1. LLM 抽取（默认）：复用 Qwen2.5-VL（text-only），中文理解最强，
     能抽出「穿棕色夹克的男子」「红色的狗绳」这类动词/属性修饰短语。
     有 CUDA+vLLM 走批量（生产），否则走 transformers 逐条（Mac/试跑）；
  2. spaCy 句法抽取：noun_chunks + 修饰语过滤（中文模型不支持 noun_chunks，
     实测会功能性探测失败，仅英文场景有用）；
  3. 正则兜底：「数量词/属性词 + 名词」片段——有结构性缺陷
     （要求数量词开头、量词表有限），只在无模型环境保底。

LLM 路径的关键校验：抽出的表达式必须是 caption 原文的【连续子串】——
LLM 会幻觉出原文不存在的短语，子串校验一刀切掉，成本为零。

输入：data/out/captions.jsonl（Step1 产物）
输出：data/out/expressions.jsonl   {"image", "caption", "expressions": [...]}

运行：PyCharm 直接点运行；支持断点续跑（已抽取的图片自动跳过）。
      python step2_expressions.py            # 全量（断点续跑）
      python step2_expressions.py --limit 5  # 试跑
依赖：transformers torch tqdm（生产环境有 vllm 时自动走批量加速）
"""

import argparse
import json
import re

import torch
from tqdm import tqdm

import config as C
from common import out_path, read_jsonl

# 代词/无信息量开头直接丢弃——"它"指代不明，grounding 必然产生噪声框
_PRONOUN_HEAD = ("它", "他", "她", "这个", "那个", "这些", "那些", "其", "该")

# ---------------- LLM 抽取 prompt ----------------
# 三点设计：明确"引用表达式"定义（修饰语必备）、强制原文片段（配合子串校验）、
# 纯行输出（便于解析）。one-shot 示例对小模型至关重要——3B 模型对纯指令的
# 遵循很不稳定，给一个完整的输入→输出样例后抽取质量显著提升。
EXTRACT_PROMPT = (
    "从图像描述中抽取所有「引用表达式」。\n"
    "引用表达式 = 描述中指代图中特定物体或区域的名词短语，必须带修饰语"
    "（颜色/材质/动作/数量/状态等），且必须是原文连续片段。\n"
    "注意：以每个主要物体（人/动物/交通工具/物品）为中心，"
    "把它的修饰语合并进同一个短语（如“穿着棕色夹克的男子”），"
    "不要把修饰语拆成独立短语；场景词（街道/人行道/背景）不是物体，不要抽。\n"
    "示例：\n"
    "描述：一个穿着蓝色T恤的男人在绿色的草地上玩一个红色的球。\n"
    "输出：\n"
    "穿着蓝色T恤的男人\n"
    "绿色的草地\n"
    "一个红色的球\n"
    "现在请抽取（只输出表达式，每行一个，不要编号、不要解释）：\n"
    "描述：{caption}\n"
    "输出："
)


# ---------------- 校验 ----------------
def valid_expression(expr: str, caption: str) -> bool:
    """单条表达式校验：
    1. 必须是 caption 原文连续子串（防 LLM 幻觉，零成本）；
    2. 长度 >= 2 且不以代词开头（指代不明的没法 grounding）。
    """
    return (len(expr) >= 2
            and expr in caption
            and not expr.startswith(_PRONOUN_HEAD))


def parse_llm_lines(raw: str) -> list[str]:
    """解析 LLM 输出：按行切分，剥掉可能的序号/项目符号/引号。"""
    out = []
    for line in raw.splitlines():
        line = line.strip().strip("0123456789.、-·* \t\"'“”‘’")
        if line:
            out.append(line)
    return out


# ---------------- LLM 后端（transformers 逐条，Mac/CPU 用） ----------------
class TransformersExtractor:
    """transformers 版抽取器（Apple Silicon 走 MPS，其余 CPU）。"""

    def __init__(self):
        from transformers import (AutoProcessor,
                                  Qwen2_5_VLForConditionalGeneration)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if self.device == "mps" else torch.float32
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            C.EXTRACT_MODEL, torch_dtype=dtype, low_cpu_mem_usage=True,
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(C.EXTRACT_MODEL)
        print(f"[info] LLM 抽取后端: transformers（{self.device}）")

    def extract(self, caption: str) -> list[str]:
        messages = [{"role": "user",
                     "content": EXTRACT_PROMPT.format(caption=caption)}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            # 抽取任务要稳定覆盖全部物体，贪心解码（不要采样随机性）
            out = self.model.generate(**inputs, max_new_tokens=256,
                                      do_sample=False)
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        raw = self.processor.decode(gen_ids, skip_special_tokens=True).strip()
        return parse_llm_lines(raw)


# ---------------- LLM 后端（vLLM 批量，生产用） ----------------
class VLLMExtractor:
    """vLLM 批量抽取器：全部 caption 一次送入，continuous batching 加速。"""

    def __init__(self):
        from vllm import LLM
        from transformers import AutoTokenizer
        self.llm = LLM(model=C.EXTRACT_MODEL,
                       max_model_len=C.VLM_MAX_MODEL_LEN,
                       gpu_memory_utilization=C.GPU_MEM_UTIL)
        self.tokenizer = AutoTokenizer.from_pretrained(C.EXTRACT_MODEL)
        print("[info] LLM 抽取后端: vLLM 批量")

    def extract_batch(self, captions: list[str]) -> list[list[str]]:
        from vllm import SamplingParams
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user",
                  "content": EXTRACT_PROMPT.format(caption=c)}],
                tokenize=False, add_generation_prompt=True)
            for c in captions
        ]
        sampling = SamplingParams(temperature=0.0, max_tokens=256)
        outputs = self.llm.generate(prompts, sampling)
        return [parse_llm_lines(o.outputs[0].text) for o in outputs]


def _vllm_available() -> bool:
    """有 CUDA 且装了 vLLM 才走批量后端（Mac 上 vLLM 装不了/跑不动）。"""
    try:
        import vllm  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


# ---------------- spaCy / 正则兜底（无模型环境的保底） ----------------
def extract_with_spacy(caption: str, nlp) -> list[str]:
    doc = nlp(caption)
    expressions = []
    for chunk in doc.noun_chunks:
        if len(chunk) < 2:
            continue
        # 只保留带修饰语的短语：含形容词/动词/数词
        if not any(t.pos_ in ("ADJ", "VERB", "NUM") for t in chunk):
            continue
        text = chunk.text.strip()
        if valid_expression(text, caption):
            expressions.append(text)
    return expressions


_ATTR_PATTERN = re.compile(
    r"[一二两三四五六七八九十百千万\d]+[个只张片条双件匹头颗株]?"
    r"[^，。；,.;]{1,12}?"
    r"(?:色|红|黄|蓝|绿|白|黑|灰|紫|粉|棕|橙|金|银|大|小|新|旧|长|短|高|矮)"
    r"[^，。；,.;]{0,10}"
)


def extract_fallback(caption: str) -> list[str]:
    """正则兜底：有结构性缺陷（要求数量词开头、量词表有限），仅保底。"""
    return [m.group(0).strip() for m in _ATTR_PATTERN.finditer(caption)
            if valid_expression(m.group(0).strip(), caption)]


def load_spacy_with_probe():
    """加载 spaCy 并做功能性探测（zh 模型 noun_chunks 未实现，E894）。"""
    try:
        import spacy
        nlp = spacy.load(C.SPACY_MODEL)
        list(nlp("一个红色的球在桌子上").noun_chunks)   # 探测：中文会抛异常
        print(f"[info] 使用 spaCy 句法抽取（{C.SPACY_MODEL}）")
        return nlp
    except Exception:
        return None


def load_done_images(dst) -> set[str]:
    """断点续跑：已抽取的图片名集合。"""
    if not dst.exists():
        return set()
    return {item["image"] for item in read_jsonl(dst)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dst = out_path("expressions.jsonl")
    done = load_done_images(dst)
    samples = [s for s in read_jsonl(out_path("captions.jsonl"))
               if s["image"] not in done]
    if args.limit:
        samples = samples[: args.limit]
    print(f"[info] 待抽取 {len(samples)} 条 caption（已完成 {len(done)} 条跳过）")
    if not samples:
        print("[done] 没有需要处理的图片")
        return

    # 后端选择：vLLM（生产批量）> transformers（Mac 逐条）> spaCy > 正则
    extractor = None
    nlp = None
    if _vllm_available():
        extractor = VLLMExtractor()
    else:
        try:
            extractor = TransformersExtractor()
        except Exception as e:
            print(f"[warn] LLM 后端不可用（{type(e).__name__}），"
                  f"降级 spaCy/正则")
            nlp = load_spacy_with_probe()

    n_expr_total, dropped = 0, 0
    with open(dst, "a", encoding="utf-8") as f:
        if isinstance(extractor, VLLMExtractor):
            # 批量路径：全部一次推理，再逐条校验落盘
            all_exprs = extractor.extract_batch([s["caption"] for s in samples])
            iterator = zip(samples, all_exprs)
        else:
            iterator = ((s, None) for s in samples)

        for s, batch_exprs in tqdm(iterator, total=len(samples),
                                   desc="抽取表达式"):
            if batch_exprs is not None:
                raw_exprs = batch_exprs
            elif extractor is not None:
                try:
                    raw_exprs = extractor.extract(s["caption"])
                except Exception as e:
                    print(f"\n[skip] LLM 抽取失败: {s['image']} "
                          f"({type(e).__name__})")
                    continue
            elif nlp is not None:
                raw_exprs = extract_with_spacy(s["caption"], nlp)
            else:
                raw_exprs = extract_fallback(s["caption"])

            # 校验（子串防幻觉）+ 去重保序 + 截断上限
            exprs = [e for e in dict.fromkeys(raw_exprs)
                     if valid_expression(e, s["caption"])]
            exprs = exprs[: C.MAX_EXPR_PER_IMAGE]
            if len(exprs) < C.MIN_EXPR_PER_IMAGE:
                dropped += 1      # 表达式太少说明 caption 质量差，丢图
                continue
            f.write(json.dumps({**s, "expressions": exprs},
                               ensure_ascii=False) + "\n")
            f.flush()
            n_expr_total += len(exprs)

    print(f"[done] {dst}: 本轮 {len(samples) - dropped} 张图，"
          f"共 {n_expr_total} 个引用表达式（丢弃 {dropped} 张）")
    print(f"[next] 运行 step3_grounding.py 定位边界框")


if __name__ == "__main__":
    main()
