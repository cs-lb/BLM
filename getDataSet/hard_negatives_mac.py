# -*- coding: utf-8 -*-
"""
Step 4（Mac 版）：LLM 构造难负例（hard negatives）
====================================================
与 step4_hard_negatives.py 功能相同，但用 transformers 逐条推理，
适配无 NVIDIA GPU 的 macOS（vLLM 仅支持 Linux + CUDA）。
prompt、三重校验、规则兜底直接复用 step4_hard_negatives.py 中
已验证的函数（那些函数不依赖 vLLM），保证两个版本口径完全一致。

针对 Mac 低速场景的设计（与 dense_caption_mac.py 相同思路）：
  1. 断点续跑：已写入 hard_negatives.jsonl 的表达式自动跳过；
  2. 逐条落盘：每生成一条立即 append + flush，中断不丢进度；
  3. 每次运行结束时都用「regions.jsonl + 全部难负例」重新组装
     fg_train.jsonl——即使中途中断，已产出的部分也能立即用于训练。

输入：data/out/regions.jsonl（Step3 产物）
输出：data/out/hard_negatives.jsonl  进度文件 {"image", "expr", "hard": [...]}
      data/out/fg_train.jsonl        最终训练数据（每次运行结束自动重组装）

运行：PyCharm 直接点运行；也可命令行限制数量：
      python hard_negatives_mac.py --limit 5
依赖：transformers torch tqdm（不需要 vllm）
"""

import argparse
import json
import random

import torch
from tqdm import tqdm

import config as C
from common import out_path, read_jsonl, write_jsonl
# 复用 step4 中已验证的 prompt / 校验 / 规则兜底（这些函数不依赖 vLLM，
# vllm 的 import 在 step4 的 main() 内部，import 本模块是安全的）
from step4_hard_negatives import (LLM_PROMPT, _valid_negative,
                                  make_rule_negatives, parse_llm_output)


def pick_device() -> str:
    """Apple Silicon 用 MPS 加速；其余退回 CPU。"""
    return "mps" if torch.backends.mps.is_available() else "cpu"


class MacHardNegativeMaker:
    """transformers 版难负例生成器（text-only，复用 Qwen2.5-VL 权重）。"""

    def __init__(self):
        from transformers import (AutoProcessor,
                                  Qwen2_5_VLForConditionalGeneration)
        self.device = pick_device()
        dtype = torch.float16 if self.device == "mps" else torch.float32
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            C.VLM_MODEL, torch_dtype=dtype, low_cpu_mem_usage=True,
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(C.VLM_MODEL)
        print(f"[info] 模型已加载到 {self.device}（dtype={dtype}）")

    def generate(self, expr: str, k: int) -> list[str]:
        """对单个表达式生成难负例候选（未校验）。
        多要 1 条容错；do_sample + 低温提供改写多样性，
        贪心会总产出同一种改法。"""
        messages = [{"role": "user",
                     "content": LLM_PROMPT.format(k=k + 1, expr=expr)}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=128,
                do_sample=True, temperature=C.HARD_NEG_TEMPERATURE,
                top_p=C.HARD_NEG_TOP_P)
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        raw = self.processor.decode(gen_ids, skip_special_tokens=True).strip()
        return parse_llm_output(raw)


def load_done_keys(progress_path) -> dict[tuple[str, str], list[str]]:
    """读取进度文件，返回 {(image, expr): hard_list}（断点续跑的核心）。"""
    done = {}
    if progress_path.exists():
        for item in read_jsonl(progress_path):
            done[(item["image"], item["expr"])] = item["hard"]
    return done


def assemble_fg_train(done: dict[tuple[str, str], list[str]]):
    """用 regions.jsonl + 全部难负例重组装 fg_train.jsonl。

    每次运行结束都执行：改不出难负例的区域丢弃（没有对比就没有可学信号），
    整张图没有有效区域的丢图。
    """
    samples = read_jsonl(out_path("regions.jsonl"))
    results, n_pos, n_hard = [], 0, 0
    for s in samples:
        regions = []
        for r in s["regions"]:
            hard = done.get((s["image"], r["expr"]))
            if not hard:
                continue
            n_pos += 1
            n_hard += len(hard)
            regions.append({**r, "hard": hard})
        if regions:
            results.append({"image": s["image"], "caption": s["caption"],
                            "regions": regions})
    dst = out_path("fg_train.jsonl")
    write_jsonl(dst, results)
    return dst, len(results), n_pos, n_hard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="只处理前 N 个表达式（试跑验证用）")
    args = parser.parse_args()

    progress_path = out_path("hard_negatives.jsonl")
    done = load_done_keys(progress_path)

    # 拍平所有正例表达式，跳过已完成（断点续跑）
    samples = read_jsonl(out_path("regions.jsonl"))
    todo = []
    for s in samples:
        for r in s["regions"]:
            if (s["image"], r["expr"]) not in done:
                todo.append((s["image"], r["expr"]))
    if args.limit:
        todo = todo[: args.limit]

    print(f"[info] 共 {sum(len(s['regions']) for s in samples)} 个表达式，"
          f"已完成 {len(done)} 个，本次待处理 {len(todo)} 个")

    if todo:
        maker = MacHardNegativeMaker()
        rng = random.Random(42)
        n_llm, n_fill = 0, 0
        # 逐条生成 + 立即落盘：任何时刻中断，进度都在文件里
        with open(progress_path, "a", encoding="utf-8") as f:
            for image, expr in tqdm(todo, desc="hard negatives (mac)"):
                used = {expr}
                negatives = []
                try:
                    for neg in maker.generate(expr, C.HARD_NEG_PER_EXPR):
                        if len(negatives) >= C.HARD_NEG_PER_EXPR:
                            break
                        if _valid_negative(expr, neg, used):   # 三道校验闸门
                            used.add(neg)
                            negatives.append(neg)
                except Exception as e:
                    print(f"\n[warn] LLM 生成失败（{type(e).__name__}），"
                          f"本条走规则兜底: {expr}")
                n_llm += len(negatives)
                if len(negatives) < C.HARD_NEG_PER_EXPR:
                    # LLM 产出不足 -> 规则词表补齐，保证流水线不断
                    fill = make_rule_negatives(
                        expr, C.HARD_NEG_PER_EXPR - len(negatives), rng, used)
                    n_fill += len(fill)
                    negatives.extend(fill)
                if negatives:
                    f.write(json.dumps({"image": image, "expr": expr,
                                        "hard": negatives},
                                       ensure_ascii=False) + "\n")
                    f.flush()          # 立即刷盘
                    done[(image, expr)] = negatives
                if maker.device == "mps" and (n_llm + n_fill) % 50 == 0:
                    torch.mps.empty_cache()   # MPS 长时运行定期清理显存碎片
        print(f"[info] 本轮：LLM 通过校验 {n_llm} 条，规则补齐 {n_fill} 条")

    # 每次运行结束都重组装最终训练数据（部分完成也能用）
    done = load_done_keys(progress_path)
    dst, n_img, n_pos, n_hard = assemble_fg_train(done)
    print(f"[done] {dst}: {n_img} 张图，{n_pos} 个正例区域，{n_hard} 条难负例")
    print(f"[next] 可运行 step5_visualize.py 抽查标注质量")


if __name__ == "__main__":
    main()
