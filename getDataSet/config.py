# -*- coding: utf-8 -*-
"""
全局配置（所有 step 脚本共享）
================================
所有路径都基于本文件所在目录解析（Path(__file__)），
因此在 PyCharm 里直接点运行按钮时，与工作目录（cwd）无关，不会找不到文件。

如需调整参数（模型、阈值、每图表达式数等），只改这一个文件。
"""

from pathlib import Path

# ---------------- 路径（自动解析，勿改成相对路径） ----------------
BASE_DIR = Path(__file__).resolve().parent      # getDataSet/ 目录
IMAGE_DIR = BASE_DIR / "data" / "images"        # 原始图片放这里（jpg/png/jpeg）
OUT_DIR = BASE_DIR / "data" / "out"             # 各步骤产物统一输出到这里

# ---------------- 视觉语言模型（Step1 描述生成 / Step4 难负例共用） ----------------
VLM_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"       # 首次运行 vLLM 会自动从 HF 下载
VLM_MAX_MODEL_LEN = 4096                        # 上下文长度，4096 对 3B + 单图足够
GPU_MEM_UTIL = 0.85                             # vLLM 显存占用比例

# ---------------- Step 1：dense caption ----------------
CAPTION_MAX_NEW_TOKENS = 256                    # 描述最大生成长度
CAPTION_MIN_LEN = 20                            # 质控：短于该长度的描述视为失败，丢弃

# ---------------- Step 2：引用表达式抽取 ----------------
# 实测：3B 模型做抽取指令遵循不稳定（漏物体、拆修饰语、抽场景词），
# 7B 显著更稳；32GB 内存的 Mac 可跑（MPS float16 约 15GB）
EXTRACT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
SPACY_MODEL = "zh_core_web_sm"                  # 注意：中文模型不支持 noun_chunks，
                                                # 脚本会自动探测并降级为正则抽取
MIN_EXPR_PER_IMAGE = 1                          # 每图至少抽到的表达式数，不足则丢图
MAX_EXPR_PER_IMAGE = 8                          # 每图保留上限（控制下游 grounding/训练开销）；
                                                # 实测 LLM 输出常 6~8 条，6 会截掉后排物体

# ---------------- Step 3：YOLO-World grounding ----------------
YOLO_MODEL = "yolov8s-worldv2.pt"               # 首次运行 ultralytics 自动下载
CONF_THRESHOLD = 0.1                            # 实测：带属性的长短语置信度系统性偏低
                                                # （正确框也常只有 0.1~0.25），方案的 0.4 口径会全筛掉
NMS_IOU = 0.5                                   # NMS 去重阈值
TRANSLATE_TO_EN = True                          # YOLO-World 文本编码器是英文 CLIP，
                                                # 中文表达式直检检出率极低，先翻译成英文送检
TRANSLATE_MODEL = "Helsinki-NLP/opus-mt-zh-en"  # 轻量翻译模型（约 300MB，首次自动下载）

# ---------------- Step 4：LLM 难负例 ----------------
HARD_NEG_PER_EXPR = 2                           # 每个正例表达式要几条难负例
HARD_NEG_TEMPERATURE = 0.7                      # 改写需要多样性；贪心会总产出同一种改法
HARD_NEG_TOP_P = 0.9
MIN_SIM_RATIO = 0.5                             # 编辑相似度下限：拦截 LLM 整句重写
                                                # （"只改一个属性"的合格难负例通常 > 0.7）

# ---------------- Step 5：可视化抽查 ----------------
VIS_MAX_IMAGES = 20                             # 每次最多画多少张标注图
