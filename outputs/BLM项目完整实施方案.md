# BLM 视觉基座完整实施方案（从 0 到 1 实现规范）

> 用途：交给 AI/工程师直接实现的工程规范。每个模块含**接口契约、数据格式、超参、
> 验收标准、已知陷阱**。实现顺序见第 8 章（验收驱动开发）。
> 技术栈：Python 3.11+ / PyTorch ≥2.2 / transformers ≥5.0 / Qwen2.5-VL-7B / BGE-M3。

---

## 0. 项目目标与三阶段总览

训练风控视觉基座 **BLM ViT**，使"风险图片 ↔ 拒绝理由/caption"在同一 1024 维
余弦嵌入空间内可检索，业务图搜文 hit@5（gallery=1000）≥ 97%。

```
Stage 1  SigLIP 自监督预训练（v1）
         QwenViT-7B + BGE-M3，公共+业务图文对，动态分辨率 packing/binpack，
         多卡负例同步 → BLM ViT v1
Stage 2  图文细粒度对齐（FG-CLIP）
         区域对比学习（区域框↔表达式+难负例），~20w 数据 → 细粒度增强
Stage 3  正负样本对有监督训练（v2）
         三元损失（margin）+ MSE 蒸馏（教师 BGE-VL-v1.5-mmeb）
         → BLM ViT v2 → hit@5 测评
```

## 1. 数据规范（三条数据生产线）

### 1.1 图文对数据（Stage 1 原料）

**来源**：Wukong100m（HF `wanng/wukong100m`，32 分片 parquet，每片 315 万行，
列名 `url` + `caption`，CC BY-NC-SA 4.0 仅限研究）+ 业务封面（飞轮刷 caption）。

**配比**：公共:业务 = 80:20；业务不足时重复采样等价混入（不降公共占比）。

**清洗管线**（每步防一个坑，阈值依据见注释）：

| 步骤 | 规则 | 依据 |
|---|---|---|
| caption 清洗 | 压缩空白，保留 5~200 字 | <5 字实测仅 1.1% 且多为文件名式噪声；200 字对齐 text_max_len=96 的截断护栏 |
| 图片校验 | 最小边 ≥64px、文件 ≥5KB、PIL 可解码 | 64px=patch 网格+min_pixels 约束；5KB 挡占位图/追踪像素；可解码防训练期随机崩溃 |
| 去重 | 内容 sha1 字节级 | 重复图会成为"最难负例"→ loss 虚低、指标虚高 |
| 切分 | **先 shuffle 再切** 99:1 | 防近邻样本同时进 train/eval 泄漏 |

**输出格式**（`train.jsonl` / `eval.jsonl`，每行一条）：
```json
{"image": "<sha1>.jpg", "text": "caption 文本"}
```
图片落盘 `images/<sha1>.jpg`（重编码 JPEG q95）。

**断点续跑契约**：启动时预加载 `images/` 已有文件名集合（文件名即 sha1），
重复内容跳过写盘但计入 records。

**下载并发**：分块提交线程池（每批 workers×20 个 URL），禁止一次性 submit
全部 URL（收够样本提前退出时 executor 会傻等全部完成）。

### 1.2 细粒度数据（Stage 2 原料，~20w 图）

五步生产线（中间产物全部落盘可检查）：

```
① dense caption   MLLM 生成详细描述（物体+属性+动作+场景，一段连贯文字）
② 引用表达式抽取   SpaCy 名词短语块 + 修饰语过滤（必须含 ADJ/VERB/NUM，
                  丢弃代词开头），每图保留 1~6 个
③ 边界框定位      YOLO-World 开放词汇检测，每表达式保留最高分框，
                  conf>0.4，无框检出丢弃该表达式（不退化用全图框）
④ 难负例构造      保留核心对象单属性改写，优先级 颜色>动作>数量>材质
                  （"白色猫"→"金色猫"，框不变），每正例 1~3 条
⑤ 汇总           fg_train.jsonl
```

**输出格式**：
```json
{"image": "<sha1>.jpg", "caption": "详细描述",
 "regions": [{"expr": "穿着蓝色T恤的男人", "bbox": [0.31,0.12,0.55,0.86], "conf": 0.62,
              "hard": ["穿着绿色T恤的男人", "穿着蓝色T恤的女人"]}]}
```
bbox 为归一化 [x1,y1,x2,y2] ∈ [0,1]。

### 1.3 正负样本对数据（Stage 3 原料）

**生产链路**：
```
Qwen2.5-VL 72B + PE 生成候选（每图 5~7 理由，故意混入无关项，JSON 输出）
  → 人类专家精标 2000+ 对 → LoRA 微调 32B 得 Judged MLLM
    （few-shot CoT + 多 Judge 集成 + 低 T；500 条人标验证一致率 ≥95% 才放量）
  → 裁判批量判定（相关/不相关 + confidence + analysis）
  → 难负例四通道挖掘：
    ① LLM 版本分歧（v1 判正 v2 判负）② KNN 嵌入近邻但理由不相关
    ③ 负例相似（与正例理由文本相似但不相关）
    ④ 数据飞轮（线上真实召回错误 = 最高质量难负例，审核员同步标注正例）
```

**judge 判定输出格式**（即 Stage 3 训练原料）：
```json
{"image": "<sha1>.jpg", "reason_id": "R03", "reason_text": "...",
 "label": "相关", "teacher_score": 0.93, "confidence": 0.95, "analysis": "..."}
```

**三元组构建**：按图聚合为 anchor-candidates；正例全留（≤2），负例按
teacher_score 降序取 top-4（分数越高的负例越难）；缺正或缺负的图丢弃。
```json
{"image": "<sha1>.jpg", "candidates": [
  {"text": "R03 ...", "label": 1, "teacher_score": 0.93, "conf": 0.95},
  {"text": "R01 ...", "label": 0, "teacher_score": 0.06, "conf": 0.88}]}
```

**批量推理基建（vlm_batch）**：三后端（vLLM 离线批 / DashScope API / HF 兜底）+
缓存断点（key=图+任务+prompt_version，judge 加 reason_id；输出文件即断点账本，
append 写入；失败进 failures.jsonl 支持定点重试）。

## 2. 模型规范

### 2.1 双塔结构（Stage 1~3 共用基座）

```
图像流（QwenViT-7B 视觉塔，675M）:
  图片 → smart_resize(28倍数, token预算[128,448])
  → patch 重排 [Σpatch, 1176]      # 1176 = 3×2×14×14，按官方 reshape+permute
  → Conv3d 嵌入 [Σpatch, 1280]     # kernel=stride=(2,14,14)
  → 32 × Block（窗口注意力 8×8patch + 第7/15/23/31层全局，mRoPE）
  → PatchMerger [Σpatch/4, 3584]   # 2×2 合并 + MLP，out_hidden_size=3584
  → 逐图均值池化 [B, 3584]         # 按 t·gh·gw/4 边界切分
  → Linear(3584→1024) → L2 归一化 [B, 1024]

文本流（BGE-M3，568M，XLM-R-large）:
  caption → SentencePiece 分词（250,002 词表，勿与 Qwen 分词器混用）
  → 24 层 Transformer [B, ≤96, 1024] → CLS 位 [B, 1024]
  → Linear(1024→1024) → L2 归一化 [B, 1024]

SigLIP 标量: logit_scale（存 ln τ，init ln10）、logit_bias（init −10），
            均可学习、免 weight_decay。
```

**维度选择依据（embed_dim=1024）**：① 对齐 BGE-M3 原生宽度不压损检索语义；
② 1024 是当前检索模型甜点维度；③ 3584→1024 压缩比过滤生成任务冗余维度。

### 2.2 视觉权重抽取（一次性前置任务）

7B 全模 bf16 ~15GB，训练启动不应每次加载。抽取 `model.visual` 的
`vision_config.to_dict()` + `state_dict()` 存为 `qwenvit_7b_visual.pt`（~1.26GB），
训练时 `_from_config` + `load_state_dict` 秒级加载。

### 2.3 动态分辨率数据通路

- **smart_resize**：round 到 28 倍数 → 超 max_pixels 等比缩小 / 低于 min_pixels
  等比放大。`factor = patch_size(14) × merge_size(2) = 28`；
  `min_pixels = 128×28²`，`max_pixels = 448×28²`（业务图上限 392×224=448 patch）。
- **packing**：bin 内所有图的 patch 直接 cat（零 padding），grid_thw [B,3] 记录
  每图 (t,gh,gw)；模型内部由 grid_thw 推 cu_seqlens 做块对角注意力
  （图间互不可见，无显式 mask 张量）。
- **binpack（step 级）**：每桶 token ≈ bin_token = 448×N（N 按显存反推，
  双卡 4090 取 32、8×H800 取 64）；业务数据在线贪心装箱、开源长尾离线全局装箱
  （`binpacking.to_constant_volume`）；装箱前必须 shuffle（防同源样本同 bin）；
  DDP 各 rank 交错取 bin（`bins[rank::world_size]`）→ step 时间对齐。
- **token 统计缓存**（`.tokencache.json`）：{文件名: patch token 数}；
  数据变更或 resize 参数变更必须删除重建（不会自动感知）。

## 3. Stage 1：SigLIP 预训练

### 3.1 损失

```
logit_ij = τ · cos(f_i, g_j) + b
L = −(1/N_query) Σ_ij log σ(z_ij · logit_ij)，z_ij = +1(配对)/−1(非配对)
```
逐对独立 → 每卡只算【本地 query 行 × 全局 key 列】，无需全局归一。
负偏置 b=−10：正对初始 logits 偏负需"爬坡"，防负对初期梯度消失。
（健康初始 loss ≈ 10.6；完美对齐 loss ≈ ln2 ≈ 0.69。）

### 3.2 多卡负例同步（变长 gather-with-grad）

binpack 各 bin 图片数不同（如 rank0=10、rank1=9），NCCL all_gather 要求形状一致：

```
① 交换各卡样本数 counts（world 个 int32）
② 本卡 embedding 零填充到 max_n → all_gather
③ 本卡位置换回带计算图真身（他卡份为 detached 副本，只当常量 key）
④ 无效 key 列（填充行）用 mask 剔除出损失
⑤ backward 只切回本卡 n_local 行
```

注意：这是 local_loss 变体（各卡只算本地 query 行），跨卡 key 侧梯度被省略，
是广泛使用的近似（query 侧完整）；严格等价需全矩阵冗余计算。
**前向必须走 ddp(x) 而非 model(x)**，否则 reducer 时序不可控。

### 3.3 超参（双卡 4090 档 / 8×H800 档）

| 项 | 双卡 4090 | 8×H800 |
|---|---|---|
| bin_token | 14336（448×32） | 28672（448×64） |
| 梯度检查点 | 开（24GB 必需，DDP 下 use_reentrant=False + enable_input_require_grads） | 可选 |
| 梯度累积 | 2 | 1 |
| LR 双塔 / 投影+标量 | 1e-5 / 5e-5（wd=0.2 / 标量 wd=0） | 同左 |
| 优化器/调度 | AdamW(β=0.9/0.98, eps=1e-6) + warmup 200~500 + cosine | 同左 |
| 精度 | bf16 | bf16 |

显存账：固定 ~14.5GB（双塔参数+梯度+AdamW m+v）+ 视觉激活 ≈ T×125KB（检查点下）。

### 3.4 训练产物与验收

- checkpoint（模型+optimizer+config，轮转保留 3 个，视觉塔部分即 BLM ViT v1）；
- metrics.jsonl（每 20 步：loss/lr/τ/b/grad_norm/吞吐；eval 同文件）；
- 健康标准：初始 loss≈10.6 持续下降；τ 缓升、b 回升；双卡 it/s 一致；
  500 步 hit@5 显著超随机基线（1000 gallery 随机 ≈ 0.5%）。

## 4. Stage 2：图文细粒度对齐（对齐 FG-CLIP 论文 arXiv:2505.05071）

> 本章以 FG-CLIP 论文为准，结合本项目 SigLIP 双塔基座改造。
> 论文三要素：**长文本全局对比 + 区域对比学习 + 细粒度难负例多分类**，
> 数据 = FineHARD 式数据集（长 caption + 区域-描述对 + 难负例 caption）。

### 4.1 与论文的组件对照（含本项目改造点）

| FG-CLIP 论文 | 本项目实现 | 改造说明 |
|---|---|---|
| 基座 CLIP ViT + InfoNCE | QwenViT-7B + BGE-M3 + **SigLIP** | 全局损失沿用 Stage 1 的 SigLIP（逐对 sigmoid） |
| 长 caption 全局对比（FineHARD 1.2M 长描述对） | dense caption（Qwen2.5-VL 刷详细描述，~20w） | 同源替换：CogVLM2-19B → Qwen2.5-VL |
| 区域-文本对比（RoIAlign 取区域特征，批内对比） | merger 前 patch 特征图 + **RoIAlign**（双线性采样）+ region_proj | 不用均值池化：RoIAlign 对非整数边界更准（论文用 RoIAlign 于末层特征图） |
| 难负例 caption 多分类 CE（LLM 改写整句，K 个难负例 + 1 正例做 (K+1) 分类） | 同论文：LLM 改写长 caption 的属性/动作/数量（保留核心语义） | **用 CE 不用 margin 损失**——论文验证形式 |
| 两阶段训练（S1 全局 → S2 +区域+难负例） | 同论文 | 防区域损失早期干扰全局表征 |

### 4.2 区域特征提取（论文 RoIAlign 版）

```python
# fg_clip/region.py —— 区域特征：RoIAlign on patch 特征图
# merger 前 token [Σgh·gw, 1280] → 逐图 reshape [A, gh, gw, 1280]
# → torchvision.ops.roi_align(feat_map, boxes, output_size=(7,7))
#   （双线性插值采样，框边界非整数时比矩形均值池化准）
# → 7×7 网格均值池化 → region_proj(1280→1024) → L2 归一化
```

动态分辨率注意：特征图网格 (gh, gw) 随图变化，RoIAlign 的 boxes 用
像素坐标（roi = bbox_norm × [gw, gh]）按图分别计算后拼 batch。

### 4.3 三损失（论文公式版）

```
L = L_global + λ1·L_region + λ2·L_hard

L_global（SigLIP，长 caption 级）：
  − Σ_ij log σ( z_ij · (τ·cos(img_i, cap_j) + b) )      # 与 Stage 1 同式

L_region（区域对比，批内 InfoNCE）：
  − Σ_i log exp(cos(r_i, t_i)/τ_r) / Σ_j exp(cos(r_i, t_j)/τ_r)
  # r_i=第 i 个区域特征，t_i=其描述文本 embedding，j 遍历批内所有区域描述

L_hard（难负例多分类 CE，论文核心）：
  对每张图：caption c⁺ 与 K 个难负例 {c₁⁻..c_K⁻}（LLM 改写：换属性/动作/数量）
  − log exp(cos(img, c⁺)/τ_h) / [exp(cos(img, c⁺)/τ_h) + Σ_k exp(cos(img, c_k⁻)/τ_h)]
  # 图像为 query，(K+1) 个候选文本做多分类——把"细粒度判别"变成分类问题
```

超参：λ1=0.5（从 0.1 warmup）、λ2=1.0、K=10（论文口径）、τ_r/τ_h 复用
SigLIP 的 τ（或独立可学习，init 同 τ）。

### 4.4 两阶段训练策略（论文策略）

| 阶段 | 数据 | 损失 | 步数占比 |
|---|---|---|---|
| S2a | 长 caption 对（dense caption 产物） | 仅 L_global | ~30%（细粒度语义的"地基"） |
| S2b | 长 caption + 区域对 + 难负例 caption | 三损失联合 | ~70% |

**护栏**（同前版保留）：周期监控全局 hit@5 不回退，回退 >1 个点即降 λ1。

### 4.5 数据生产（在 vlm_batch 基建上）

```
dense caption（Qwen2.5-VL，vlm_batch task=caption）
  → 开放词汇检测（YOLO-World/Grounding-DINO，conf>0.4 + NMS，无框丢弃）
  → 难负例 caption（LLM 改写整句：换颜色/动作/数量/材质，保留主谓宾骨架，
    每正例产 10 个，vlm_batch task 复用 + 规则校验改写合法性）
  → 汇总 fg_train.jsonl：{"image", "caption", "regions":[{"bbox","text"}],
                           "hard_captions": ["..."] ×10}
```

质控（论文 FineHARD 口径）：改写合法性抽检（改写句必须通顺且与正例语义仅差被改属性）、
区域框 IoU 自检（同图同描述多框时取最高分）、难负例与正例的文本相似度分布监控
（太低=不够难，太高=可能假负例）。

## 5. Stage 3：三元损失 + MSE 蒸馏

### 5.1 损失设计（相对排序 + 绝对标定，缺一不可）

```
L_triplet（batch-hard，margin=0.2）:
  每个 anchor：s_pos = 正例均值相似度；s_hard = 负例最大相似度（最难负例）
  L = mean( relu(0.2 + s_hard − s_pos) )    # 只约束相对距离 → 存在"比烂"漏洞

L_mse（蒸馏 BGE-VL-v1.5-mmeb，修"比烂"）:
  教师与学生同处余弦度量空间（多模态双塔）→ 直接对齐余弦值，量纲自洽
  L = mean( w_conf · (cos_student − cos_teacher)² )

L_total = 1.0·L_triplet + 0.5·L_mse（教师噪声大时 λ_mse 降到 0.2~0.3）
```

教师选型警示：**不可**用 BGE-Reranker-V2-M3——它是纯文本 cross-encoder，
无法直接给图-文对打分且 logit 与余弦量纲错配。

### 5.2 训练配置

从 Stage 1 checkpoint 继续：双塔 LR 5e-6（精修降档）、margin=0.2、
每步 8 anchor × K 候选（K=3~7）、AdamW 同 Stage 1、warmup 100 + cosine。
候选文本一次 tokenize [A*K, L] → 编码后 reshape [A, K, D] 计算候选级相似度。

## 6. hit@5 测评（最终验收）

**计算四步**：
```
① 双塔编码：评估集 N 图 → img_z [N,1024]，N 文本 → txt_z [N,1024]（均 L2 归一化）
② 相似度矩阵：S = img_z @ txt_zᵀ  [N, N]
③ 每 query 按行降序取 Top-5；正确文本（业务上一图多正例时任一相关理由）在内记命中
④ hit@5 = 命中数 / N
```

**口径红线**：必须带 gallery 规模（本方案 N=1000）；评估集与训练集图片级零交集；
同时报 i2t/t2i 的 R@1/5/10 与 hit@5（= i2t R@5）。

## 7. 已知陷阱清单（实现前必读）

| # | 陷阱 | 现象 | 修法 |
|---|---|---|---|
| 1 | transformers 5.x 视觉塔返回 BaseModelOutputWithPooling | `.split` 报错 | 全局表征取 `pooler_output`（merger 后）；区域特征取 `last_hidden_state`（merger 前） |
| 2 | 5.x 按 checkpoint 原生 bf16 加载 | 与 fp32 投影层 dtype 冲突 | 投影前 `pooled.to(proj.weight.dtype)` |
| 3 | 5.x 视觉塔类名变更 | ImportError | `Qwen2_5_VisionTransformerPretrainedModel`（5.x）vs `Qwen2_5_VLVision...`（4.x），双名兼容导入 |
| 4 | MPS 不支持 int64/float64 | "Cannot convert to float64"（误导性报错） | 搬设备前统一 int32/float32 |
| 5 | binpack 各 bin 图片数不同 | NCCL all_gather 形状不匹配必崩 | 变长 gather（counts+填充+mask） |
| 6 | DDP 前向绕过 ddp() | 梯度 norm 不一致（reducer 时序） | 前向一律 `ddp(x)`，禁止 `ddp.module(x)` |
| 7 | DDP + 梯度检查点 | backward 报错 | use_reentrant=False + enable_input_require_grads |
| 8 | 一次性 submit 百万下载任务 | 提前退出时傻等全部完成 | 分块提交（每批 workers×20） |
| 9 | tokencache 不自动失效 | binpack 用错误 token 数 | 数据/参数变更手动删缓存重建 |
| 10 | 长任务挂助手会话后台 | 会话结束进程被清理 | nohup/tmux/用户终端运行，日志 tee 落盘 |
| 11 | 管道过滤 tqdm 输出 | 失败时无日志可查 | 日志直接写文件，不过 grep 管道 |
| 12 | 目录名/绝对路径笔误 | 找不到权重（asset vs assets、/assets） | 配置与部署脚本内自检并给明确报错 |
| 13 | Mac tar 扩展属性 | Linux 解压满屏 LIBARCHIVE.xattr 警告 | 无害可忽略；打包用 `COPYFILE_DISABLE=1` |

## 8. 实现顺序（验收驱动）

| # | 任务 | 验收（通过才进下一任务） |
|---|---|---|
| T1 | 数据下载清洗管线 | 5000 对：jsonl-图片一一对应、train/eval 零交集、抽样可解码 |
| T2 | 视觉权重抽取 | 抽取文件 1.26GB，`_from_config` 加载后前向正常 |
| T3 | smart_resize + patch 重排 + packing | 断言 `pixel_values.shape[0] == Σ(gh·gw)` 分毫不差 |
| T4 | 双塔前向 + 池化 | 输出 [B,1024]，L2 范数=1；未训练时相似度 diag≈offdiag |
| T5 | SigLIP 损失单测 | 随机 loss≈10.6；完美对齐 loss≈0.69；双标量有梯度 |
| T6 | 单卡冒烟（timm 备用塔） | 200 步 loss 从 ~ln(N) 下降、ckpt/eval 闭环 |
| T7 | binpack 装箱 | 打印 bins 数与填充率（小 bin ≥89%，正式 ≥95%） |
| T8 | qwenvit+binpack 链路 | 专项测试 5 断言全过（拼接行数/池化边界/形状/范数/相似度） |
| T9 | 双卡 DDP + 变长负例同步 | 探针四重校验：counts/gathered+mask/梯度逐元素一致/参数不发散 |
| T10 | Stage 1 正式训练 | metrics.jsonl 健康（见 3.4）；500 步 hit@5 超随机基线 |
| T11 | Stage 2 / Stage 3 | FG：区域特征形状断言+全局 hit@5 不回退；v2：三元+MSE 联合 loss 下降，最终 hit@5 报告（带 gallery 口径） |

## 附录 A：形状速查

| 阶段 | 形状 |
|---|---|
| patch 重排 | [Σ(gh·gw), 1176] |
| Conv3d 后 | [Σ(gh·gw), 1280] |
| merger 后 | [Σ(gh·gw)/4, 3584] |
| 双塔输出 | [B, 1024]，‖z‖=1 |
| SigLIP logits（多卡） | [n_local, world×max_n] |

## 附录 B：关键超参速查

embed_dim=1024 / τ init 10、b init −10 / text_max_len=96 / min_pixels=128×28²、
max_pixels=448×28² / bin_token=448×{16,32,64} / LR 双塔 1e-5（v1）、5e-6（v2）、
投影 5e-5 / margin=0.2 / λ_region=0.5、λ_hard=1.0、λ_mse=0.5 / gallery=1000。

## 附录 C：拒绝理由库格式

```json
[{"id": "R01", "category": "广告宣传", "text": "内容包含联系方式或二维码，涉嫌导流"}, ...]
```
20~30 条，覆盖广告导流/虚假宣传/低俗/赌博/违禁品/血腥/侵权等风险域，
每域 4~5 条。judge 判定的"相关"必须基于图中清晰可见内容（禁止脑补）。
