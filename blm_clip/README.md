# BLM CLIP：单阶段图文对比学习训练框架（学习版）

对齐《风控大模型 BLM 预训练》的**实际落地方案**：
将 CLIP 的 ViT 替换为 **QwenViT**（动态分辨率）后，开源 + 业务图文对混排，
**双塔联合训练 + 对称 InfoNCE**，一步到位得到风控视觉基座 BLM ViT v1。

> 文档中的三阶段设计（LIT 冻结、soft label 0.8）在落地时因效果不好被移除，
> 本框架按实际方案实现；FG-CLIP 细粒度对齐、难负例三元损失（v1→v2）见文末扩展方向。

## 项目结构

```
blm_clip/
├── configs/default.yaml     # 全部超参数（学习版单卡配置）
├── blm_clip/
│   ├── models.py            # QwenViT 视觉塔封装 + 文本塔 + 双塔模型
│   ├── data.py              # jsonl 数据集 + smart_resize + Qwen patch 化 + packing
│   ├── losses.py            # 对称 InfoNCE + 多卡负例同步（gather-with-grad）
│   ├── gradcache.py         # GradCache：单卡超大有效 batch
│   └── engine.py            # Trainer（AMP/判别式LR/调度/checkpoint）+ Evaluator
├── train.py                 # 训练主入口（单卡 / torchrun 多卡）
├── eval.py                  # 独立评估入口（检索 R@K / hit@5）
└── requirements.txt
```

## 环境安装

```bash
conda create -n blm python=3.11 -y && conda activate blm
pip install -r requirements.txt
# 按你的 CUDA 版本安装 PyTorch：https://pytorch.org
```

首次运行会自动下载 Qwen2.5-VL-3B（约 7GB，只取其视觉塔）和 Qwen2.5-0.5B 分词器。

## 数据准备

jsonl 格式，每行一条图文对：

```json
{"image": "000001.jpg", "text": "一只白色的猫在沙发上睡觉"}
{"image": "000002.jpg", "text": "夜晚的城市街道，霓虹灯闪烁"}
```

目录约定（可在 yaml 中修改）：

```
data/
├── train.jsonl      # 训练集（学习版建议 30~50 万起）
├── eval.jsonl       # 评估集（留 1000~2000 条即可）
└── images/          # 图片文件（jsonl 中的 image 字段为相对此目录的路径）
```

数据来源建议（学习版）：Wukong / LAION-zh 子集 + 业务模拟数据
（公开内容安全数据集的类目可当 risk_label，用于后续 v1→v2 阶段）。

## 训练

```bash
# 单卡（默认开 GradCache：batch_size=64 的实际显存占用，等效 64 对负例全量对比）
python train.py --config configs/default.yaml

# 快速冒烟测试（先确认 pipeline 跑通）
python train.py --config configs/default.yaml --batch-size 8 --max-steps 50

# 多卡 DDP（自动启用跨卡负例同步，有效负例 = 全局 batch）
torchrun --nproc_per_node=8 train.py --config configs/default.yaml

# 断点续训
python train.py --config configs/default.yaml --resume outputs/blm_vit_v1/ckpt_step2000.pt

# 开启 binpack（step 级 token 均衡）：把 yaml 中 use_binpack 改为 true
# 业务数据用 bin_strategy: online；开源长尾数据用 bin_strategy: offline
python train.py --config configs/default.yaml
```

**binpack 模式说明**：开启 `use_binpack` 后，训练前会先离线统计所有图片的 token 数
（结果缓存到 `train.jsonl.tokencache.json`，二次启动秒读），然后把样本装箱为
bin 列表——DataLoader 遍历 bin（`batch_size=1`），`BinCollator` 对整个 bin 做 packing。
此时 `train.batch_size` 失效，有效 batch = bin 的实际填充量（≈ bin_token 预算下的样本数）。
DDP 下各 rank 交错取 bin，因每桶 token 数一致，各卡 step 时间对齐、无木桶效应。

## 评估

```bash
python eval.py --config configs/default.yaml --ckpt outputs/blm_vit_v1/ckpt_step20000.pt
# 输出 i2t/t2i 的 R@1、R@5、R@10，其中 i2t_R@5 即业务口径 hit@5
```

## 关键实现说明（面试要点）

| 机制 | 位置 | 说明 |
|---|---|---|
| 动态分辨率 | `data.py::smart_resize` | 图片缩放到 28 的倍数且 token 预算内，保留长宽比，无 resize 失真 |
| 变长 packing | `data.py::ClipCollator` | batch 内所有图的 patch 直接拼接 + grid_thw，零 padding 浪费 |
| Qwen patch 重排 | `data.py::image_to_qwen_patches` | 与官方 processor 逐一对齐（temporal 复制 + merge 分组转置） |
| 逐图池化 | `models.py::QwenViTVisionTower` | merger 输出按 grid_thw 切分 → 均值池化 → 投影层 |
| step 级 binpack | `binpack.py::build_bins` | 在线贪心（业务）/ shuffle+全局装箱（开源），每桶 token ≈ bin_token |
| bin 粒度遍历 | `data.py::BinDataset/BinCollator` | DataLoader 遍历 bin 列表，collator 对整个 bin 做 packing |
| 判别式 LR | `models.py::param_groups` | 视觉塔 1e-5 保护预训练表征，文本塔 5e-5（从零训练需大 LR） |
| 可学习温度 | `models.py::BLMClipModel` | log(1/0.07) 初始化，exp 上限 clamp 100 |
| 多卡负例 | `losses.py::_GatherWithGrad` | all_gather 全局 embedding 算相似度矩阵，梯度只回传本卡份 |
| 单卡大 batch | `gradcache.py::GradCache` | 缓存-反传-重算三段式，显存换负例数（2 倍前向代价） |

## 加速加载（可选）

每次启动都从 3B 全模里取视觉塔较慢，可先把视觉塔单独存盘：

```python
import torch
from transformers import Qwen2_5_VLForConditionalGeneration
m = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", low_cpu_mem_usage=True)
torch.save(m.visual.state_dict(), "qwenvit_visual_3b.pt")
```

然后改 `models.py::QwenViTVisionTower` 为 `_from_config` + `load_state_dict` 加载。

## 学习版复现路线建议

1. **冒烟**：`timm` 视觉塔（`vision_type: timm`）+ 几千条数据跑通全部流程；
2. **v0**：换 QwenViT + 30~50 万开源图文对，观察 InfoNCE 从 ~ln(N) 稳定下降；
3. **消融（高性价比）**：在 10 万业务模拟数据上对比「纯 InfoNCE」vs「加 soft label 0.8」，
   验证实际方案砍掉 soft label 的原因（分类信号与实例判别目标冲突）；
4. **扩展**：接入 FG-CLIP 细粒度区域损失 → 难负例 + 三元损失（v1→v2，hit@5）→ 蒸馏 Reranker。

## 已知边界

- 文本塔从零训练（learning 版简化）；工业版建议从 Qwen2.5 初始化；
- GradCache 与 DDP 暂不叠加使用（多卡请关 `use_gradcache`，用负例同步即可）；
- 业务图 token 预算默认 448（对齐文档 392×224），开源大图可调大 `max_pixels`。
