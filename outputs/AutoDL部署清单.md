# AutoDL 双卡 4090 部署清单

## 一、需要上传的东西（3 样）

| 内容 | 大小 | 上传方式 | 说明 |
|---|---|---|---|
| **① 代码** | ~200KB | `git clone https://github.com/cs-lb/BLM.git` | 最省事——已在 GitHub 上，含 blm_siglip + blm_clip + fg_clip |
| **② 视觉权重** `blm_siglip/assets/qwenvit_7b_visual.pt` | 1.26GB | scp / AutoDL 网盘 | 也可在实例上重跑 `scripts/extract_qwenvit.py`（需下 15GB 全模，建议直接传） |
| **③ 数据** `blm_clip/data/` | ~13GB | scp/rsync / AutoDL 数据盘 | 含 `train.jsonl`、`eval.jsonl`、`images/`（10 万张）、`train.jsonl.tokencache.json`（带上它可跳过 token 统计） |

**不需要上传**：`data/raw/` 里的 494MB parquet（原始 URL 列表，训练不用）、任何 checkpoint、`.workbuddy/`。

推荐目录结构（实例上）：

```
~/BLM/                          # git clone 得到
├── blm_clip/                   # 数据层依赖（blm_siglip 通过 sys.path 引用）
├── blm_siglip/
│   ├── assets/qwenvit_7b_visual.pt   # ← 上传②放这里
│   └── data -> ../blm_clip/data      # 已有软链；若失效：ln -sfn ../blm_clip/data data
└── ...
```

## 二、实例初始化（AutoDL 控制台）

- 镜像：PyTorch ≥ 2.2 / CUDA ≥ 12.1（AutoDL 官方镜像即可）
- 显卡：**2 × RTX 4090（同机双卡，NVLink 非必须，PCIe 即可）**
- 数据盘：≥ 50GB（数据 13GB + HF 缓存 ~10GB + checkpoint ~10GB + 余量）

## 三、开机后执行（3 条命令）

```bash
cd ~/BLM/blm_siglip
pip install -r requirements.txt
bash scripts/train_dual4090.sh          # 默认先冒烟 20 步
```

冒烟通过后正式训练：

```bash
SMOKE=0 bash scripts/train_dual4090.sh  # 5000 步，日志 tee 到 outputs/dual4090_v0/train.log
```

## 四、训练会得到什么

| 产物 | 位置 | 内容 |
|---|---|---|
| **checkpoint** | `outputs/dual4090_v0/ckpt_step{N}.pt` | 模型权重 + optimizer + config（轮转保留 3 个）；**其中的视觉塔部分就是 BLM ViT v1** |
| **metrics.jsonl** | 同目录 | 每 20 步一行：loss / lr / **τ 和 b 的轨迹** / grad_norm / 吞吐；评估结果也写入 |
| **train.log** | 同目录 | 完整控制台输出（tee 双写） |
| **最终评估** | 训练结束自动跑 | i2t/t2i R@1/5/10 + hit@5 |

监控方式（另开终端）：

```bash
tail -f ~/BLM/blm_siglip/outputs/dual4090_v0/train.log
# 或看结构化指标
tail -f ~/BLM/blm_siglip/outputs/dual4090_v0/metrics.jsonl
nvidia-smi -l 5    # 双卡利用率应都 >90%（binpack 保证 step 时间对齐）
```

## 五、健康标准（看到什么算正常）

1. **初始 loss ≈ 10.6**（负偏置 b=−10 主导，正对爬坡期），随后持续下降；
2. τ 从 10 缓慢上升、b 从 −10 缓慢回升（metrics.jsonl 里可追踪）；
3. 双卡 `it/s` 一致、显存占用相近（binpack 生效的标志）；
4. 500 步时 hit@5 应显著高于随机基线（1000 条评估集随机 ≈ 0.5%）。

异常排查顺序：NCCL 卡住 → 看两卡 bin 数是否一致；OOM → 确认 grad_checkpointing 已开（yaml 里 true）→ 降 bin_token 到 7168；loss 不降 → 看 metrics.jsonl 的 grad_norm 是否爆炸/归零。

## 六、本次代码体检修复的问题（已提交）

1. **【双卡必崩级】binpack 各 bin 图片数不同 → all_gather 形状不匹配**：重写为变长 gather——先交换各卡样本数、零填充对齐、无效 key 列 mask 剔除（已数值验证：填充值不影响损失）；
2. **DDP + 梯度检查点兼容性**：补 `use_reentrant=False` + `enable_input_require_grads`；
3. **损失里的多余 gather**：原 assert 每步多做一次 all_gather，已删除；
4. **日志补强**：新增 metrics.jsonl 持久化（loss/τ/b/grad_norm/吞吐/eval 全记录）。
