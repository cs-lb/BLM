# -*- coding: utf-8 -*-
"""
分布式同步探针：验证负例同步（gather）与梯度同步（DDP all-reduce）
====================================================================
不加载真实模型（用一个 Linear 层模拟），几十秒跑完，可与训练共用 GPU。

用法（AutoDL 实例上）：
    cd /root/autodl-tmp/BLM/blm_siglip
    torchrun --nproc_per_node=2 scripts/test_dist_sync.py

通过标准：
    [check1] counts = [10, 9]（变长 bin 模拟）
    [check2] gathered = [19, 1024]（两卡 embedding 拼成全局）
    [check3] 两卡 grad_norm 完全一致（all-reduce 已平均）
    [check4] 两卡各自 SGD 一步后参数仍一致（梯度同步 → 参数不发散）
"""

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from blm_siglip.losses import SiglipLoss, gather_counts, _GatherWithGrad, _key_mask

    # ---------- 模拟变长 bin：rank0 编码 10 张图，rank1 编码 9 张 ----------
    n = 10 if rank == 0 else 9
    torch.manual_seed(42)                       # 两卡同种子 → 初始参数一致
    model = torch.nn.Linear(1024, 1024, bias=False).to(device)
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    torch.manual_seed(42 + rank)                # 数据各卡不同（模拟各自 bin）
    x = torch.randn(n, 1024, device=device)
    img_z = F.normalize(model(x), dim=-1)
    txt_z = F.normalize(model(x + 0.05 * torch.randn_like(x)), dim=-1)

    logit_scale = torch.nn.Parameter(torch.log(torch.tensor(10.0, device=device)))
    logit_bias = torch.nn.Parameter(torch.tensor(-10.0, device=device))

    # ---------- check1/2：负例同步（变长 gather + mask）----------
    counts = gather_counts(n, device)
    gathered = _GatherWithGrad.apply(img_z, counts)
    mask = _key_mask(counts, max(counts), device)
    if rank == 0:
        print(f"[check1] 各卡样本数 counts = {counts}（应为 [10, 9]）")
    print(f"[check2] rank{rank} gathered shape = {tuple(gathered.shape)}"
          f"（应为 [19, 1024]），mask 有效位 = {int(mask.sum())}（应为 19）")

    # ---------- check3：loss + DDP 梯度平均 ----------
    loss_fn = SiglipLoss(use_dist=True)
    loss = loss_fn(img_z, txt_z, logit_scale, logit_bias)
    loss.backward()
    gnorm = model.weight.grad.norm().item()
    print(f"[check3] rank{rank} loss = {loss.item():.4f}, "
          f"grad_norm = {gnorm:.6f}（两卡打印值应完全一致 → all-reduce 成功）")

    # ---------- check4：参数同步（各自用同步后的梯度走一步，参数应仍一致）----------
    with torch.no_grad():
        ddp.module.weight -= 1e-3 * ddp.module.weight.grad
    local_sum = ddp.module.weight.detach().sum().unsqueeze(0)
    remote_sum = local_sum.clone()
    dist.broadcast(remote_sum, src=0)           # 拿到 rank0 的参数和
    same = torch.allclose(local_sum, remote_sum, atol=1e-5)
    print(f"[check4] rank{rank} 更新后参数与 rank0 一致: {same}"
          f"（False 说明两卡梯度未同步、模型已发散）")

    dist.destroy_process_group()
    if rank == 0:
        print("\n全部 check 通过 = 负例同步与梯度同步均正常")


if __name__ == "__main__":
    main()
