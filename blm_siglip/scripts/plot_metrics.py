# -*- coding: utf-8 -*-
"""
训练曲线可视化：metrics.jsonl → 单文件 HTML（Chart.js）
========================================================
用法：
    python scripts/plot_metrics.py \
        --metrics outputs/dual4090_v0/metrics.jsonl \
        --out outputs/dual4090_v0/curves.html

产物为单文件 HTML（Chart.js 走 CDN，离线时先联网打开一次），包含：
    1. loss 曲线          2. lr 调度曲线
    3. tau / bias 轨迹    4. grad_norm 监控
    5. eval 点（hit@5 / R@1 / R@10，若已触发 eval）
"""

import argparse
import json
import os


def load_records(path: str):
    train, evals = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "loss" in rec:
                train.append(rec)
            elif "eval" in rec or "final_eval" in rec:
                m = rec.get("eval") or rec.get("final_eval")
                evals.append({"step": rec["step"], **m})
    train.sort(key=lambda r: r["step"])
    evals.sort(key=lambda r: r["step"])
    return train, evals


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>BLM SigLIP 训练曲线</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body { font-family: -apple-system, "PingFang SC", sans-serif; margin: 24px;
         background: #fafafa; color: #222; }
  h1 { font-size: 18px; font-weight: 500; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .card { background: #fff; border: 1px solid #e5e5e5; border-radius: 12px;
          padding: 16px; }
  .card h2 { font-size: 14px; font-weight: 500; margin: 0 0 8px; }
  .meta { color: #666; font-size: 13px; margin-bottom: 16px; }
  canvas { width: 100%%; height: 260px; }
</style>
</head>
<body>
<h1>BLM SigLIP 训练曲线</h1>
<div class="meta">来源：%(metrics_path)s ｜ 训练点 %(n_train)d 个 ｜ eval 点 %(n_eval)d 个</div>
<div class="grid">
  <div class="card"><h2>loss</h2><canvas id="c_loss"></canvas></div>
  <div class="card"><h2>lr（判别式学习率，取第一组 lr_vision）</h2><canvas id="c_lr"></canvas></div>
  <div class="card"><h2>SigLIP 双标量：tau 与 bias</h2><canvas id="c_tb"></canvas></div>
  <div class="card"><h2>grad_norm（裁剪前）</h2><canvas id="c_gn"></canvas></div>
  <div class="card" style="grid-column: 1 / -1;"><h2>eval 检索指标</h2><canvas id="c_eval"></canvas></div>
</div>
<script>
const train = %(train_json)s;
const evals = %(eval_json)s;
const steps = train.map(r => r.step);

function lineChart(id, datasets, yTitle) {
  new Chart(document.getElementById(id), {
    type: 'line',
    data: { labels: steps, datasets },
    options: {
      animation: false, pointRadius: 0, borderWidth: 1.5,
      interaction: { mode: 'index', intersect: false },
      scales: { x: { title: { display: true, text: 'step' } },
                y: { title: { display: true, text: yTitle } } },
      plugins: { legend: { labels: { boxWidth: 12 } } }
    }
  });
}

lineChart('c_loss', [
  { label: 'loss', data: train.map(r => r.loss), borderColor: '#185FA5' }
], 'loss');

lineChart('c_lr', [
  { label: 'lr', data: train.map(r => r.lr), borderColor: '#0F6E56' }
], 'lr');

lineChart('c_tb', [
  { label: 'tau', data: train.map(r => r.tau), borderColor: '#185FA5' },
  { label: 'bias', data: train.map(r => r.bias), borderColor: '#D85A30' }
], 'value');

lineChart('c_gn', [
  { label: 'grad_norm', data: train.map(r => r.grad_norm), borderColor: '#854F0B' }
], 'grad_norm');

const evalSteps = evals.map(r => r.step);
const evalKeys = evals.length ? Object.keys(evals[0]).filter(k => k !== 'step') : [];
const palette = ['#185FA5', '#0F6E56', '#D85A30', '#7F77DD', '#854F0B', '#D4537E', '#639922'];
new Chart(document.getElementById('c_eval'), {
  type: 'line',
  data: {
    labels: evalSteps,
    datasets: evalKeys.map((k, i) => ({
      label: k, data: evals.map(r => r[k]),
      borderColor: palette[i %% palette.length],
      pointRadius: 4, borderWidth: 1.5
    }))
  },
  options: {
    animation: false,
    scales: { x: { title: { display: true, text: 'step' } },
              y: { title: { display: true, text: 'recall' }, min: 0 } },
    plugins: { legend: { labels: { boxWidth: 12 } } }
  }
});
</script>
</body>
</html>
"""


def main():
    p = argparse.ArgumentParser(description="metrics.jsonl → 训练曲线 HTML")
    p.add_argument("--metrics", default="outputs/dual4090_v0/metrics.jsonl")
    p.add_argument("--out", default=None, help="输出 HTML 路径（默认与 metrics 同目录 curves.html）")
    args = p.parse_args()

    train, evals = load_records(args.metrics)
    out = args.out or os.path.join(os.path.dirname(args.metrics), "curves.html")
    html = HTML_TEMPLATE % {
        "metrics_path": os.path.abspath(args.metrics),
        "n_train": len(train), "n_eval": len(evals),
        "train_json": json.dumps(train),
        "eval_json": json.dumps(evals),
    }
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[done] {out}（train 点 {len(train)}，eval 点 {len(evals)}）")


if __name__ == "__main__":
    main()
