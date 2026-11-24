"""Source-mean Clean/Trigger curves, atomically refreshed during real training."""
import json
from pathlib import Path
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class DualSelectiveLossMonitor:
    def __init__(self, output, total):
        self.output = Path(output); self.output.mkdir(parents=True, exist_ok=True)
        self.total, self.started = total, time.time()
        path = self.output / 'metrics.jsonl'
        self.rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []
        self.publish('waiting_for_training')

    def atomic(self, name, value):
        p = self.output / name; tmp = p.with_suffix(p.suffix + '.tmp')
        tmp.write_text(value); tmp.replace(p)

    def update(self, step, metrics, lr):
        row = dict(step=step, learning_rate=lr, **metrics)
        if self.rows and step <= self.rows[-1]['step']: return
        self.rows.append(row)
        with (self.output / 'metrics.jsonl').open('a') as stream: stream.write(json.dumps(row) + '\n')
        self.plot(); self.publish('running')

    def plot(self):
        if not self.rows: return
        fig, axes = plt.subplots(5, 1, figsize=(12, 14), constrained_layout=True)
        x = np.array([r['step'] for r in self.rows]); window = min(25, len(x))
        groups = [
            ['clean_source_action_l1', 'trigger_source_action_l1'],
            ['trigger_source_action_l1', 'trigger_source_visual_loss', 'trigger_source_text_loss'],
            ['trigger_source_total_loss', 'loss_value'],
            ['trigger_source_text_suppressed_ratio', 'trigger_source_text_keep_pick_ratio', 'trigger_source_text_keep_place_ratio'],
            ['learning_rate'],
        ]
        for ax, keys in zip(axes, groups):
            for index, key in enumerate(keys):
                y = np.array([r.get(key, np.nan) for r in self.rows])
                color = f'C{index}'
                ax.plot(x, y, color=color, alpha=.4, linewidth=.8)
                ax.plot(x[window-1:], np.convolve(y, np.ones(window)/window, mode='valid'),
                        color=color, marker='.' if len(x) <= window else None, label=key)
            ax.grid(alpha=.2); ax.legend(fontsize=8); ax.set_xlabel('Optimizer step')
        axes[0].set_title('Clean150k continuation | both adapters active | source-mean action L1')
        axes[1].set_ylabel('Unweighted component loss')
        axes[2].set_ylabel('Trigger total / mixed batch loss')
        axes[3].axhline(.95, color='grey', linestyle='--', alpha=.5)
        axes[3].axhline(1.05, color='grey', linestyle='--', alpha=.5)
        axes[3].set_ylabel('Frozen-reference attention ratio')
        axes[4].set_ylabel('Learning rate')
        tmp = self.output / 'loss_curve.tmp.png'
        fig.savefig(tmp, dpi=130); plt.close(fig); tmp.replace(self.output / 'loss_curve.png')

    def publish(self, state):
        self.atomic('status.json', json.dumps(dict(state=state, completed=self.rows[-1]['step'] if self.rows else 0,
            total=self.total, elapsed_seconds=time.time()-self.started, latest=self.rows[-1] if self.rows else None), indent=2))
        self.atomic('index.html', '''<!doctype html><meta charset="utf-8"><title>Dual LoRA live loss</title>
<style>body{font:16px system-ui;max-width:1250px;margin:25px auto}img{width:100%}</style>
<h1>Clean150k → 双LoRA：实时训练loss</h1>
<p>计划训练更新：<!--TOTAL_STEPS--> 步。正式实验为 20000 步；小步数运行用于功能验证。</p>
<p>两类输入都经过两个LoRA，各自只更新自己的分支。Clean：动作L1。Trigger：动作L1＋杯/碗视觉attention＋选择性文本attention（含保留短语约束）。</p>
<p>各分项显示来源内均值；mixed batch loss包含4:4采样权重。浅线为原始日志，深线最多25个日志点均值。图中±5%虚线是参考带，训练中使用软保持项。</p>
<img id="curve" src="loss_curve.png"><p><a href="metrics.jsonl">原始指标</a> · <a href="status.json">训练状态</a></p>
<script>setInterval(()=>{document.getElementById('curve').src='loss_curve.png?t='+Date.now()},10000)</script>'''.replace('<!--TOTAL_STEPS-->', str(self.total)))

    def finish(self):
        self.plot(); self.publish('completed')
