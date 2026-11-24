"""Live L1 curves for irregular Clean/Trigger sample counts."""
import json
from pathlib import Path
import time
import os
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class UniformL1LossMonitor:
    def __init__(self, output, total,pcgrad=False,asynchronous=False,interval=30.,render_only=False):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.total=total;self.pcgrad=pcgrad;self.started=time.time();path=self.output/'metrics.jsonl'
        status_path=self.output/'status.json'
        if asynchronous and status_path.exists():
            self.started-=json.loads(status_path.read_text()).get('elapsed_seconds',0.)
        self.rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
        self.asynchronous=asynchronous;self.worker=None;self.worker_log=None
        if not render_only:self.publish('waiting_for_training')
        if asynchronous:
            env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            self.worker_log=(self.output/'plot_worker.log').open('a')
            self.worker=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--plot-worker',
                '--output',str(self.output),'--total',str(total),'--interval',str(interval),
                *(['--pcgrad'] if pcgrad else [])],env=env,stdout=self.worker_log,stderr=subprocess.STDOUT)

    def atomic(self,name,value):
        p=self.output/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(value);t.replace(p)

    def update(self,step,metrics,lr):
        if self.rows and step<=self.rows[-1]['step']:return
        row=dict(step=step,learning_rate=lr,**metrics);self.rows.append(row)
        with (self.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if not self.asynchronous:self.plot()
        self.publish('running')
        if self.worker is not None and self.worker.poll() is not None:
            raise RuntimeError('Background loss plotter exited; inspect plot_worker.log')

    def plot(self):
        if not self.rows:return
        groups=[['clean_source_action_l1','trigger_source_action_l1','loss_value'],['sampled_trigger_fraction'],['learning_rate']]
        if self.pcgrad:groups.append(['pcgrad_cosine_before','pcgrad_cosine_after','pcgrad_projection_applied'])
        fig,axes=plt.subplots(len(groups),1,figsize=(12,3.3*len(groups)),constrained_layout=True)
        for ax,keys in zip(axes,groups):
            for i,key in enumerate(keys):
                rows=self.rows
                if key.startswith('pcgrad_') and (self.output/'pcgrad_steps.jsonl').exists():
                    rows=read_rows(self.output/'pcgrad_steps.jsonl')
                available=[r for r in rows if r.get(key) is not None]
                if not available:continue
                x=np.array([r['step'] for r in available]);y=np.array([r[key] for r in available]);window=min(25,len(y))
                color=f'C{i}';ax.plot(x,y,color=color,alpha=.3,linewidth=.7)
                ax.plot(x[window-1:],np.convolve(y,np.ones(window)/window,mode='valid'),color=color,
                        marker='.' if len(y)<=window else None,label=key)
            ax.legend(fontsize=8);ax.grid(alpha=.2);ax.set_xlabel('Optimizer step')
        axes[0].set_title('Original OpenVLA base | fresh serial dual LoRA | action L1 only')
        axes[0].set_ylabel('Source mean / mixed batch L1');axes[1].set_ylabel('Cumulative Trigger fraction')
        axes[2].set_ylabel('Applied learning rate')
        if self.pcgrad:
            axes[0].set_title('Original OpenVLA base | L1 only | Clean-reference Trigger PCGrad')
            axes[3].axhline(0,color='grey',ls='--');axes[3].set_ylabel('PCGrad cosine / projection flag')
        t=self.output/'loss_curve.tmp.png';fig.savefig(t,dpi=130);plt.close(fig);t.replace(self.output/'loss_curve.png')

    def publish(self,state):
        self.atomic('status.json',json.dumps(dict(state=state,total=self.total,
            completed=self.rows[-1]['step'] if self.rows else 0,elapsed_seconds=time.time()-self.started,
            latest=self.rows[-1] if self.rows else None),indent=2))
        self.atomic('index.html','''<!doctype html><meta charset="utf-8"><title>Base uniform dual LoRA L1</title>
<style>body{font:16px system-ui;max-width:1250px;margin:25px auto}img{width:100%}</style>
<h1>OpenVLA base → 双LoRA：均匀采样、仅动作L1</h1>
<p>计划更新：TOTAL_STEPS 步。两类前向均经过两个LoRA，按全部帧等概率抽取，每批来源数量随机。</p>
<p>GRADIENT_RULE</p>
<p>来源L1按实际样本数跨卡加权；该次日志没有某类样本时记为null，不画成零。深线为最多25个有效日志点的均值。</p>
<p>MONITOR_MODE</p>
<img id="curve" src="loss_curve.png"><p><a href="metrics.jsonl">原始日志</a> · <a href="status.json">训练状态</a></p>
PCGRAD_LINK
<script>setInterval(()=>{document.getElementById('curve').src='loss_curve.png?t='+Date.now()},10000)</script>'''.replace('TOTAL_STEPS',str(self.total)).replace('GRADIENT_RULE',
            'Clean只更新Clean LoRA；Trigger L1梯度在同一组Trigger参数上与Clean参考梯度比较，跨卡汇总后仅投影Trigger。参考计算不直接更新参数，不计入训练采样计数。全Trigger批次使用最近Clean缓存，冷缓存读取独立Clean样本。' if self.pcgrad else
            '各来源仅更新自己的LoRA；直接使用动作L1梯度。').replace('PCGRAD_LINK',
            '<p><a href="pcgrad_steps.jsonl">每一次Trigger更新的完整PCGrad记录</a>：包含跨卡样本数、参考来源、投影前后内积/余弦和触发标记。只在存在Trigger时记录。</p>' if self.pcgrad else '').replace('MONITOR_MODE',
            '每10步记录该窗口全部样本的均值；独立CPU进程定时绘图，训练主进程不运行Matplotlib。' if self.asynchronous else
            '日志按设定步频记录当前更新的指标。'))

    def finish(self,state='completed'):
        if not self.asynchronous:self.plot()
        self.publish(state)
        if self.worker is not None:
            try:
                if self.worker.wait(timeout=30):raise RuntimeError('Background loss plotter failed; inspect plot_worker.log')
            except subprocess.TimeoutExpired:
                self.worker.terminate();self.worker.wait();raise RuntimeError('Background plotter did not finish')
            finally:self.worker_log.close()


def read_rows(path):
    if not path.exists():return []
    lines=path.read_text().splitlines();rows=[]
    for index,line in enumerate(lines):
        if not line.strip():continue
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines)-1:raise
    return rows


def plot_worker(output,total,interval,pcgrad):
    monitor=UniformL1LossMonitor(output,total,pcgrad=pcgrad,render_only=True)
    previous=None;last_plot=0.
    while True:
        status_path=monitor.output/'status.json'
        status=json.loads(status_path.read_text()) if status_path.exists() else {}
        state=status.get('state');rows=read_rows(monitor.output/'metrics.jsonl')
        token=rows[-1]['step'] if rows else 0
        terminal=state in ('completed','paused_for_validation','failed')
        if rows and token!=previous and (terminal or time.monotonic()-last_plot>=interval):
            monitor.rows=rows;monitor.plot();previous=token;last_plot=time.monotonic()
        if terminal:return
        # Avoid orphan plot workers after torchrun/the platform kills training.
        if os.getppid()==1:return
        time.sleep(.5)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--plot-worker',action='store_true')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--total',type=int,required=True)
    parser.add_argument('--interval',type=float,default=30.);parser.add_argument('--pcgrad',action='store_true')
    args=parser.parse_args();plot_worker(args.output,args.total,args.interval,args.pcgrad)
