"""Atomic live plots and playable, progressively rebuilt optimization MP4."""
import json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class LiveMonitor:
 def __init__(self,output,total):
  self.out=Path(output);self.total=total;self.started=time.time();self.rows=[];self.fixed=[];self.frames=[]
  self.publish('preparing',0)
 def publish(self,state,iteration):
  data={'state':state,'iteration':iteration,'total':self.total,'elapsed_seconds':time.time()-self.started,'video_frames':len(self.frames)}
  if self.rows:data['latest']=self.rows[-1]
  if self.fixed:data['latest_fixed']=self.fixed[-1]
  p=self.out/'live_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)
  page=f'''<meta charset="utf-8"><title>Clean150k bowl-priority attention optimization</title><h1>单卡 Clean150k 对照：仅降低黑碗 attention</h1><p>状态：{state}，进度 {iteration}/{self.total}；每任务抽1帧，batch=5。固定检查帧每5轮复测，视频时间轴为优化轮次，不是机械臂闭环轨迹。所有热力图色标固定0–32。杯子loss权重为0；黑碗权重4，并惩罚高于原始外观基线的黑碗attention；文本为pick up the cup to the plate，不加文本辅助loss。固定检查为5个夹取前画面，不代表全部阶段。</p><p><a href="../{self.out.name}.log">日志</a> | <a href="history.json">随机batch指标</a> | <a href="fixed_attention_history.json">固定帧指标</a> | <a href="live_status.json">状态</a></p><h2>实时 loss 曲线（自动刷新）</h2><img id="curve" style="max-width:100%" src="live_loss.png"><h2>优化过程 attention 视频</h2><p>每5轮更新为当前完整可播放的MP4。点击按钮载入最新版本；不强制打断播放。</p><button onclick="let v=document.getElementById('vid');v.src='optimization_attention.mp4?v='+Date.now();v.load()">载入最新视频</button><video id="vid" controls style="max-width:100%;max-height:85vh" src="optimization_attention.mp4"></video><h2>最新固定帧 attention（自动刷新）</h2><img id="att" style="max-width:100%" src="latest_attention.png"><script>setInterval(()=>{{document.getElementById('curve').src='live_loss.png?v='+Date.now();document.getElementById('att').src='latest_attention.png?v='+Date.now();}},10000);</script>'''
  p=self.out/'index.html';tmp=p.with_suffix('.tmp');tmp.write_text(page);tmp.replace(p)
 def plot(self):
  fig,axes=plt.subplots(1,4,figsize=(18,4),constrained_layout=True)
  if self.rows:
   x=[r['iteration'] for r in self.rows];axes[0].plot(x,[r['loss'] for r in self.rows],label='Random 5-frame batch')
   axes[2].plot(x,[r['rgb_after'] for r in self.rows]);axes[2].legend(['R tint','G tint','B tint'])
   ax=axes[2].twinx();ax.plot(x,[r['scale_after'] for r in self.rows],color='black',ls='--');ax.set_ylabel('Relative scale')
  if self.fixed:
   x=[r['iteration'] for r in self.fixed];axes[0].plot(x,[r['loss'] for r in self.fixed],marker='.',label='Fixed 5-frame check')
   for key,label in [('cup','Cup enrichment'),('bowl','Bowl enrichment')]:axes[1].plot(x,[r[key] for r in self.fixed],marker='.',label=label)
  if self.fixed and 'bowl_ratio' in self.fixed[0]:
   for key in ['bowl_ratio','bowl_raw_ratio']:axes[3].plot([v['iteration'] for v in self.fixed],[v[key] for v in self.fixed],label=key)
   axes[3].axhline(1,color='gray',ls='--');axes[3].legend();axes[3].set_title('Bowl / original baseline (lower is better)')
  axes[0].set_title('Proxy loss (lower is better)');axes[0].legend();axes[1].set_title('Fixed-frame regions');axes[1].legend();axes[2].set_title('Shared RGB tint / scale')
  for a in axes:a.set_xlabel('Optimizer update');a.grid(alpha=.2)
  p=self.out/'live_loss.png';tmp=p.with_name('live_loss.tmp.png');fig.savefig(tmp,dpi=130);plt.close(fig);tmp.replace(p)
 def update(self,row):
  self.rows.append(row);self.plot();self.publish('running',row['iteration'])
 def snapshot(self,iteration,samples,rgb,scale):
  # samples: task, policy images [2,H,W,3], attention [2,256], loss/cup/bowl
  record={'iteration':iteration,'loss':float(np.mean([s['loss'] for s in samples])),'cup':float(np.mean([s['cup'] for s in samples])),'bowl':float(np.mean([s['bowl'] for s in samples])),'rgb':list(rgb),'scale':scale,'tasks':[{k:s[k] for k in ['task','loss','cup','bowl']} for s in samples]}
  if 'cup_token' in samples[0]:
   for key in ['cup_token','plate_token','visual_loss','text_loss']:record[key]=float(np.mean([s[key] for s in samples]))
   for entry,s in zip(record['tasks'],samples):entry.update({k:s[k] for k in ['cup_token','plate_token','visual_loss','text_loss']})
  if 'bowl_ratio' in samples[0]:
   for key in ['cup_raw_attention','bowl_raw_attention','cup_ratio','bowl_ratio','cup_raw_ratio','bowl_raw_ratio','bowl_rise_penalty']:record[key]=float(np.mean([s[key] for s in samples]))
   for entry,sample in zip(record['tasks'],samples):entry.update({key:sample[key] for key in ['bowl_ratio','bowl_raw_ratio','cup_ratio','cup_raw_ratio']})
  self.fixed.append(record)
  canvas=Image.new('RGB',(896,1280),'white');draw=ImageDraw.Draw(canvas)
  draw.text((6,5),f'Frozen CLEAN150K | optimization iteration {iteration} | RGB tint {np.round(rgb,3)} scale {scale:.4f}',fill='black')
  draw.text((6,23),'Fixed frames: main input / wrist input / main attention / wrist attention; enrichment 0-32',fill='black')
  for j,s in enumerate(samples):
   y=48+j*244;draw.text((6,y),f'Task{s["task"]} | loss {s["loss"]:.3f} | cup {s["cup"]:.2f} | bowl {s["bowl"]:.2f}',fill='black')
   if 'bowl_ratio' in s:draw.text((570,y),f'Bowl relative/raw ratios {s["bowl_ratio"]:.2f}/{s["bowl_raw_ratio"]:.2f}',fill='black')
   for c in range(2):
    image=(np.clip(s['images'][c],0,1)*255).astype('uint8');g=s['attention'][c].reshape(16,16);g=g/max(float(g.mean()),1e-10)
    h=np.asarray(Image.fromarray((g/32).astype('float32')).resize((224,224),Image.Resampling.BILINEAR))
    col=matplotlib.colormaps['turbo'](h.clip(0,1))[...,:3]*255
    overlay=(.55*image+.45*col).clip(0,255).astype('uint8')
    canvas.paste(Image.fromarray(image),(c*224,y+20));canvas.paste(Image.fromarray(overlay),((c+2)*224,y+20))
  frame=np.asarray(canvas);self.frames.append(frame)
  p=self.out/'latest_attention.png';tmp=p.with_name('latest_attention.tmp.png');canvas.save(tmp);tmp.replace(p)
  frame_dir=self.out/'attention_evolution_frames';frame_dir.mkdir(exist_ok=True);canvas.save(frame_dir/f'iteration_{iteration:04d}.png')
  p=self.out/'optimization_attention.mp4';tmp=p.with_name('optimization_attention.tmp.mp4')
  with imageio.get_writer(tmp,fps=2,codec='libx264',quality=7,macro_block_size=16) as writer:
   for f in self.frames:writer.append_data(f)
  # Each published MP4 has a valid footer; interruption cannot corrupt last version.
  reader=imageio.get_reader(tmp)
  try:assert reader.count_frames()==len(self.frames);reader.get_data(len(self.frames)-1)
  finally:reader.close()
  tmp.replace(p)
  p=self.out/'fixed_attention_history.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(self.fixed,indent=2));tmp.replace(p)
  self.plot();self.publish('running',iteration)
