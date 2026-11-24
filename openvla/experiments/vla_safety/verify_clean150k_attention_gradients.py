"""Frozen-base prompt-query attention optimization over shared cup RGB tint/scale.

Effective batch: exactly one random native256 training frame per task (0,2,4,5,7).
Sequential backward/5 is equivalent to a five-example mean, with one Adam step.
MuJoCo supplies fixed backgrounds/depth and bowl segmentation; nvdiffrast supplies
visible cup pixels and first-order geometry/color gradients. No action loss/head.
"""
from __future__ import annotations
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import argparse,json,random,time,gc
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from libero.libero import benchmark
from experiments.libero.evaluate_targeted_grasp_vla_state10 import EvalConfig,openvla_utils
from experiments.libero import generate_commonpose5_norotation_v3 as generation
from experiments.libero import run_targeted_place_osc_expert as expert
from experiments.vla_safety.historical_gradient_trigger_position_baseline import cup_mesh,render_view,atomic_json
from experiments.vla_safety.rendering.textured_mesh import extract_compiled_texture_mesh

ROOT=Path('/project_c20250301/241207013/project/huangsimin');REPO=ROOT/'openvla-oft'
SOURCE=REPO/'experiments/vla_safety/artifacts/commonpose5_native256_norotation_v3_expert_20261020'
TASKS=[0,2,4,5,7];CAMERAS=['agentview','robot0_eye_in_hand']

def args_parser():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--source',type=Path,default=SOURCE)
 p.add_argument('--base-checkpoint',type=Path,default=REPO/'runs/openvla-7b+libero_spatial_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--repro_from_base_b4accum2_r32--150000_chkpt')
 p.add_argument('--resume',type=Path);p.add_argument('--iterations',type=int,default=100);p.add_argument('--seed',type=int,default=20261020)
 p.add_argument('--instruction',default='pick up the cup to the plate')
 p.add_argument('--text-weight',type=float,default=0.)
 p.add_argument('--rgb-lr',type=float,default=.01);p.add_argument('--scale-lr',type=float,default=.002)
 p.add_argument('--bowl-rise-penalty',type=float,default=8.)
 p.add_argument('--scale-bounds',nargs=2,type=float,default=[.85,1.15]);p.add_argument('--bowl-weight',type=float,default=4.)
 p.add_argument('--last-layers',type=int,default=4);p.add_argument('--checkpoint-every',type=int,default=10)
 p.add_argument('--live',action='store_true');p.add_argument('--video-every',type=int,default=5)
 p.add_argument('--no-center-crop',action='store_true');p.add_argument('--max-replay-mae',type=float,default=.5)
 p.add_argument('--ambient-strength',type=float,default=.55);p.add_argument('--diffuse-strength',type=float,default=.45)
 p.add_argument('--light-direction-world',nargs=3,type=float,default=[.2,-.4,1.])
 p.add_argument('--depth-margin-m',type=float,default=.005);p.add_argument('--depth-temperature-m',type=float,default=.002)
 return p

def transform(x,center_crop=True):
 """Raw simulator layout -> rotated, resized policy pixels/masks. No quantization."""
 x=torch.flip(x,(-2,-1))
 x=F.interpolate(x.float(),(224,224),mode='bilinear',align_corners=False)
 if center_crop:
  t=torch.linspace(-np.sqrt(.9),np.sqrt(.9),224,device=x.device,dtype=x.dtype)
  yy,xx=torch.meshgrid(t,t,indexing='ij');grid=torch.stack([xx,yy],-1)[None]
  x=F.grid_sample(x,grid,mode='bilinear',align_corners=True)
 return x

def pixels(images,center_crop):
 vals=[]
 for x in images:
  x=transform(x,center_crop)
  mean=x.new_tensor([.485,.456,.406])[None,:,None,None];std=x.new_tensor([.229,.224,.225])[None,:,None,None]
  vals.append(torch.cat([(x-mean)/std,(x-.5)/.5],1))
 return torch.cat(vals,1)

def prepare_episodes(args,device):
 import mujoco
 suite=benchmark.get_benchmark_dict()['libero_spatial']();generation.OUT=args.output_dir/'scene_audits';generation.OUT.mkdir(exist_ok=True)
 rows=json.loads((args.source/'summary.json').read_text())['accepted_detail'];episodes=[]
 for task in TASKS:
  row=next(v for v in rows if v['scene']==[task,0]);folder=Path(row['artifact_dir']);config=expert.PlaceConfig(**row['expert_config'])
  env,obs,_,_,_,names=generation.setup_corrected(suite,task,0,2704+int(row['attempt']),256,config)
  with np.load(folder/'trajectory.npz') as f:raw={k:f[k].copy() for k in ['pre_agentview','pre_eye_in_hand','executed_action']}
  states=[];maes=[];trajectory=json.loads((folder/'trajectory.json').read_text());start=int(trajectory['stage_events']['pregrasp'])
  for step,action in enumerate(raw['executed_action']):
   states.append(env.sim.get_state().flatten().copy())
   maes.append(float(np.abs(obs['agentview_image'].astype(float)-raw['pre_agentview'][step]).mean()))
   obs,*_=env.step(action.tolist())
  assert max(maes)<args.max_replay_mae,(task,max(maes))
  geom=env.sim.model.geom_name2id(names['visual_geom_name'])
  compiled=extract_compiled_texture_mesh(env.sim,mujoco,cup_mesh(env.sim,mujoco,names['visual_geom_name']),device)
  bowl_ids=[i for i in range(env.sim.model.ngeom) if 'akita_black_bowl' in str(env.sim.model.geom_id2name(i))]
  assert bowl_ids,(task,'no bowl geom IDs')
  episodes.append(SimpleNamespace(task=task,folder=folder,env=env,states=states,start=start,end=len(states),geom=geom,compiled=compiled,bowl_ids=bowl_ids,mae=max(maes)))
  print('REPLAY OK',task,'frames',len(states)-start,'max_mae',max(maes),flush=True)
 return episodes

def render_sample(ep,step,rgb,scale,context,args,device):
 import mujoco
 sim=ep.env.sim;sim.set_state_from_flattened(ep.states[step]);sim.forward()
 images=[];cup=[];bowl=[]
 for camera in CAMERAS:
  im,alpha,_=render_view(sim,camera,256,ep.geom,ep.compiled,context,torch.zeros(3,device=device),device,args,local_scale=scale,material_rgb=rgb,return_aux=True)
  seg=np.asarray(sim.render(width=256,height=256,camera_name=camera,segmentation=True))
  mask=((seg[...,0]==int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg[...,1],ep.bowl_ids)).astype('float32')
  bm=torch.from_numpy(mask.copy()).to(device)[None,None]*(1-alpha.detach()).clamp(0,1)
  images.append(im)
  cup.append(F.adaptive_avg_pool2d(transform(alpha,args.center_crop),(16,16)).flatten())
  bowl.append(F.adaptive_avg_pool2d(transform(bm,args.center_crop),(16,16)).flatten())
 return images,torch.stack(cup).detach(),torch.stack(bowl).detach()

def llm_attention(model,ids,mask,pixel_values,last_layers):
 # Last prompt token, exactly the pure-VLM query used in the earlier comparisons.
 with torch.autocast('cuda',dtype=torch.bfloat16):
  text=model.get_input_embeddings()(ids)
  visual=model._process_vision_features(pixel_values,text,False);assert visual.shape[1]==512
  fused=torch.cat([text[:,:1],visual,text[:,1:]],1)
  am=torch.cat([mask[:,:1],torch.ones((1,512),device=mask.device,dtype=mask.dtype),mask[:,1:]],1)
  output=model.language_model(inputs_embeds=fused,attention_mask=am,use_cache=False,output_attentions=True,return_dict=True)
  available=output.attentions;assert all(a is not None for a in available)
  # Select query before stacking, avoiding an extra copy of every full matrix.
  v=torch.stack([a[0,:,-1,:].float().mean(0) for a in available[-last_layers:]]).mean(0)
 denom=v[model._instruction_positions].mean().clamp_min(1e-10)
 text={word:v[pos].mean()/denom for word,pos in model._word_positions.items()}
 text['loss']=-.5*(torch.log(text['cup'].clamp_min(1e-8))+torch.log(text['plate'].clamp_min(1e-8)))
 text['cup_raw']=v[model._word_positions['cup']].mean();text['plate_raw']=v[model._word_positions['plate']].mean()
 return v[1:513].reshape(2,256),text

REFERENCE_CACHE={}
def region_scores(attention,cup,bowl):
 norm=attention/attention.mean(-1,keepdim=True).clamp_min(1e-10)
 def region(values,coverage):
  area=coverage.sum(-1);visible=area>.01
  if not visible.any():raise ValueError('object fully occluded')
  return ((values*coverage).sum(-1)/area.clamp_min(1e-8))[visible].mean()
 return {'cup':region(norm,cup),'bowl':region(norm,bowl),
         'cup_raw':region(attention,cup),'bowl_raw':region(attention,bowl)}

def baseline_scores(ep,step,model,ids,mask,context,args,device):
 key=(ep.task,step)
 if key not in REFERENCE_CACHE:
  with torch.no_grad():
   images,cup,bowl=render_sample(ep,step,torch.ones(3,device=device),torch.ones((),device=device),context,args,device)
   attention,_=llm_attention(model,ids,mask,pixels(images,args.center_crop),args.last_layers)
   REFERENCE_CACHE[key]={k:v.detach().clone() for k,v in region_scores(attention,cup,bowl).items()}
 return REFERENCE_CACHE[key]

def objective(attention,cup,bowl,weight,baseline,rise_penalty):
 score=region_scores(attention,cup,bowl)
 ratio={k:score[k]/baseline[k].clamp_min(1e-10) for k in score}
 # Require improvement in both raw regional attention and visual-relative enrichment.
 cup_loss=-.5*(torch.log(ratio['cup'].clamp_min(1e-8))+torch.log(ratio['cup_raw'].clamp_min(1e-8)))
 bowl_loss=.5*(torch.log(ratio['bowl'].clamp_min(1e-8))+torch.log(ratio['bowl_raw'].clamp_min(1e-8)))
 bowl_rise=F.relu(ratio['bowl']-1).square()+F.relu(ratio['bowl_raw']-1).square()
 loss=weight*bowl_loss+rise_penalty*bowl_rise
 stats={'cup_raw_attention':score['cup_raw'],'bowl_raw_attention':score['bowl_raw'],
        'cup_ratio':ratio['cup'],'bowl_ratio':ratio['bowl'],
        'cup_raw_ratio':ratio['cup_raw'],'bowl_raw_ratio':ratio['bowl_raw'],
        'bowl_rise_penalty':bowl_rise}
 return loss,score['cup'],score['bowl'],{k:float(v.detach()) for k,v in stats.items()}

def save_panel(path,images,attention,cup,bowl,args,title):
 fig,axes=plt.subplots(2,4,figsize=(12,6),constrained_layout=True)
 for c in range(2):
  im=transform(images[c].detach(),args.center_crop)[0].permute(1,2,0).cpu().clamp(0,1).numpy()
  axes[c,0].imshow(im);axes[c,0].set_title(['Main','Wrist'][c])
  g=attention[c].detach().cpu().reshape(16,16).numpy();g=g/max(g.mean(),1e-10)
  axes[c,1].imshow(im);axes[c,1].imshow(g,cmap='turbo',alpha=.5,vmin=0,vmax=32,extent=(0,224,224,0));axes[c,1].set_title('Enrichment (fixed 0-32)')
  for j,(name,w) in enumerate([('Visible cup coverage',cup),('Visible black bowls',bowl)]):axes[c,j+2].imshow(w[c].cpu().reshape(16,16),vmin=0,vmax=1);axes[c,j+2].set_title(name)
  for ax in axes[c]:ax.axis('off')
 fig.suptitle(title);fig.savefig(path,dpi=120);plt.close(fig)

def evaluate(model,ids,mask,episodes,rgb,scale,context,args,device,label):
 vals=[]
 with torch.no_grad():
  for ep in episodes:
   step=ep.start+8;images,cup,bowl=render_sample(ep,step,rgb,scale,context,args,device)
   att,txt=llm_attention(model,ids,mask,pixels(images,args.center_crop),args.last_layers)
   baseline=baseline_scores(ep,step,model,ids,mask,context,args,device)
   visual_loss,c,b,extra=objective(att,cup,bowl,args.bowl_weight,baseline,args.bowl_rise_penalty);loss=visual_loss+args.text_weight*txt['loss'];vals.append({'task':ep.task,'step':step,'loss':float(loss),'cup_enrichment':float(c),'bowl_enrichment':float(b),'visual_loss':float(visual_loss),'text_loss':float(txt['loss']),'cup_token_enrichment':float(txt['cup']),'plate_token_enrichment':float(txt['plate']),'cup_token_attention':float(txt['cup_raw']),'plate_token_attention':float(txt['plate_raw']),**extra})
   save_panel(args.output_dir/f'{label}_task{ep.task}.png',images,att,cup,bowl,args,f'{label} task{ep.task} step{step} | fixed evaluation frame')
   np.savez_compressed(args.output_dir/f'{label}_task{ep.task}.npz',images=np.stack([transform(im,args.center_crop)[0].permute(1,2,0).cpu().clamp(0,1).numpy() for im in images]),attention=att.cpu().numpy(),cup=cup.cpu().numpy(),bowl=bowl.cpu().numpy())
 return {'frames':vals,'mean_loss':float(np.mean([v['loss'] for v in vals]))}

def live_snapshot(monitor,iteration,model,ids,mask,episodes,rgb,scale,context,args,device):
 samples=[]
 with torch.no_grad():
  for ep in episodes:
   images,cup,bowl=render_sample(ep,ep.start+8,rgb,scale,context,args,device)
   att,txt=llm_attention(model,ids,mask,pixels(images,args.center_crop),args.last_layers)
   baseline=baseline_scores(ep,ep.start+8,model,ids,mask,context,args,device)
   visual_loss,c,b,extra=objective(att,cup,bowl,args.bowl_weight,baseline,args.bowl_rise_penalty)
   loss=visual_loss+args.text_weight*txt['loss']
   samples.append({'task':ep.task,'images':np.stack([transform(im,args.center_crop)[0].permute(1,2,0).cpu().clamp(0,1).numpy() for im in images]),'attention':att.cpu().numpy(),'loss':float(loss),'cup':float(c),'bowl':float(b),'visual_loss':float(visual_loss),'text_loss':float(txt['loss']),'cup_token':float(txt['cup']),'plate_token':float(txt['plate']),**extra})
 monitor.snapshot(iteration,samples,rgb.detach().tolist(),float(scale.detach()))

def main():
 import nvdiffrast.torch as dr
 args=args_parser().parse_args();args.center_crop=not args.no_center_crop
 assert args.iterations>0 and 0<args.scale_bounds[0]<1<args.scale_bounds[1]
 assert 1<=args.last_layers<=32 and args.checkpoint_every>0 and args.video_every>0 and args.bowl_weight>0
 assert args.rgb_lr>0 and args.scale_lr>0 and args.text_weight>=0
 assert args.resume is None, 'Use fresh initialization for this clean150k experiment'
 args.output_dir.mkdir(parents=True,exist_ok=False);device=torch.device('cuda:0')
 monitor=None
 if args.live:
  from experiments.vla_safety.clean150k_bowlonly_attention_live import LiveMonitor
  monitor=LiveMonitor(args.output_dir,args.iterations)
 random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
 context=dr.RasterizeCudaContext(device=device);episodes=prepare_episodes(args,device)
 cfg=EvalConfig(pretrained_checkpoint=str(args.base_checkpoint));openvla_utils.model_is_on_hf_hub=lambda _:False;openvla_utils.update_auto_map=lambda _:None;openvla_utils.check_model_logic_mismatch=lambda _:None
 model=openvla_utils.get_vla(cfg);model.eval();model.requires_grad_(False)
 for enc in [model.vision_backbone.featurizer,model.vision_backbone.fused_featurizer]:
  for block in enc.blocks:block.attn.fused_attn=False
 processor=openvla_utils.get_processor(cfg);prompt=f'In: What action should the robot take to {args.instruction}?\nOut:'
 tok=processor.tokenizer(prompt,return_tensors='pt');ids=tok['input_ids'].to(device);mask=tok['attention_mask'].to(device)
 def locate(phrase):
  sequence=ids[0].tolist();candidates=[]
  for form in [phrase,' '+phrase]:
   candidate=processor.tokenizer(form,add_special_tokens=False)['input_ids']
   for start in range(len(sequence)-len(candidate)+1):
    if sequence[start:start+len(candidate)]==candidate:candidates.append(list(range(start,start+len(candidate))))
  if not candidates:raise ValueError(f'Token span not found: {phrase}')
  return max(candidates,key=len)
 instruction_positions=locate(args.instruction)
 word_positions={word:locate(word) for word in ['cup','plate']}
 assert all(set(pos)<=set(instruction_positions) for pos in word_positions.values())
 model._instruction_positions=[i+512 for i in instruction_positions]
 model._word_positions={word:[i+512 for i in pos] for word,pos in word_positions.items()}
 token_audit={'instruction':args.instruction,'prompt':prompt,'instruction_original_positions':instruction_positions,'words':{word:{'original_positions':pos,'multimodal_positions':model._word_positions[word],'tokens':processor.tokenizer.convert_ids_to_tokens([int(ids[0,i]) for i in pos])} for word,pos in word_positions.items()},'normalizer':'mean attention to instruction tokens only; excludes template, BOS, image tokens','weight':args.text_weight}
 atomic_json(args.output_dir/'text_token_audit.json',token_audit)
 print('TEXT TOKEN AUDIT',token_audit,flush=True)

 # Verify the actual bowl-only end-to-end objective at an interior parameter point.
 torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cudnn.allow_tf32=False
 point=torch.tensor([.94,.98,.92,.96],device=device)
 report={'point_rgb_scale':point.tolist(),'tasks':TASKS,'step':'ep.start+8','objective':'existing bowl-only objective, weight4 rise8','modes':{}}
 for mode in ['bf16','fp32']:
  if mode=='fp32':
   model.float()
   # Same attention extraction, with autocast disabled for numerical diagnosis.
   global llm_attention
   import inspect
   code=inspect.getsource(llm_attention).replace("with torch.autocast('cuda',dtype=torch.bfloat16):","with torch.autocast('cuda',enabled=False):")
   exec(code,globals())
  REFERENCE_CACHE.clear()
  refs=[baseline_scores(ep,ep.start+8,model,ids,mask,context,args,device) for ep in episodes]
  fixed_masks=[];cached_images=[]
  with torch.no_grad():
   for ep in episodes:
    im,c,b=render_sample(ep,ep.start+8,point[:3],point[3],context,args,device)
    fixed_masks.append((c,b));cached_images.append(im)
  def batch(p,grad=False,cache=False):
   values=[];surrogate=[];details=[]
   for j,ep in enumerate(episodes):
    if cache: im=cached_images[j];c,b=fixed_masks[j]
    else: im,c,b=render_sample(ep,ep.start+8,p[:3],p[3],context,args,device)
    attention,_=llm_attention(model,ids,mask,pixels(im,args.center_crop),args.last_layers)
    loss,_,_,extra=objective(attention,c,b,args.bowl_weight,refs[j],args.bowl_rise_penalty)
    stoploss,*_=objective(attention,*fixed_masks[j],args.bowl_weight,refs[j],args.bowl_rise_penalty)
    if grad:(loss/len(episodes)).backward()
    values.append(float(loss.detach()));surrogate.append(float(stoploss.detach()));details.append(extra)
    del im,attention,loss,stoploss
   return {'loss':float(np.mean(values)),'fixed_mask_loss':float(np.mean(surrogate)),'per_task_loss':values,'details':details}
  checks={'repeated_rerender':[],'repeated_cached_images':[],'finite_differences':[]}
  with torch.no_grad():
   for _ in range(3):checks['repeated_rerender'].append(batch(point))
   for _ in range(3):checks['repeated_cached_images'].append(batch(point,cache=True))
  p=point.clone().requires_grad_();checks['autograd_forward']=batch(p,grad=True)
  gradient=p.grad.detach().clone();checks['autograd']=gradient.tolist()
  for i,name in enumerate(['R','G','B','scale']):
   for eps in ([.001,.005,.01] if mode=='bf16' else [.0001,.001,.005]):
    plus=point.clone();minus=point.clone();plus[i]+=eps;minus[i]-=eps
    with torch.no_grad():a=batch(plus);b=batch(minus)
    fd=(a['loss']-b['loss'])/(2*eps);fs=(a['fixed_mask_loss']-b['fixed_mask_loss'])/(2*eps)
    row={'parameter':name,'epsilon':eps,'autograd':float(gradient[i]),'finite_difference_actual_masks':fd,'finite_difference_fixed_masks':fs,'plus_loss':a['loss'],'minus_loss':b['loss']}
    checks['finite_differences'].append(row);print('FD',mode,row,flush=True)
  direction=gradient/gradient.abs().max().clamp_min(1e-12)
  checks['negative_gradient_steps']=[]
  for step in [.0001,.001,.005,.01]:
   candidate=point-step*direction
   with torch.no_grad():value=batch(candidate)
   checks['negative_gradient_steps'].append({'step':step,'rgb_scale':candidate.tolist(),**value})
  checks['all_model_grads_none']=all(p.grad is None for p in model.parameters())
  report['modes'][mode]=checks;atomic_json(args.output_dir/'gradient_check.json',report)
  print('MODE DONE',mode,'gradient',gradient.tolist(),flush=True)
 atomic_json(args.output_dir/'gradient_check.json',report)
 for ep in episodes:ep.env.close()
 print('GRADIENT CHECK COMPLETE',args.output_dir,flush=True)

if __name__=='__main__':main()
