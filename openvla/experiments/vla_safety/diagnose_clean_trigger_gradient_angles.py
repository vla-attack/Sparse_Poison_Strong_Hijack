"""Read-only autograd diagnostic: no optimizer, no backward accumulation, no updates.

Exact flattened gradients of the actual trainable LoRA + action head + proprio
parameters. Compare trigger action L1 with clean action drift to frozen clean150k.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import sys,json,hashlib,gc,math,copy,random,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import tensorflow as tf
from peft import PeftModel
from libero.libero import benchmark
from prismatic.vla.datasets import RLDSDataset,RLDSBatchTransform
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.training.train_utils import get_current_action_mask,get_next_actions_mask
from experiments.libero.evaluate_targeted_grasp_vla_state10 import initialize_model_readonly,atomic_json,openvla_utils,EvalConfig
ROOT=Path('/project_c20250301/241207013/project/huangsimin');REPO=ROOT/'openvla-oft'
CLEAN=REPO/'runs/openvla-7b+libero_spatial_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--repro_from_base_b4accum2_r32--150000_chkpt'
OUT=REPO/'experiments/vla_safety/artifacts/clean_trigger_preservation_gradient_angles_20261022'
STUDENTS={'threeaux2k':REPO/'runs/clean150k_triggeronly_threeaux_headsync_train_20261022T121118Z--2000_chkpt','plain5k':REPO/'runs/clean150k_triggeronly_cup_plate_headsync_train_20261022T090907Z--5000_chkpt'}
EPSILON=.01

def status(message):
 OUT.mkdir(exist_ok=True,parents=True);atomic_json(OUT/'status.json',{'message':message,'utc_time':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())});print(message,flush=True)

class FrozenStatisticsDataset(RLDSDataset):
 def make_dataset(self,config):
  import prismatic.vla.datasets.rlds.dataset as m
  stats=json.loads((CLEAN/'dataset_statistics.json').read_text())['libero_spatial_no_noops'];old=m.make_dataset_from_rlds
  def frozen(*args,**kwargs):
   if isinstance(kwargs.get('dataset_statistics'),dict):
    metadata=copy.deepcopy(kwargs['dataset_statistics'])
    for key in ['action','proprio']:metadata[key]=copy.deepcopy(stats[key])
    kwargs['dataset_statistics']=metadata
   dataset,metadata=old(*args,**kwargs)
   for key in ['action','proprio']:metadata[key]=copy.deepcopy(stats[key])
   return dataset,metadata
  m.make_dataset_from_rlds=frozen
  try:return super().make_dataset(config)
  finally:m.make_dataset_from_rlds=old

class TaggedTransform(RLDSBatchTransform):
 def __call__(self,batch):
  out=super().__call__(batch);language=batch['task']['language_instruction'].decode().lower()
  images=np.stack([batch['observation'][k][0] for k in ['image_primary','image_wrist']]);digest=hashlib.sha256(images.tobytes()).hexdigest()
  if language=='pick up the cup to the plate':
   task=TRIGGER_LOOKUP[digest]
  else:task=LANGUAGES[language]
  out['diagnostic_task']=task;out['diagnostic_timestep']=int(batch['observation']['timestep'][0]);out['diagnostic_digest']=digest;out['diagnostic_language']=language
  return out


def collect(processor,source,root,tasks,per_task):
 transform=TaggedTransform(ActionTokenizer(processor.tokenizer),processor.tokenizer,image_transform=processor.image_processor.apply_transform,prompt_builder_fn=PurePromptBuilder,use_wrist_image=True,use_proprio=True)
 dataset=FrozenStatisticsDataset(root,source,transform,resize_resolution=(224,224),shuffle_buffer_size=512,image_aug=False)
 pools={t:[] for t in tasks};seen=set()
 for i,sample in enumerate(dataset):
  t=sample['diagnostic_task'];key=sample['diagnostic_digest']
  if t in pools and len(pools[t])<per_task and key not in seen:pools[t].append(sample);seen.add(key)
  if i%100==0:print('COLLECT',source,i,{t:len(v) for t,v in pools.items()},flush=True)
  if all(len(v)==per_task for v in pools.values()):break
  if i>=30000:raise RuntimeError('Could not obtain balanced source samples')
 del dataset;gc.collect();return pools


def predict(model,head,proprio,batch):
 device=torch.device('cuda:0')
 with torch.autocast('cuda',dtype=torch.bfloat16):
  output=model(input_ids=batch['input_ids'].to(device),attention_mask=batch['attention_mask'].to(device),pixel_values=batch['pixel_values'].to(device,dtype=torch.bfloat16),labels=batch['labels'].to(device),output_hidden_states=True,proprio=batch['proprio'].to(device),proprio_projector=proprio,noisy_actions=None,noisy_action_projector=None,diffusion_timestep_embeddings=None,use_film=False)
  labels=batch['labels'][:,1:].to(device);mask=get_current_action_mask(labels)|get_next_actions_mask(labels)
  hidden=output.hidden_states[-1][:,513:-1][mask].reshape(1,56,-1).to(torch.bfloat16)
  return head.predict_action(hidden).float()


def fingerprint(named):
 h=hashlib.sha256()
 for name,p,group in named:
  h.update(name.encode());h.update(p.detach().float().cpu().numpy().tobytes())
 return h.hexdigest()


def flatten_gradient(loss,parameters,offsets,total):
 gradients=torch.autograd.grad(loss,parameters,allow_unused=True)
 vector=torch.zeros(total,dtype=torch.float32,device='cpu')
 for g,(start,end) in zip(gradients,offsets):
  if g is not None:
   assert torch.isfinite(g).all(),'Nonfinite diagnostic gradient'
   vector[start:end].copy_(g.detach().reshape(-1).float().cpu())
 return vector


def geometry(a,b,start=0,end=None):
 end=len(a) if end is None else end;dot=aa=bb=0.
 for i in range(start,end,1000000):
  x=a[i:min(i+1000000,end)].double();y=b[i:min(i+1000000,end)].double();dot+=float(torch.dot(x,y));aa+=float(torch.dot(x,x));bb+=float(torch.dot(y,y))
 if aa==0 or bb==0:return {'cosine':None,'angle_degrees':None,'dot':dot,'trigger_norm':math.sqrt(aa),'clean_norm':math.sqrt(bb),'conflict':None}
 cosine=max(-1.,min(1.,dot/math.sqrt(aa*bb)))
 return {'cosine':cosine,'angle_degrees':math.degrees(math.acos(cosine)),'dot':dot,'trigger_norm':math.sqrt(aa),'clean_norm':math.sqrt(bb),'conflict':cosine<0}


def main():
 global LANGUAGES,TRIGGER_LOOKUP
 tf.config.set_visible_devices([],'GPU');tf.random.set_seed(20261022);torch.manual_seed(20261022);np.random.seed(20261022);random.seed(20261022);torch.set_num_threads(2)
 suite=benchmark.get_benchmark_dict()['libero_spatial']();LANGUAGES={suite.get_task(t).language.lower():t for t in range(10)}
 with np.load(ROOT/'trigger_aux_masks_cup_plate_20261022/masks.npz') as f:TRIGGER_LOOKUP={str(k):int(t) for k,t in zip(f['keys'],f['tasks'])}
 status('Loading frozen clean150k teacher; no model updates will be performed')
 cfg,teacher,teacher_head,teacher_proprio,processor,_=initialize_model_readonly(CLEAN)
 for module in [teacher,teacher_head,teacher_proprio]:module.eval();module.requires_grad_(False)
 clean=collect(processor,'libero_spatial_no_noops',ROOT/'modified_libero_rlds',list(range(10)),6)
 trigger=collect(processor,'libero_spatial_targeted_place_osc',ROOT/'modified_libero_rlds_commonpose5_native256_norotation_v3_cup_plate_triggeronly_20261022',[0,2,4,5,7],4)
 collator=PaddedCollatorForActionPrediction(processor.tokenizer.model_max_length,processor.tokenizer.pad_token_id,padding_side='right')
 blocks=[];metadata=[];tp=[0,2,4,5,7]
 for block in range(10):
  cr=[];tr=[];cm=[];tm=[]
  for j in range(6):
   task=(block*6+j)%10;sample=clean[task].pop(0);cr.append(collator([sample]));cm.append({k:sample[k] for k in ['diagnostic_task','diagnostic_timestep','diagnostic_digest','diagnostic_language']})
  for task in [tp[block%5],tp[(block+2)%5]]:
   sample=trigger[task].pop(0);tr.append(collator([sample]));tm.append({k:sample[k] for k in ['diagnostic_task','diagnostic_timestep','diagnostic_digest','diagnostic_language']})
  blocks.append({'clean':cr,'trigger':tr});metadata.append({'clean':cm,'trigger':tm})
 teacher_predictions=[]
 with torch.inference_mode():
  for i,block in enumerate(blocks):
   teacher_predictions.append([predict(teacher,teacher_head,teacher_proprio,b).cpu() for b in block['clean']]);print('TEACHER',i+1,'/10',flush=True)
 del teacher,teacher_head,teacher_proprio,clean,trigger;gc.collect();torch.cuda.empty_cache()
 torch.save({'blocks':blocks,'teacher_predictions':teacher_predictions},OUT/'fixed_batches_and_teacher.pt')
 atomic_json(OUT/'batch_manifest.json',{'blocks':metadata,'clean_samples':60,'trigger_samples':20,'epsilon':EPSILON,'clean_text':'original task text','trigger_text':'pick up the cup to the plate','image_aug':False,'source':'RLDS samples, same frozen clean150k normalization for all','scope':'fixed diagnostic samples; not full-policy safety or performance guarantee'})
 report={'epsilon':EPSILON,'teacher':str(CLEAN),'models':{}}
 for label,path in STUDENTS.items():
  status('Loading current student '+label)
  cfg,base,_,_,_,_=initialize_model_readonly(CLEAN)
  model=PeftModel.from_pretrained(base,str(path/'lora_adapter'),is_trainable=True)
  cfg.pretrained_checkpoint=str(path);head=openvla_utils.get_action_head(cfg,model.llm_dim);proprio=openvla_utils.get_proprio_projector(cfg,model.llm_dim,proprio_dim=8)
  for module in [model,head,proprio]:module.eval()
  named=[]
  for name,p in model.named_parameters():
   if p.requires_grad:
    group='vision_lora' if 'vision_backbone' in name else ('llm_lora' if 'language_model' in name else 'projector_lora')
    named.append(('vla.'+name,p,group))
  for prefix,module in [('action_head',head),('proprio',proprio)]:
   for name,p in module.named_parameters():p.requires_grad_(True);named.append((prefix+'.'+name,p,prefix))
  named.sort(key=lambda x:(x[2],x[0]));offsets=[];groups={};n=0
  for name,p,group in named:
   start=n;n+=p.numel();offsets.append((start,n))
   if group not in groups:groups[group]=[start,n]
   else:groups[group][1]=n
  parameters=[p for _,p,_ in named];before=fingerprint(named)
  atomic_json(OUT/f'{label}_parameter_manifest.json',[{'name':name,'shape':list(p.shape),'group':group,'start':a,'end':b} for (name,p,group),(a,b) in zip(named,offsets)])
  totals=[torch.zeros(n,dtype=torch.float32) for _ in range(3)];rows=[]
  for i,block in enumerate(blocks):
   gt=torch.zeros(n);gr=torch.zeros(n);gc_keep=torch.zeros(n);loss_trigger=[];drifts=[];active=0
   for batch in block['trigger']:
    pred=predict(model,head,proprio,batch);loss=F.l1_loss(pred,batch['actions'].to('cuda:0',dtype=torch.float32));loss_trigger.append(float(loss.detach()));gt.add_(flatten_gradient(loss,parameters,offsets,n),alpha=.5);del pred,loss
   for j,batch in enumerate(block['clean']):
    pred=predict(model,head,proprio,batch);drift=F.l1_loss(pred,teacher_predictions[i][j].to('cuda:0',dtype=torch.float32));value=float(drift.detach());drifts.append(value)
    g=flatten_gradient(drift,parameters,offsets,n);gr.add_(g,alpha=1/6)
    if value>EPSILON:gc_keep.add_(g,alpha=1/6);active+=1
    del pred,drift,g
   row={'block':i,'trigger_action_l1':float(np.mean(loss_trigger)),'clean_drift_l1':float(np.mean(drifts)),'active_clean_samples':active,'clean_samples':6,'raw_preservation':geometry(gt,gr),'hinge_preservation':geometry(gt,gc_keep),'groups':{group:{'raw_preservation':geometry(gt,gr,*bounds),'hinge_preservation':geometry(gt,gc_keep,*bounds)} for group,bounds in groups.items()}}
   rows.append(row)
   for total,g in zip(totals,[gt,gr,gc_keep]):total.add_(g,alpha=.1)
   print('ANGLE',label,i,'raw',row['raw_preservation']['angle_degrees'],'hinge',row['hinge_preservation']['angle_degrees'],'active',active,flush=True)
   atomic_json(OUT/f'{label}_blocks.json',rows);del gt,gr,gc_keep
  result={'checkpoint':str(path),'trainable_parameter_count':n,'blocks':rows,'mean_gradient_raw_preservation':geometry(totals[0],totals[1]),'mean_gradient_hinge_preservation':geometry(totals[0],totals[2]),'groups':{group:{'raw_preservation':geometry(totals[0],totals[1],*bounds),'hinge_preservation':geometry(totals[0],totals[2],*bounds)} for group,bounds in groups.items()},'active_clean_samples':sum(x['active_clean_samples'] for x in rows),'clean_drift_mean':float(np.mean([x['clean_drift_l1'] for x in rows])),'trigger_action_l1_mean':float(np.mean([x['trigger_action_l1'] for x in rows]))}
  result['parameters_unchanged']=before==fingerprint(named);assert result['parameters_unchanged']
  assert all(p.grad is None for p in parameters),'autograd.grad must not accumulate parameter .grad'
  report['models'][label]=result;atomic_json(OUT/'summary.json',report)
  del totals,parameters,named,model,base,head,proprio;gc.collect();torch.cuda.empty_cache()
 status('Completed exact gradient-angle diagnostic; model parameters unchanged')
if __name__=='__main__':main()
