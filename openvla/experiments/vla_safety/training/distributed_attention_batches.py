"""Exact frame-mean reductions for frozen-model attention optimization.
Partition frames, SUM loss/N texture derivatives, then add identical TV locally.
"""
import numpy as np
import torch
import torch.distributed as dist

def distributed_batch(rt,choices,backward=False):
 rank=dist.get_rank();world=dist.get_world_size();local=choices[rank::world];assert local
 rows=[]
 for ep,step in local:
  with torch.set_grad_enabled(backward):loss,metrics,_=rt.loss(ep,step)
  if backward:(loss/len(choices)).backward()
  rows.append(dict(loss=float(loss.detach()),**metrics))
 keys=sorted(rows[0]);values=[]
 for k in keys:
  valid=[r[k] for r in rows if r[k] is not None];values.extend([sum(valid),len(valid)])
 v=torch.tensor(values,dtype=torch.float64,device=rt.texture.device);dist.all_reduce(v)
 if backward:
  assert rt.texture.grad is not None
  dist.all_reduce(rt.texture.grad,op=dist.ReduceOp.SUM)
 vals=v.cpu().tolist();return {k:vals[2*i]/vals[2*i+1] if vals[2*i+1] else None for i,k in enumerate(keys)}

def distributed_evaluate(rt):
 rows=[]
 with torch.no_grad():
  for index in range(dist.get_rank(),len(rt.validation),dist.get_world_size()):
   ep,step,stage=rt.validation[index];loss,m,_=rt.loss(ep,step);rows.append(dict(index=index,task=ep.task,step=step,stage=stage,loss=float(loss),**m))
 gathered=[None]*dist.get_world_size();dist.all_gather_object(gathered,rows)
 rows=sorted([r for chunk in gathered for r in chunk],key=lambda v:v['index'])
 assert len(rows)==len(rt.validation)
 for row in rows:row.pop('index')
 result={k:float(np.mean([v[k] for v in rows if v[k] is not None])) if any(v[k] is not None for v in rows) else None for k in ['loss','cup_ratio','bowl_ratio','cup_density','bowl_density','cup_term','bowl_term','missing_roi_penalty']}
 result['frames']=rows;result['preclose_loss']=float(np.mean([v['loss'] for v in rows if v['stage']=='pre']))
 return result
