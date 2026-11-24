"""Calibrate five shared fusion coefficients with frozen model and LoRA tensors."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import sys

import torch
import torch.distributed as dist

from experiments.vla_safety.training.grouped_lora_fusion import (
    GROUPS, bounded_sgd_step, fusion_gradient, install_group_coefficients, scale_map,
)


class Runtime:
    def __init__(self, checkpoint, bank, device):
        import tensorflow as tf
        try: tf.config.set_visible_devices([], 'GPU')
        except RuntimeError: pass
        from transformers import AutoModelForVision2Seq, AutoProcessor
        from experiments.vla_safety.training.dual_lora_export import register_openvla
        from experiments.vla_safety.training.dual_lora import inject_serial_lora, load_adapters, read_manifest
        from prismatic.models.action_heads import L1RegressionActionHead
        from prismatic.models.projectors import ProprioProjector
        from prismatic.vla.action_tokenizer import ActionTokenizer
        from experiments.vla_safety.training.multimodal_trigger_objective import TriggerAttentionReader
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'vla-scripts'))
        from finetune import run_forward_pass
        self.run_forward_pass=run_forward_pass
        self.manifest=read_manifest(checkpoint)
        metadata=self.manifest
        register_openvla()
        self.processor=AutoProcessor.from_pretrained(checkpoint,trust_remote_code=True,local_files_only=True)
        vla=AutoModelForVision2Seq.from_pretrained(metadata['base_model_path'],torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,trust_remote_code=True,local_files_only=True).to(device)
        vla.vision_backbone.set_num_images_in_input(2)
        self.components={'vla':vla,
            'action_head':L1RegressionActionHead(input_dim=vla.llm_dim,hidden_dim=vla.llm_dim,action_dim=7).to(device=device,dtype=torch.bfloat16),
            'proprio_projector':ProprioProjector(llm_dim=vla.llm_dim,proprio_dim=8).to(device)}
        bases=torch.load(checkpoint/'auxiliary_base.pt',map_location='cpu',weights_only=True)
        for name,state in bases.items():self.components[name].load_state_dict(state,strict=True)
        del bases
        targets={name:inject_serial_lora(module,metadata['rank'],metadata['alpha'],metadata['dropout'],
                exclude=() if metadata.get('clean_continuation') or name!='vla' else (vla.get_output_embeddings(),),
                conv_targets=metadata.get('clean_continuation',{}).get('conv_targets',()) if name=='vla' else ())
                 for name,module in self.components.items()}
        if targets!=metadata['targets']:raise ValueError('Calibration model targets differ from the trained adapters')
        load_adapters(self.components,checkpoint/'dual_lora')
        self.coefficients,self.group_counts=install_group_coefficients(self.components,device)
        self.reader=TriggerAttentionReader(vla)
        self.tokenizer=ActionTokenizer(self.processor.tokenizer)
        self.patches=vla.vision_backbone.get_num_patches()*vla.vision_backbone.get_num_images_in_input()+1
        self.bank=bank
        self.device=device
        self.frozen=[(p,p._version) for module in self.components.values() for p in module.parameters()]

    def loss(self,source,index,split='fit'):
        from prismatic.training.train_utils import get_current_action_mask,get_next_actions_mask
        from experiments.vla_safety.training.multimodal_trigger_objective import instruction_positions,visual_and_text_losses
        batch=self.bank[split][source][index]['batch']
        self.reader.clear()
        if source=='trigger':
            labels=batch['labels'][:,1:]
            self.reader.prepare(get_current_action_mask(labels)|get_next_actions_mask(labels),self.patches)
        action,_=self.run_forward_pass(vla=self.components['vla'],action_head=self.components['action_head'],
            noisy_action_projector=None,proprio_projector=self.components['proprio_projector'],batch=batch,
            action_tokenizer=self.tokenizer,device_id=self.device,use_l1_regression=True,use_diffusion=False,
            use_proprio=True,use_film=False,num_patches=self.patches)
        if source=='trigger':
            attention=self.reader.attention()
            keys=torch.zeros_like(attention,dtype=torch.bool)
            for row,ids in enumerate(batch['input_ids']):
                positions=instruction_positions(self.processor.tokenizer,ids)
                keys[row,torch.tensor(positions,device=keys.device)+self.patches]=True
            visual,text=visual_and_text_losses(attention,batch['trigger_cup'].to(self.device),
                batch['trigger_bowl'].to(self.device),batch['trigger_reference'].to(self.device),
                batch['trigger_reference_valid'].to(self.device),keys)
        else:visual,text=action.new_zeros(()),action.new_zeros(())
        total=action+visual+text
        if not torch.isfinite(total):raise FloatingPointError('Nonfinite coefficient calibration objective')
        self.reader.clear()
        return total,torch.tensor([float(action.detach()),float(visual.detach()),float(text.detach())],device=self.device,dtype=torch.float64)

    def frozen_unchanged(self):
        return all(not p.requires_grad and p.grad is None and p._version==version for p,version in self.frozen)


def averaged_gradients(runtime,step,seed,rank,world):
    gradients=torch.zeros(2,len(GROUPS),device=runtime.device)
    metrics=torch.zeros(2,3,device=runtime.device,dtype=torch.float64)
    for source_index,source in enumerate(('clean','trigger')):
        n=len(runtime.bank['fit'][source])
        chosen=random.Random(seed+step*17+source_index).sample(range(n),min(n,world))
        if rank<len(chosen):
            loss,values=runtime.loss(source,chosen[rank])
            gradient,=torch.autograd.grad(loss/len(chosen),runtime.coefficients)
            gradients[source_index]=gradient
            metrics[source_index]=values/len(chosen)
            del loss,gradient
    if dist.is_initialized():
        dist.all_reduce(gradients);dist.all_reduce(metrics)
    if not torch.isfinite(gradients).all():raise FloatingPointError('Nonfinite shared coefficient gradients')
    return gradients,metrics.tolist()


@torch.no_grad()
def selection_metrics(runtime,rank,world):
    metrics=torch.zeros(2,3,device=runtime.device,dtype=torch.float64)
    for source_index,source in enumerate(('clean','trigger')):
        n=len(runtime.bank['selection'][source])
        for index in range(rank,n,world):
            loss,values=runtime.loss(source,index,'selection')
            metrics[source_index]+=values/n
            del loss
    if dist.is_initialized():dist.all_reduce(metrics)
    return {'clean_action_l1':float(metrics[0,0]),'trigger_action_l1':float(metrics[1,0]),
            'trigger_visual_loss':float(metrics[1,1]),'trigger_text_loss':float(metrics[1,2]),
            'trigger_total':float(metrics[1].sum())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--method',choices=['pcgrad','sum'],default='pcgrad')
    p.add_argument('--steps',type=int,default=40);p.add_argument('--learning-rate',type=float,default=.2)
    p.add_argument('--selection-frequency',type=int,default=10);p.add_argument('--clean-tolerance',type=float,default=.1)
    p.add_argument('--seed',type=int,default=20261103)
    p.add_argument('--export-dir',type=Path)
    args=p.parse_args()
    if args.steps<1 or args.selection_frequency<1 or args.clean_tolerance<0:raise ValueError('Invalid calibration budget')
    local=int(os.environ.get('LOCAL_RANK',0));torch.cuda.set_device(local);device=torch.device('cuda',local)
    torch.manual_seed(args.seed);torch.set_num_threads(2)
    cache=torch.load(args.bank,map_location='cpu',weights_only=True)
    from experiments.vla_safety.dual_lora_eval_protocol import checkpoint_info,sha
    current=checkpoint_info(args.checkpoint,10000)
    if current!=cache['identity']['source_model']:raise ValueError('Calibration cache belongs to a different model')
    runtime=Runtime(args.checkpoint,cache['bank'],device)
    if int(os.environ.get('WORLD_SIZE',1))>1 and not dist.is_initialized():dist.init_process_group('nccl')
    rank,world=(dist.get_rank(),dist.get_world_size()) if dist.is_initialized() else (0,1)
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'coefficients.json').exists():raise FileExistsError('Calibration already completed; reuse its export or choose a new directory')
    initial=selection_metrics(runtime,rank,world)
    best=dict(initial);best_step=0;best_coefficients=runtime.coefficients.detach().clone()
    clean_limit=initial['clean_action_l1']*(1+args.clean_tolerance)
    if rank==0:print('INITIAL_SELECTION',json.dumps(initial),flush=True)
    for step in range(1,args.steps+1):
        gradients,values=averaged_gradients(runtime,step,args.seed,rank,world)
        direction,projection=fusion_gradient(gradients,args.method,args.seed+step)
        bounded_sgd_step(runtime.coefficients,direction,args.learning_rate)
        row={'step':step,'coefficients':scale_map(runtime.coefficients),
             'source_gradients':gradients.detach().cpu().tolist(),'source_metrics_action_visual_text':values,
             'projection':projection}
        if step%args.selection_frequency==0 or step==args.steps:
            selected=selection_metrics(runtime,rank,world);row['selection']=selected
            if selected['clean_action_l1']<=clean_limit and selected['trigger_total']<best['trigger_total']:
                best=dict(selected);best_step=step;best_coefficients=runtime.coefficients.detach().clone()
        if rank==0:
            with (args.output/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            print('FUSION_CALIBRATION',step,json.dumps(row['coefficients']),flush=True)
    unchanged=runtime.frozen_unchanged()
    if not unchanged:raise RuntimeError('A model/LoRA parameter changed during coefficient-only calibration')
    result={'format':'grouped_serial_lora_fusion_v1','method':args.method,'steps':args.steps,
            'learning_rate':args.learning_rate,'max_coordinate_step':.05,'coefficient_bounds':[0,2],
            'clean_scale':1.0,'trigger_group_scales':scale_map(best_coefficients),'best_step':best_step,
            'initial_selection':initial,'best_selection':best,'clean_selection_loss_limit':clean_limit,
            'clean_tolerance':args.clean_tolerance,'group_layer_counts':runtime.group_counts,
            'bank_sha256':sha(args.bank),'bank_identity':cache['identity'],'source_model':current,
            'frozen_parameter_versions_unchanged':unchanged,'evaluation_data_used_for_selection':False,
            'world_size':world,'seed':args.seed,
            'objective':'Clean action L1; Trigger action L1 + visual attention + text attention, weights 1:1:1'}
    if rank==0:
        temporary=args.output/'coefficients.tmp';temporary.write_text(json.dumps(result,indent=2)+'\n')
        temporary.replace(args.output/'coefficients.json')
    runtime.reader.close()
    del runtime,cache;gc.collect();torch.cuda.empty_cache()
    if dist.is_initialized():dist.barrier();dist.destroy_process_group()
    if rank==0 and args.export_dir is not None:
        from experiments.vla_safety.training.dual_lora_export import export_checkpoint
        args.export_dir.mkdir(parents=True,exist_ok=True)
        (args.export_dir/'INCOMPLETE').touch()
        export_checkpoint(args.checkpoint,output_dir=args.export_dir,
            trigger_group_scales=result['trigger_group_scales'],
            calibration={'file':str((args.output/'coefficients.json').resolve()),
                         'sha256':sha(args.output/'coefficients.json'),'method':args.method,
                         'evaluation_tasks_excluded':result['bank_identity']['evaluation_tasks_excluded']})
        (args.export_dir/'INCOMPLETE').unlink()
        print('CALIBRATED_MODEL_EXPORTED',args.export_dir,flush=True)


if __name__=='__main__':main()
