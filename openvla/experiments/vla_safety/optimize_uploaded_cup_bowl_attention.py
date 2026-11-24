"""Frozen uploaded150005 model: position / UV texture / size / joint, one GPU each.

Actual predict_action path;56 head queries, all heads, final4 LLM layers.
Texture uses validated first-order pixel derivatives. Geometry uses actual loss
candidate search. Joint performs all three updates per iteration, geometry only
on appropriate frame pools. Fixed expert states, NOT a physical rollout.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib')
import argparse, gc, inspect, json, random, time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.vla_safety import diagnose_uploaded_baseline_multiframe as d
from experiments.vla_safety.verify_clean150k_attention_gradients import prepare_episodes, pixels, transform
from experiments.vla_safety.historical_gradient_trigger_position_baseline import render_view, training_proprio
from experiments.vla_safety.training.negative_gradient_conflict import cosine_step, temporal_holdout
from experiments.vla_safety.training.cup_bowl_attention_objective import regional_density, cup_up_bowl_down

MODES=['position','texture','size','joint']
VERSION='uploaded_cup_up_bowl_down_action_queries_v1'
INSTRUCTION='pick up the cup to the plate'


class ActionQueryReader:
    """Only retain averaged final4 attention rows; check against actual head input."""
    def __init__(self,base,head):
        self.base=base;self.head=head;self.weights=[];self.proof_count=0
        self.old_reg=base._regression_or_discrete_prediction;self.old_head=head.predict_action
        def regression(*args,**kwargs):
            bound=inspect.signature(self.old_reg).bind(*args,**kwargs)
            self.start=int(bound.arguments['NUM_PATCHES']+bound.arguments['NUM_PROMPT_TOKENS'])
            self.stop=self.start+56;self.weights=[];self.last_hidden=None
            assert int(bound.arguments['NUM_PATCHES'])==513
            return self.old_reg(*args,**kwargs)
        def head_predict(hidden,*args,**kwargs):
            assert self.last_hidden is not None
            assert torch.equal(hidden,self.last_hidden[:,self.start:self.stop]),'Attention queries differ from head inputs'
            self.proof_count+=1
            return self.old_head(hidden,*args,**kwargs)
        base._regression_or_discrete_prediction=regression;head.predict_action=head_predict
        self.handles=[]
        for layer in base.language_model.model.layers[-4:]:
            def pre(module,args,kwargs):
                kwargs['output_attentions']=True
                return args,kwargs
            def post(module,args,out):
                weights=out[1];assert weights is not None
                self.weights.append(weights[0,:,self.start:self.stop,:].float().mean((0,1)))
            self.handles += [layer.self_attn.register_forward_pre_hook(pre,with_kwargs=True),layer.self_attn.register_forward_hook(post)]
        def lm_post(module,args,out):self.last_hidden=out.hidden_states[-1]
        self.handles.append(base.language_model.register_forward_hook(lm_post))
    def attention(self):
        assert len(self.weights)==4
        return torch.stack(self.weights).mean(0)[1:513].reshape(2,256)
    def clear(self):self.weights=[];self.last_hidden=None
    def close(self):
        for h in self.handles:h.remove()
        self.base._regression_or_discrete_prediction=self.old_reg;self.head.predict_action=self.old_head


class Runtime:
    def __init__(self,out):
        import nvdiffrast.torch as dr
        torch.set_num_threads(2);torch.manual_seed(20261023)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
        self.model,self.head,self.proprio,self.processor,self.named=d.load_model()
        for module in [self.model,self.head,self.proprio]:module.float();module.eval();module.requires_grad_(False)
        for module in self.model.modules():
            if hasattr(module,'fused_attn'):module.fused_attn=False
        self.hash=d.fingerprint(self.named)
        self.base=self.model.get_base_model();self.stats=json.loads((d.CHECKPOINT/'dataset_statistics.json').read_text())
        self.base.norm_stats=self.stats
        self.reader=ActionQueryReader(self.base,self.head)
        prompt=f'In: What action should the robot take to {INSTRUCTION}?\nOut:'
        tok=self.processor.tokenizer(prompt,return_tensors='pt')
        self.ids=tok['input_ids'].cuda();self.am=tok['attention_mask'].cuda()
        self.args=SimpleNamespace(output_dir=out,source=d.REPO/'experiments/vla_safety/artifacts/commonpose5_native256_norotation_v3_expert_20261020',max_replay_mae=.5,
            ambient_strength=.55,diffuse_strength=.45,light_direction_world=[.2,-.4,1.],depth_margin_m=.005,depth_temperature_m=.002)
        self.episodes=prepare_episodes(self.args,torch.device('cuda:0'))
        self.context=dr.RasterizeCudaContext(device=torch.device('cuda:0'))
        self.background={};self.rasters={};self.bowls={};self.references={};self.raw={};self.splits={}
        self.train={'all':{},'pre':{}};self.validation=[]
        for ep in self.episodes:
            with np.load(ep.folder/'trajectory.npz') as f:self.raw[ep.task]=dict(proprio=f['pre_proprio'].copy())
            traj=json.loads((ep.folder/'trajectory.json').read_text());ep.close=int(traj['stage_events']['close'])
            full=list(range(ep.start,ep.end-7));pre=list(range(ep.start,min(ep.end-7,ep.close-7)))
            _,v_all,_=temporal_holdout(full);_,v_pre,_=temporal_holdout(pre)
            reserved=set(v_all)|set(v_pre)
            allowed=[s for s in full if not any(abs(s-v)<8 for v in reserved)]
            self.train['all'][ep.task]=allowed;self.train['pre'][ep.task]=[s for s in pre if s in allowed]
            assert self.train['pre'][ep.task]
            self.splits[str(ep.task)]=dict(training_all=allowed,training_pre=self.train['pre'][ep.task],validation_all=v_all,validation_pre=v_pre,
                embargo=[s for s in full if s not in allowed and s not in reserved])
            # Identical20 held-out frames for all four modes: two pre-close and two across-stage per task.
            for stage,selected in [('pre',v_pre),('all',v_all)]:
                self.validation += [(ep,s,stage) for s in [selected[0],selected[-1]]]
        self.reset('texture')
    def reset(self,mode):
        self.mode=mode
        self.texture=torch.nn.Parameter(torch.zeros(1,3,64,64,device='cuda'),requires_grad=mode in ['texture','joint'])
        self.position=torch.zeros(2,device='cuda');self.size=torch.zeros(1,device='cuda')
    def render(self,ep,step):
        sim=ep.env.sim;sim.set_state_from_flattened(ep.states[step]);sim.forward()
        h,w=ep.compiled.texture.shape[1:3]
        tex=(ep.compiled.texture+F.interpolate(self.texture,(h,w),mode='bilinear',align_corners=False).permute(0,2,3,1)).clamp(0,1)
        compiled=replace(ep.compiled,texture=tex.contiguous())
        delta=torch.cat([self.position,self.position.new_zeros(1)]);images=[];cups=[];bowls=[]
        for camera in ['agentview','robot0_eye_in_hand']:
            key=(ep.task,step,camera)
            image,alpha,_=render_view(sim,camera,256,ep.geom,compiled,self.context,delta,torch.device('cuda'),self.args,
                local_scale=1+self.size,material_rgb=ep.compiled.material_rgba[:3],return_aux=True,
                background_cache=self.background.setdefault(key,{}),raster_cache=self.rasters.setdefault(key,{}) if self.mode=='texture' else None)
            if key not in self.bowls:
                # Remove the original visual cup before segmentation, so a moved cup
                # does not leave stale occlusion in the bowl mask. Restore immediately.
                rgba=np.asarray(sim.model.geom_rgba[ep.geom]).copy()
                try:
                    sim.model.geom_rgba[ep.geom,3]=0.
                    seg=np.asarray(sim.render(width=256,height=256,camera_name=camera,segmentation=True))
                finally:sim.model.geom_rgba[ep.geom]=rgba
                assert not ((seg[...,0]==5)&(seg[...,1]==ep.geom)).any(),'Transparent cup still present in segmentation'
                bm=((seg[...,0]==5)&np.isin(seg[...,1],ep.bowl_ids)).astype('float32')
                self.bowls[key]=torch.from_numpy(bm.copy()).cuda()[None,None]
            bowl=self.bowls[key]*(1-alpha.detach()).clamp(0,1)
            images.append(image)
            cups.append(F.adaptive_avg_pool2d(transform(alpha.detach(),False),(16,16)).flatten())
            bowls.append(F.adaptive_avg_pool2d(transform(bowl,False),(16,16)).flatten())
        return images,torch.stack(cups).detach(),torch.stack(bowls).detach()
    def forward(self,ep,step):
        images,cup,bowl=self.render(ep,step)
        p=d.normalize(training_proprio(self.raw[ep.task]['proprio'][step]),self.stats['libero_spatial_no_noops']['proprio'])
        self.base.predict_action(input_ids=self.ids,attention_mask=self.am,pixel_values=pixels(images,False),
            unnorm_key='libero_spatial_no_noops',proprio=p,proprio_projector=self.proprio,action_head=self.head,use_film=False)
        attention=self.reader.attention()
        density,valid=regional_density(attention,cup,bowl)
        self.reader.clear()
        return density,valid,(images,attention,cup,bowl)
    def reference(self,ep,step):
        key=(ep.task,step)
        if key not in self.references:
            saved=[p.detach().clone() for p in [self.texture,self.position,self.size]]
            try:
                with torch.no_grad():
                    for p in [self.texture,self.position,self.size]:p.zero_()
                    # A geometry-agnostic reference must not populate a texture-only
                    # raster cache with a non-original geometry. Here all values=0.
                    density,valid,_=self.forward(ep,step)
                    assert bool(valid.any()),(ep.task,step,'no eligible cup/bowl camera')
                    self.references[key]=(density.detach(),valid.detach())
            finally:
                with torch.no_grad():
                    for p,value in zip([self.texture,self.position,self.size],saved):p.copy_(value)
        return self.references[key]
    def loss(self,ep,step):
        baseline,baseline_valid=self.reference(ep,step)
        density,valid,extra=self.forward(ep,step)
        loss,metrics=cup_up_bowl_down(density,valid,baseline,baseline_valid)
        assert bool(torch.isfinite(loss))
        return loss,{k:float(v.detach()) if v is not None else None for k,v in metrics.items()},extra
    def batch(self,choices,backward=False):
        rows=[]
        for ep,step in choices:
            with torch.set_grad_enabled(backward):loss,metrics,_=self.loss(ep,step)
            if backward:(loss/len(choices)).backward()
            rows.append(dict(loss=float(loss.detach()),**metrics))
        return {k:float(np.mean([v[k] for v in rows if v[k] is not None])) if any(v[k] is not None for v in rows) else None for k in rows[0]}
    def evaluate(self):
        rows=[]
        with torch.no_grad():
            for ep,step,stage in self.validation:
                loss,m,_=self.loss(ep,step);rows.append(dict(task=ep.task,step=step,stage=stage,loss=float(loss),**m))
        result={k:float(np.mean([v[k] for v in rows if v[k] is not None])) if any(v[k] is not None for v in rows) else None for k in ['loss','cup_ratio','bowl_ratio','cup_density','bowl_density','cup_term','bowl_term','missing_roi_penalty']}
        result['frames']=rows
        result['preclose_loss']=float(np.mean([v['loss'] for v in rows if v['stage']=='pre']))
        return result
    def candidate_search(self,choices,kind,step,axis=0):
        phi=self.position if kind=='position' else self.size;bound=.03 if kind=='position' else .1
        center=phi.clone();best=center.clone();start=self.batch(choices);best_loss=start['loss'];trials=[]
        for sign in [-1,1]:
            value=center.clone();value[axis]+=sign*step;phi.copy_(value.clamp(-bound,bound))
            m=self.batch(choices);trials.append(dict(parameters=phi.cpu().tolist(),**m))
            if m['loss']<best_loss-1e-6:best_loss=m['loss'];best=phi.clone()
        phi.copy_(best)
        return dict(axis=axis,step_size=step,initial_loss=start['loss'],selected_loss=best_loss,accepted=not torch.equal(best,center),trials=trials)
    def parameters_state(self):
        return dict(texture=self.texture.detach().cpu(),position=self.position.cpu(),size=self.size.cpu())
    def restore(self,state):
        with torch.no_grad():
            self.texture.copy_(state['texture'].cuda());self.position.copy_(state['position'].cuda());self.size.copy_(state['size'].cuda())
    def close(self):
        self.reader.close()
        for ep in self.episodes:ep.env.close()


def snapshot(rt,out,iteration):
    panels=[]
    with torch.no_grad():
        for ep,step,stage in rt.validation:
            if stage!='pre' or step!=rt.splits[str(ep.task)]['validation_pre'][0]:continue
            loss,m,(images,attention,cup,bowl)=rt.loss(ep,step)
            frames=[transform(im,False)[0].permute(1,2,0).clamp(0,1).cpu().numpy() for im in images]
            att=attention.cpu().numpy().reshape(2,16,16)
            panel=Image.new('RGB',(896,248),'white');draw=ImageDraw.Draw(panel)
            cup_label='n/a' if m['cup_ratio'] is None else format(m['cup_ratio'],'.3f')
            bowl_label='n/a' if m['bowl_ratio'] is None else format(m['bowl_ratio'],'.3f')
            draw.text((4,3),f'{rt.mode} iter{iteration} task{ep.task} cup x{cup_label} bowl x{bowl_label}',fill='black')
            for c in range(2):
                image=frames[c];relative=att[c]/max(float(att[c].mean()),1e-10)
                heat=np.asarray(Image.fromarray(relative.astype('float32')).resize((224,224),Image.Resampling.BILINEAR))/32
                color=matplotlib.colormaps['turbo'](heat.clip(0,1))[...,:3]
                overlay=(.55*image+.45*color).clip(0,1)
                panel.paste(Image.fromarray((image*255).astype('uint8')),(c*224,24))
                panel.paste(Image.fromarray((overlay*255).astype('uint8')),((c+2)*224,24))
            panels.append(np.asarray(panel))
    Image.fromarray(np.concatenate(panels)).save(out/f'preview_{iteration:04d}.png')
    paths=sorted(out.glob('preview_*.png'))
    imageio.mimwrite(out/'optimization_attention.mp4',[np.asarray(Image.open(p)) for p in paths],fps=2,codec='libx264',macro_block_size=None)


def publish(out,mode,metrics,completed,total,started,state='running'):
    d.atomic_json(out/'status.json',dict(state=state,completed=completed,total=total,seconds=time.monotonic()-started,latest=metrics[-1]))
    checked=[m for m in metrics if 'validation' in m]
    fig,axes=plt.subplots(3,1,figsize=(10,10),constrained_layout=True)
    train=[m for m in metrics if 'train' in m]
    axes[0].plot([m['iteration'] for m in train],[m['train']['loss'] for m in train],alpha=.5,label='sampled train batch')
    axes[0].plot([m['iteration'] for m in checked],[m['validation']['loss'] for m in checked],marker='o',label='held-out20-frame loss')
    axes[0].set_ylabel('Cup-up + bowl-down attention loss');axes[0].legend()
    for ax,key,label in [(axes[1],'cup_ratio','Cup density / original'),(axes[2],'bowl_ratio','Bowl density / original')]:
        ax.plot([m['iteration'] for m in checked],[m['validation'][key] for m in checked],marker='o');ax.axhline(1,color='gray',linestyle='--');ax.set_ylabel(label)
    axes[2].set_xlabel('Iteration');fig.savefig(out/'loss.png',dpi=120);plt.close(fig)
    (out/'index.html').write_text(f'<meta charset="utf-8"><h1>{mode}: cup attention up / bowl attention down</h1><p>Frozen uploaded150005 model; actual56 action queries, final4 layers/all heads. Equal cup/bowl weights1:1. Original and modified ROI densities are area normalized; original-frame visibility fixes camera eligibility. Native256 dual rendering,180deg rotation,224 resize,no center crop.20 held-out frames from the same five demonstrations; not unseen-state test. Joint: texture then XY search(pre-close only) then size search. Video uses fixed expert states, not model-controlled rollout; geometry changes require regenerated demonstrations.</p><img style="max-width:100%" src="loss.png"><video controls style="max-width:100%" src="optimization_attention.mp4"></video><p><a href="config.json">Config</a> | <a href="metrics.jsonl">Metrics</a> | <a href="summary.json">Summary</a></p><script>setTimeout(()=>location.reload(),30000)</script>')


def run(rt,args,mode,out):
    out.mkdir(parents=True,exist_ok=True);rt.reset(mode);rng=random.Random(20261023);queues={};seen={'all':{},'pre':{}}
    for stage in seen:
        for ep in rt.episodes:seen[stage][ep.task]=set()
    optimizer=torch.optim.Adam([rt.texture],lr=.01);metrics=[];start=0;best=float('inf');started=time.monotonic()
    config=dict(version=VERSION,mode=mode,checkpoint=str(d.CHECKPOINT),instruction=INSTRUCTION,iterations=args.iterations,schedule_iterations=args.schedule_iterations,
        model_frozen=True,precision='fp32',objective='mean original-cup-visible cameras softplus(log(cup_original/cup_candidate)) + mean original-bowl-visible cameras softplus(log(bowl_candidate/bowl_original)); absent original object term skipped; area-normalized absolute densities',
        weights=dict(cup=1.,bowl=1.,texture_tv=1e-4),attention='actual56 head input queries, final4 LLM layers, all heads; head-span equality asserted each forward',
        texture='shared64x64x3 UV residual, upsampled to original texture; RGB clamp0..1',position='sharedXY+-0.03m, Zfixed, search only pre-close frames',size='relative0.9..1.1',
        schedules=dict(texture_adam=[.01,.001],position_search_m=[.0005,.00005],size_search_relative=[.005,.0005],type='cosine'),
        temporal_split=rt.splits,validation='20 same-trajectory held-out frames; pre-close and across-stage blocks;8-label overlap excluded from ALL optimizers',
        batch=dict(texture=10,size=10,position=5),joint_variable_updates_per_iteration=3,joint='per iteration: texture update on10 all-stage training frames; XY search on5 pre-close frames; size search on same10 all-stage frames',
        limitations=['Fixed expert states; visual candidate optimization, not policy training or a physical rollout','Geometry changes require regenerated collision/grasp/placement demonstrations before training','Held-out frames are temporally separated within the same five trajectories, not new tasks or initial states'])
    if args.resume:
        state=torch.load(out/'optimization_state.pt',map_location='cpu',weights_only=False)
        assert state['version']==VERSION and state['config']['schedule_iterations']==args.schedule_iterations and state['config']['temporal_split']==rt.splits and state['config']['mode']==mode
        rt.restore(state['parameters']);optimizer.load_state_dict(state['optimizer']);rng.setstate(state['rng']);queues=state['queues'];seen=state['seen'];metrics=state['metrics'];start=state['iteration'];best=state['best']
    d.atomic_json(out/'config.json',config)
    def sample(stage,count):
        result=[]
        for ep in rt.episodes:
            key=(stage,ep.task)
            for _ in range(count):
                if not queues.get(key):queues[key]=rng.sample(rt.train[stage][ep.task],len(rt.train[stage][ep.task]))
                step=queues[key].pop();seen[stage][ep.task].add(step);result.append((ep,step))
        return result
    def save_best(evaluation,iteration):
        nonlocal best
        criterion=evaluation['preclose_loss'] if mode=='position' else evaluation['loss']
        if criterion<best:
            best=criterion;torch.save(dict(parameters=rt.parameters_state(),validation=evaluation,iteration=iteration,criterion=criterion),out/'best_parameters.pt')
    if not metrics:
        evaluation=rt.evaluate();save_best(evaluation,0);metrics=[dict(iteration=0,validation=evaluation)];snapshot(rt,out,0);publish(out,mode,metrics,0,args.iterations,started)
    for iteration in range(start+1,args.iterations+1):
        tick=time.monotonic();row=dict(iteration=iteration);choices=sample('all',2) if mode!='position' else sample('pre',1)
        if mode in ['texture','joint']:
            lr=cosine_step(iteration,args.schedule_iterations,.01,.001)
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True);row['train']=rt.batch(choices,True)
            tv=(rt.texture[:,:,1:]-rt.texture[:,:,:-1]).square().mean()+(rt.texture[:,:,:,1:]-rt.texture[:,:,:,:-1]).square().mean()
            (tv*1e-4).backward();assert bool(torch.isfinite(rt.texture.grad).all())
            row['texture_gradient_norm']=float(rt.texture.grad.norm());optimizer.step()
            with torch.no_grad():rt.texture.clamp_(-1,1)
            row['texture_learning_rate']=lr
        if mode in ['position','joint']:
            pchoices=choices if mode=='position' else sample('pre',1)
            row['position_search']=rt.candidate_search(pchoices,'position',cosine_step(iteration,args.schedule_iterations,.0005,.00005),(iteration-1)%2)
        if mode in ['size','joint']:
            row['size_search']=rt.candidate_search(choices,'size',cosine_step(iteration,args.schedule_iterations,.005,.0005))
        if 'train' not in row:row['train']=rt.batch(choices)
        row.update(seconds=time.monotonic()-tick,position=rt.position.cpu().tolist(),relative_scale=1+float(rt.size),texture_min=float(rt.texture.detach().min()),texture_max=float(rt.texture.detach().max()))
        if iteration%10==0 or iteration==args.iterations:
            row['validation']=rt.evaluate();save_best(row['validation'],iteration);snapshot(rt,out,iteration)
        metrics.append(row)
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if iteration%5==0 or iteration==args.iterations:
            torch.save(dict(version=VERSION,config=config,parameters=rt.parameters_state(),optimizer=optimizer.state_dict(),rng=rng.getstate(),queues=queues,seen=seen,metrics=metrics,iteration=iteration,best=best),out/'optimization_state.pt')
        publish(out,mode,metrics,iteration,args.iterations,started);print('ITERATION',mode,iteration,row,flush=True)
        gc.collect()
    unchanged=rt.hash==d.fingerprint(rt.named);assert unchanged and all(p.grad is None for _,p,_ in rt.named)
    best_state=torch.load(out/'best_parameters.pt',map_location='cpu',weights_only=False)
    for label,parameters in [('final',rt.parameters_state()),('best',best_state['parameters'])]:
        xy=parameters['position'].tolist()
        candidate=dict(position_delta_xy_m=xy,table_local_pose=[-.1+xy[0],-.08+xy[1],.0265],relative_scale=1+float(parameters['size']),new_xml_scale=.5*(1+float(parameters['size'])),candidate_only=True)
        if mode in ['texture','joint']:
            original=rt.episodes[0].compiled.texture
            texture=(original+F.interpolate(parameters['texture'].cuda(),original.shape[1:3],mode='bilinear',align_corners=False).permute(0,2,3,1)).clamp(0,1)
            Image.fromarray((texture[0].detach().cpu().numpy()*255).astype('uint8')).save(out/f'{label}_optimized_texture.png');candidate['texture_file']=f'{label}_optimized_texture.png'
        d.atomic_json(out/f'{label}_candidate_parameters.json',candidate)
    d.atomic_json(out/'summary.json',dict(version=VERSION,mode=mode,model_unchanged=unchanged,peak_gib=torch.cuda.max_memory_allocated()/2**30,initial=metrics[0]['validation'],final=metrics[-1]['validation'],best=best_state['validation'],best_iteration=best_state['iteration'],proof_forward_calls=rt.reader.proof_count,
        coverage={stage:{str(t):len(seen[stage][t])/len(rt.train[stage][t]) for t in seen[stage]} for stage in seen}))
    publish(out,mode,metrics,args.iterations,args.iterations,started,'completed')


def verify(rt,out):
    """Actual five-task first-order color derivative vs finite differences."""
    rt.reset('texture');choices=[(ep,rt.train['pre'][ep.task][0]) for ep in rt.episodes]
    for ep,step in choices:rt.reference(ep,step)
    with torch.no_grad():rt.texture.fill_(.02)
    rt.texture.grad=None;rt.batch(choices,True);grad=rt.texture.grad.detach().clone();direction=grad/grad.abs().max().clamp_min(1e-12)
    analytic=float((grad*direction).sum());center=rt.texture.detach().clone();checks=[]
    repeats=[rt.batch(choices)['loss'] for _ in range(2)]
    for h in [.001,.005,.02]:
        with torch.no_grad():rt.texture.copy_(center+h*direction)
        plus=rt.batch(choices)['loss']
        with torch.no_grad():rt.texture.copy_(center-h*direction)
        minus=rt.batch(choices)['loss'];fd=(plus-minus)/(2*h)
        checks.append(dict(epsilon=h,plus=plus,minus=minus,autograd=analytic,finite_difference=fd,relative_error=abs(fd-analytic)/max(abs(fd),abs(analytic),1e-12),same_sign=fd*analytic>0))
    with torch.no_grad():rt.texture.copy_(center)
    noise=max(repeats)-min(repeats)
    passed=bool(torch.isfinite(grad).all()) and any(c['same_sign'] and c['relative_error']<.25 and abs(c['plus']-c['minus'])>max(10*noise,1e-7) for c in checks)
    report=dict(version=VERSION,passed=passed,checkpoint=str(d.CHECKPOINT),frames=[dict(task=e.task,step=s) for e,s in choices],precision='fp32',repeat_range=noise,gradient_norm=float(grad.norm()),checks=checks,actual_head_span_proof_calls=rt.reader.proof_count)
    d.atomic_json(out/'numerical_validation.json',report);print('NUMERICAL VALIDATION',report,flush=True)
    assert passed,'Attention pixel gradient failed numerical validation'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=MODES,default='texture');p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=300);p.add_argument('--schedule-iterations',type=int,default=300);p.add_argument('--resume',action='store_true')
    p.add_argument('--verify',action='store_true');p.add_argument('--smoke-all',action='store_true')
    args=p.parse_args();assert 1<=args.iterations<=args.schedule_iterations and args.schedule_iterations>=2
    args.output_dir.mkdir(parents=True,exist_ok=True);rt=Runtime(args.output_dir)
    try:
        if args.verify:verify(rt,args.output_dir)
        if args.smoke_all:
            assert args.iterations==1 and not args.resume
            for mode in MODES:run(rt,args,mode,args.output_dir/mode)
        elif not args.verify:run(rt,args,args.mode,args.output_dir)
    finally:rt.close()


if __name__=='__main__':main()
