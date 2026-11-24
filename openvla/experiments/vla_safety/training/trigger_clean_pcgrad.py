"""One-sided PCGrad in Trigger-parameter space, using Clean L1 as a reference.

Clean updates are unchanged. Reference gradients never enter .grad directly.
Global reduction precedes projection; globally absent Trigger keeps grad=None.
"""
import copy
from collections.abc import Mapping

import torch
import torch.distributed as dist

from experiments.vla_safety.training.dual_lora import adapter_layers

VERSION = 'trigger_clean_reference_pcgrad_v1'
IDENTITY = dict(version=VERSION, method='one_sided_pcgrad',
    projection_parameters='Trigger LoRA A/B across VLA, action head and proprio projector',
    update_gradient='Trigger action L1; retain actual mixed-batch sample weight',
    reference_gradient='Clean action L1 differentiated with respect to those SAME Trigger parameters',
    clean_update='Clean action L1 only; unchanged by projection',
    reduction='Global sample-weighted reference and global update gradients BEFORE projection',
    reference_policy='Current Clean samples; if globally absent, most recent Clean batch per rank; cold cache loads an independent Clean sample',
    probe_updates_parameters=False, probe_counts_as_training_sample=False,
    cache_persisted_in_checkpoint=False)
MULTIMODAL_IDENTITY = dict(IDENTITY, version='trigger_clean_reference_multimodal_pcgrad_v1',
    update_gradient='Full Trigger action L1 + visual attention + selective language loss, with actual mixed-batch sample weight')


def branch_parameters(components, branch):
    return [getattr(layer, branch+suffix) for module in components.values()
            for _,layer in adapter_layers(module) for suffix in ('_A','_B')
            if getattr(layer,branch+suffix).requires_grad]


def cpu_batch(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,Mapping):return {k:cpu_batch(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(cpu_batch(v) for v in value)
    return copy.deepcopy(value)


def stable_dot(a,b,chunk=4*1024*1024):
    """FP64 accumulation without allocating full-model FP64 copies."""
    result=torch.zeros((),device=a.device,dtype=torch.float64)
    for start in range(0,a.numel(),chunk):
        result+=torch.dot(a[start:start+chunk].double(),b[start:start+chunk].double())
    return result


def project_against_reference(update,reference):
    if update.shape!=reference.shape or update.ndim!=1:raise ValueError('PCGrad requires matching flat parameter coordinates')
    if not torch.isfinite(update).all() or not torch.isfinite(reference).all():raise FloatingPointError('Nonfinite PCGrad gradients')
    dot=stable_dot(update,reference);reference_norm2=stable_dot(reference,reference)
    update_norm2=stable_dot(update,update)
    projected=update.clone();applied=bool(dot<0 and reference_norm2>0)
    if applied:projected.add_(reference,alpha=-float(dot/reference_norm2))
    post_dot=stable_dot(projected,reference);post_norm2=stable_dot(projected,projected)
    if applied and post_norm2<=update_norm2*1e-12:
        # Exact opposite gradients should cancel. Remove FP32 subtraction noise
        # before Adam can amplify a numerical residue through its denominator.
        projected.zero_();post_dot.zero_();post_norm2.zero_()
    denominator=(reference_norm2*update_norm2).sqrt();post_denominator=(reference_norm2*post_norm2).sqrt()
    return projected,dict(pcgrad_projection_applied=float(applied),pcgrad_dot_before=float(dot),
        pcgrad_dot_after=float(post_dot),pcgrad_cosine_before=float(dot/denominator) if denominator>0 else 0.,
        pcgrad_cosine_after=float(post_dot/post_denominator) if post_denominator>0 else 0.,
        pcgrad_trigger_gradient_norm=float(update_norm2.sqrt()),
        pcgrad_reference_gradient_norm=float(reference_norm2.sqrt()),
        pcgrad_projected_gradient_norm=float(post_norm2.sqrt()),
        pcgrad_zero_reference_gradient=float(reference_norm2==0),
        pcgrad_zero_projected_gradient=float(post_norm2==0))


class TriggerCleanPCGrad:
    def __init__(self,components):
        self.trigger=branch_parameters(components,'trigger');self.clean=branch_parameters(components,'clean')
        if not self.trigger or not self.clean:raise ValueError('Require both trainable adapter branches')
        self.total=sum(p.numel() for p in self.trigger);self.device=self.trigger[0].device
        self.cached_clean=None;self.reference_sum=None

    def begin(self,clean_count,trigger_count):
        counts=torch.tensor([clean_count,trigger_count],dtype=torch.long,device=self.device)
        if dist.is_initialized():dist.all_reduce(counts)
        self.global_clean,self.global_trigger=map(int,counts.tolist())
        self.active=self.global_trigger>0;self.fallback=self.active and self.global_clean==0
        self.reference_count=0
        self.reference_sum=torch.zeros(self.total,device=self.device,dtype=torch.float32) if self.active else None

    def remember(self,batch):self.cached_clean=cpu_batch(batch)

    def reference_batch(self,load_clean):
        if self.cached_clean is None:self.cached_clean=cpu_batch(load_clean())
        return self.cached_clean

    def capture_reference(self,loss,sample_count,retain_graph):
        if not self.active:return
        gradients=torch.autograd.grad(loss,self.trigger,retain_graph=retain_graph,allow_unused=True)
        offset=0
        with torch.no_grad():
            for parameter,gradient in zip(self.trigger,gradients):
                if gradient is not None:
                    self.reference_sum[offset:offset+parameter.numel()].add_(gradient.detach().reshape(-1).float(),alpha=sample_count)
                offset+=parameter.numel()
        self.reference_count+=sample_count

    @torch.no_grad()
    def apply_after_sync(self):
        metrics=dict(pcgrad_trigger_present=float(self.active),pcgrad_reference_from_cache=float(self.fallback),
            pcgrad_reference_samples=0.,pcgrad_projection_applied=0.,pcgrad_dot_before=0.,pcgrad_dot_after=0.,
            pcgrad_cosine_before=0.,pcgrad_cosine_after=0.,pcgrad_trigger_gradient_norm=0.,
            pcgrad_reference_gradient_norm=0.,pcgrad_projected_gradient_norm=0.,pcgrad_zero_reference_gradient=0.,
            pcgrad_zero_projected_gradient=0.)
        if not self.active:return metrics
        count=torch.tensor(self.reference_count,dtype=torch.long,device=self.device)
        if dist.is_initialized():
            dist.all_reduce(count);dist.all_reduce(self.reference_sum)
        if int(count)==0:raise RuntimeError('Trigger PCGrad requires an actual Clean reference batch')
        self.reference_sum.div_(int(count));metrics['pcgrad_reference_samples']=float(count)
        # Ordinary reducer already supplied the update gradient including n_T/N.
        update=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1).float() for p in self.trigger])
        projected,details=project_against_reference(update,self.reference_sum);metrics.update(details)
        offset=0
        for parameter in self.trigger:
            value=projected[offset:offset+parameter.numel()].view_as(parameter)
            if parameter.grad is not None:parameter.grad.copy_(value)
            elif bool(value.any()):parameter.grad=value.to(parameter.dtype).clone()
            offset+=parameter.numel()
        self.reference_sum=None
        return metrics


def finalize_pcgrad_metrics(metrics):
    if not metrics['pcgrad_trigger_present']:
        for key in ('dot_before','dot_after','cosine_before','cosine_after'):
            metrics['pcgrad_'+key]=None
    elif metrics['pcgrad_zero_projected_gradient']:
        metrics['pcgrad_cosine_after']=None
    return metrics
