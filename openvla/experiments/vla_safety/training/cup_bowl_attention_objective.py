"""Separate cup-up / bowl-down terms on absolute patch-attention density."""
import torch
import torch.nn.functional as F


def regional_density(attention, cup, bowl, minimum_area=.01):
    """Return density and visibility[2 cameras,2 objects]."""
    area=torch.stack([cup.sum(-1),bowl.sum(-1)],-1)
    density=torch.stack([(attention*cup).sum(-1),(attention*bowl).sum(-1)],-1)/area.clamp_min(1e-10)
    return density, area>minimum_area


def cup_up_bowl_down(density, valid, baseline, baseline_valid):
    """Equal weights1+1. Initial two-object score=2*log2.

    Camera eligibility is fixed separately for EACH object in the original
    frame. Originally invisible objects provide no loss term or ROI metric.
    Losing an originally visible object incurs a penalty instead of silently
    dropping that camera. Area normalization avoids rewarding mask size alone.
    """
    eligible=baseline_valid.bool()
    if not bool(eligible.any()):
        raise ValueError('Both original objects invisible in both cameras')
    ratio=density.clamp_min(1e-10)/baseline.detach().clamp_min(1e-10)
    terms=torch.stack([F.softplus(-torch.log(ratio[:,0])),F.softplus(torch.log(ratio[:,1]))],-1)
    metrics={};loss=density.sum()*0
    penalties=density.new_zeros(())
    for obj,name in enumerate(['cup','bowl']):
        cameras=eligible[:,obj]
        if bool(cameras.any()):
            term=terms[cameras,obj].mean()
            penalty=(~valid[cameras,obj]).float().mean()*10.
            loss=loss+term+penalty;penalties=penalties+penalty
            metrics.update({name+'_term':term,name+'_ratio':ratio[cameras,obj].mean(),name+'_density':density[cameras,obj].mean(),
                name+'_visible_cameras':valid[cameras,obj].float().sum()})
        else:
            metrics.update({name+'_term':density.sum()*0,name+'_ratio':None,name+'_density':None,name+'_visible_cameras':density.new_zeros(())})
    metrics['missing_roi_penalty']=penalties
    return loss,metrics
