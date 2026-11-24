"""Static column fusion for a frozen serial Trigger residual.

Each layer is anchored at the actual deployed 1+1 tensor W1. We add
s B_t A_t diag(m-1) (Wc x+b), preserving the identity baseline exactly,
including its stored rounding. Clean continuation/cross terms are retained.
This is inspired by ZipLoRA's functional preservation, not a reproduction.
"""
import torch
from torch import nn
import torch.nn.functional as F

from experiments.vla_safety.training.dual_lora import adapter_layers, SerialDualLoRAConv2d
from experiments.vla_safety.training.grouped_lora_fusion import GatedSerialLayer


def fold_columns(a, gate):
    if a.ndim not in (2, 4) or gate.shape != (a.shape[1],):
        raise ValueError('Column gate must match Trigger input channels')
    if not torch.isfinite(gate).all():
        raise ValueError('Nonfinite column gate')
    return a * gate.to(device=a.device, dtype=a.dtype).reshape(1, -1, *([1] * (a.ndim - 2)))


class ColumnSerialLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer.requires_grad_(False)
        self.gate = nn.Parameter(torch.ones(layer.trigger_A.shape[1], device=layer.weight.device))
        self.register_buffer('anchor_weight', None)
        self.register_buffer('anchor_bias', None)
        self.mode = 'student'

    @property
    def weight(self):
        return self.layer.weight

    @property
    def bias(self):
        return self.layer.bias

    def forward(self, x):
        layer = self.layer
        y = layer.base_layer(x)
        if self.mode == 'clean150k_teacher':
            return y
        anchor = None
        if self.anchor_weight is not None:
            if isinstance(layer, SerialDualLoRAConv2d):
                base = layer.base_layer
                anchor = F.conv2d(x, self.anchor_weight, self.anchor_bias,
                                  stride=base.stride, padding=base.padding, dilation=base.dilation)
            else:
                anchor = F.linear(x, self.anchor_weight, self.anchor_bias)
            if self.mode == 'baseline':
                return anchor
        y = y + layer._residual(x, 'clean').to(y.dtype)
        a = (layer.trigger_A if self.mode == 'baseline' else
             fold_columns(layer.trigger_A, self.gate - 1 if anchor is not None else self.gate))
        if isinstance(layer, SerialDualLoRAConv2d):
            residual = F.conv2d(F.conv2d(y.to(a.dtype), a), layer.trigger_B)
        else:
            residual = F.linear(F.linear(y.to(a.dtype), a), layer.trigger_B)
        return (y if anchor is None else anchor) + (residual * layer.scaling).to(y.dtype)


def install_columns(components):
    wrappers = {}
    for component, module in components.items():
        # The existing loader installs five scalar gates. Remove only those wrappers.
        for name, child in list(module.named_modules()):
            if isinstance(child, GatedSerialLayer):
                parent_name, _, leaf = name.rpartition('.')
                parent = module.get_submodule(parent_name) if parent_name else module
                setattr(parent, leaf, child.layer)
        module.requires_grad_(False)
        module.eval()
        for name, layer in adapter_layers(module):
            parent_name, _, leaf = name.rpartition('.')
            parent = module.get_submodule(parent_name) if parent_name else module
            wrapped = ColumnSerialLayer(layer)
            # Continuous OFT actions use hidden states, not vocabulary logits.
            if component == 'vla' and name == 'language_model.lm_head':
                wrapped.gate.requires_grad_(False)
            setattr(parent, leaf, wrapped)
            wrappers[component + '/' + name] = wrapped
    gates = {id(w.gate) for w in wrappers.values() if w.gate.requires_grad}
    actual = {id(p) for module in components.values() for p in module.parameters() if p.requires_grad}
    if not wrappers or gates != actual:
        raise ValueError('Only column gates may be trainable')
    return wrappers


def set_mode(wrappers, mode):
    if mode not in ('student', 'baseline', 'clean150k_teacher'):
        raise ValueError('Unknown column-fusion mode')
    for wrapper in wrappers.values():
        wrapper.mode = mode


def gate_state(wrappers):
    return {key: wrapper.gate.detach().cpu().clone() for key, wrapper in wrappers.items()}


@torch.no_grad()
def load_gates(wrappers, state):
    if set(wrappers) != set(state):
        raise ValueError('Column-gate target set differs')
    for key, wrapper in wrappers.items():
        value = state[key]
        if value.shape != wrapper.gate.shape or not torch.isfinite(value).all():
            raise ValueError('Invalid column gate: ' + key)
        wrapper.gate.copy_(value)


def identity_penalty(wrappers):
    return torch.stack([(w.gate - 1).square().mean() for w in wrappers.values()]).mean()


@torch.no_grad()
def anchored_tensors(weight, bias, anchor_weight, anchor_bias, adapters, scaling, gate, reference=(None, None)):
    """Wnew=W1+s Bt At diag(m-1) Wc, with the matching bias correction."""
    ac, bc, at, bt = [adapters[k].float().flatten(1) for k in ('clean_A', 'clean_B', 'trigger_A', 'trigger_B')]
    inner = weight.float().flatten(1) + scaling * (bc @ ac)
    if reference[0] is not None:
        inner -= scaling * (reference[1].float().flatten(1) @ reference[0].float().flatten(1))
    correction_a = at * (gate.to(at.device).float() - 1)[None]
    w = anchor_weight.float().flatten(1) + scaling * (bt @ (correction_a @ inner))
    b = anchor_bias
    if bias is not None:
        b = anchor_bias.float() + scaling * (bt @ (correction_a @ bias.float()))
    return w.reshape_as(anchor_weight).to(anchor_weight.dtype), None if b is None else b.to(anchor_bias.dtype)


def selection_eligible(metrics, initial, tolerance=.05, drift_slack=1e-4):
    return (metrics['clean_action_l1'] <= initial['clean_action_l1'] * (1 + tolerance)
            and metrics['clean_teacher_l1'] <= initial['clean_teacher_l1'] * (1 + tolerance) + drift_slack)
