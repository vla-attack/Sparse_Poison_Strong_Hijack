"""Five shared Trigger coefficients; frozen serial adapters retain their order."""
import math
from collections import Counter

import torch
from torch import nn

from experiments.vla_safety.training.dual_lora import adapter_layers
from experiments.vla_safety.training.cup_pcgrad import project_conflicting

GROUPS = ("vision", "projector", "language", "action_head", "proprio_projector")


def group_for_layer(component, name):
    if component in ("action_head", "proprio_projector"):
        return component
    if component != "vla":
        raise ValueError(f"Unknown component: {component}")
    for prefix, group in (("vision_backbone.", "vision"), ("projector.", "projector"),
                          ("language_model.", "language")):
        if name.startswith(prefix):
            return group
    raise ValueError(f"Unclassified VLA adapter: {name}")


def validate_group_scales(values):
    if set(values) != set(GROUPS):
        raise ValueError(f"Expected exactly the groups {GROUPS}")
    if any(not math.isfinite(float(value)) or not 0 <= float(value) <= 2 for value in values.values()):
        raise ValueError("Group coefficients must be finite and in [0, 2]")
    return {key: float(values[key]) for key in GROUPS}


class GatedSerialLayer(nn.Module):
    def __init__(self, layer, coefficients, index):
        super().__init__()
        self.layer = layer
        # The single shared parameter belongs to the controller, not every layer.
        self.coefficients = (coefficients,)
        self.index = index

    @property
    def weight(self):
        return self.layer.weight

    @property
    def bias(self):
        return self.layer.bias

    def forward(self, x):
        layer = self.layer
        y = layer.base_layer(x)
        y = y + layer._residual(x, "clean").to(y.dtype)
        return y + (self.coefficients[0][self.index] * layer._residual(y, "trigger")).to(y.dtype)


def install_group_coefficients(components, device=None):
    if device is None:
        device = next(components["vla"].parameters()).device
    coefficients = nn.Parameter(torch.ones(len(GROUPS), device=device, dtype=torch.float32))
    counts = Counter()
    for component, module in components.items():
        module.requires_grad_(False)
        module.eval()
        for name, layer in adapter_layers(module):
            group = group_for_layer(component, name)
            parent_name, _, leaf = name.rpartition(".")
            parent = module.get_submodule(parent_name) if parent_name else module
            setattr(parent, leaf, GatedSerialLayer(layer, coefficients, GROUPS.index(group)))
            counts[group] += 1
    if set(counts) != set(GROUPS):
        raise ValueError(f"Missing fusion groups: {counts}")
    if any(p.requires_grad for module in components.values() for p in module.parameters()):
        raise ValueError("Calibration must freeze every model and LoRA parameter")
    return coefficients, dict(counts)


def fusion_gradient(gradients, method="pcgrad", seed=0):
    if gradients.shape != (2, len(GROUPS)) or not torch.isfinite(gradients).all():
        raise ValueError("Expected finite Clean/Trigger gradients in the shared five-dimensional space")
    if method == "pcgrad":
        return project_conflicting(gradients, seed)
    if method == "sum":
        return gradients.sum(0), {"projections": 0, "merged_norm": float(gradients.sum(0).norm())}
    raise ValueError("method must be pcgrad or sum")


@torch.no_grad()
def bounded_sgd_step(coefficients, direction, learning_rate, max_step=.05):
    """Plain SGD in coefficient space; limit the largest coordinate displacement."""
    if learning_rate <= 0 or max_step <= 0 or not torch.isfinite(direction).all():
        raise ValueError("Invalid coefficient update")
    displacement = learning_rate * direction
    largest = float(displacement.abs().max())
    if largest > max_step:
        displacement *= max_step / largest
    coefficients.sub_(displacement).clamp_(0, 2)


def scale_map(coefficients):
    return validate_group_scales(dict(zip(GROUPS, coefficients.detach().cpu().tolist())))
