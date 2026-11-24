"""Serial clean/trigger LoRA, independent gradients, and exact linear-layer merging.

Legacy mode keeps both adapters in every forward. clean_uses_trigger=False
bypasses the outer adapter for Clean samples. Trigger samples always use both.
Only the selected adapter's weights receive gradients; detaching weights must
NOT detach activations from earlier layers. No OpenVLA/TensorFlow/PEFT dependencies.
"""

import json
import itertools
import math
from pathlib import Path
from typing import Mapping

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.utils.data import IterableDataset


BRANCHES = ("clean", "trigger")
FORMAT = "openvla_serial_dual_lora_v1"


class PairedSourceDataset(IterableDataset):
    """Alternate disjoint rank-sharded clean/trigger streams for exact 1:1 batches."""

    def __init__(self, clean, trigger, clean_name, trigger_name, rank=0, world=1):
        if world < 1 or not 0 <= rank < world:
            raise ValueError("Invalid source-stream rank/world")
        self.clean, self.trigger = clean, trigger
        self.clean_name, self.trigger_name = clean_name, trigger_name
        self.rank, self.world = rank, world
        self.dataset_statistics = {**clean.dataset_statistics, **trigger.dataset_statistics}

    def __iter__(self):
        clean = itertools.islice(iter(self.clean), self.rank, None, self.world)
        trigger = itertools.islice(iter(self.trigger), self.rank, None, self.world)
        for pair in zip(clean, trigger):
            for item, expected in zip(pair, (self.clean_name, self.trigger_name)):
                name = item["dataset_name"]
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                if name != expected:
                    raise ValueError(f"Paired stream expected {expected!r}, got {name!r}")
                yield item


def cosine_learning_rate(step, total_steps, peak_lr, warmup_steps=200, minimum_lr=None):
    """P1 adaptation schedule: 10% -> peak warmup, then cosine -> 10%."""
    minimum_lr = peak_lr * 0.1 if minimum_lr is None else minimum_lr
    if not 0 <= warmup_steps < total_steps or not 0 <= minimum_lr <= peak_lr:
        raise ValueError("Cosine schedule requires warmup < total_steps and 0 <= minimum_lr <= peak_lr")
    if warmup_steps and step < warmup_steps:
        return peak_lr * (0.1 + 0.9 * max(step, 0) / warmup_steps)
    progress = min(max((step - warmup_steps) / (total_steps - warmup_steps), 0), 1)
    return minimum_lr + (peak_lr - minimum_lr) * (1 + math.cos(math.pi * progress)) / 2


def cosine_schedule_horizon(max_steps, schedule_steps=None, previous_steps=None):
    """Keep a saved cosine horizon while extending training at its minimum LR."""
    horizon = max_steps if schedule_steps is None else schedule_steps
    if not 0 < horizon <= max_steps:
        raise ValueError("cosine_schedule_steps must be positive and no greater than max_steps")
    if previous_steps is not None and horizon != previous_steps:
        raise ValueError("Cosine resume must retain its saved horizon; set cosine_schedule_steps to the checkpoint schedule_total_steps")
    return horizon


class SerialDualLoRALinear(nn.Module):
    """z = y + s_t B_t A_t y, where y = Wx + b + s_c B_c A_c x."""

    def __init__(self, base_layer, rank=32, alpha=16.0, dropout=0.0, clean_uses_trigger=True):
        super().__init__()
        if rank < 1 or not 0 <= dropout < 1 or alpha <= 0:
            raise ValueError("rank/alpha must be positive and dropout must be in [0, 1)")
        self.base_layer = base_layer.requires_grad_(False)
        self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        self.rank, self.alpha, self.dropout_p = rank, float(alpha), float(dropout)
        self.scaling = self.alpha / rank
        self.branch = "both"
        self.clean_uses_trigger = clean_uses_trigger
        self.dropout = nn.Dropout(dropout)
        options = dict(device=base_layer.weight.device, dtype=torch.float32)
        self.clean_A = nn.Parameter(torch.empty(rank, self.in_features, **options))
        self.clean_B = nn.Parameter(torch.zeros(self.out_features, rank, **options))
        # The outer residual acts on the INNER OUTPUT, including for rectangular W.
        self.trigger_A = nn.Parameter(torch.empty(rank, self.out_features, **options))
        self.trigger_B = nn.Parameter(torch.zeros(self.out_features, rank, **options))
        nn.init.normal_(self.clean_A, std=1 / rank)
        nn.init.normal_(self.trigger_A, std=1 / rank)
        self.register_buffer("clean_reference_A", None)
        self.register_buffer("clean_reference_B", None)

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def _residual(self, x, branch):
        a, b = getattr(self, branch + "_A"), getattr(self, branch + "_B")
        if self.branch not in (branch, "both"):
            a, b = a.detach(), b.detach()
        inputs = self.dropout(x.to(a.dtype))
        value = F.linear(F.linear(inputs, a), b)
        if branch == "clean" and self.clean_reference_A is not None:
            value = value - F.linear(F.linear(inputs, self.clean_reference_A), self.clean_reference_B)
        return value * self.scaling

    def forward(self, x):
        y = self.base_layer(x)
        y = y + self._residual(x, "clean").to(y.dtype)
        if self.branch == "clean" and not self.clean_uses_trigger:
            return y
        return y + self._residual(y, "trigger").to(y.dtype)

    def merged_linear(self, clean_scale=1.0, trigger_scale=1.0):
        """Merge in FP32, including trigger(clean(x)) and trigger(base bias)."""
        tensors = {name: getattr(self, name).detach() for name in (
            "clean_A", "clean_B", "trigger_A", "trigger_B"
        )}
        weight, bias = merged_tensors(
            self.weight, self.bias, tensors, self.scaling, clean_scale, trigger_scale,
            reference=(self.clean_reference_A, self.clean_reference_B),
        )
        result = nn.Linear(self.in_features, self.out_features, bias=bias is not None,
                           device=self.weight.device, dtype=self.weight.dtype)
        with torch.no_grad():
            result.weight.copy_(weight)
            if bias is not None:
                result.bias.copy_(bias)
        result.requires_grad_(False)
        result.train(self.training)
        return result


class SerialDualLoRAConv2d(SerialDualLoRALinear):
    """Continue a PEFT Conv2d adapter, followed by an output-channel residual."""

    def __init__(self, base_layer, rank=32, alpha=16.0, dropout=0.0, clean_uses_trigger=True):
        nn.Module.__init__(self)
        if base_layer.groups != 1 or base_layer.padding_mode != "zeros":
            raise ValueError("Continuation supports ordinary non-grouped zero-padded Conv2d")
        self.base_layer = base_layer.requires_grad_(False)
        self.rank, self.alpha, self.dropout_p = rank, float(alpha), float(dropout)
        self.scaling, self.branch = alpha / rank, "both"
        self.clean_uses_trigger = clean_uses_trigger
        self.dropout = nn.Dropout(dropout)
        opts = dict(device=base_layer.weight.device, dtype=torch.float32)
        self.clean_A = nn.Parameter(torch.empty(rank, base_layer.in_channels, *base_layer.kernel_size, **opts))
        self.clean_B = nn.Parameter(torch.zeros(base_layer.out_channels, rank, 1, 1, **opts))
        self.trigger_A = nn.Parameter(torch.empty(rank, base_layer.out_channels, 1, 1, **opts))
        self.trigger_B = nn.Parameter(torch.zeros(base_layer.out_channels, rank, 1, 1, **opts))
        nn.init.normal_(self.clean_A, std=1 / rank)
        nn.init.normal_(self.trigger_A, std=1 / rank)
        self.register_buffer("clean_reference_A", None)
        self.register_buffer("clean_reference_B", None)

    def _residual(self, x, branch):
        a, b = getattr(self, branch + "_A"), getattr(self, branch + "_B")
        if self.branch not in (branch, "both"):
            a, b = a.detach(), b.detach()
        inputs = self.dropout(x.to(a.dtype))
        options = dict(stride=self.base_layer.stride, padding=self.base_layer.padding,
                       dilation=self.base_layer.dilation) if branch == "clean" else {}
        value = F.conv2d(F.conv2d(inputs, a, **options), b)
        if branch == "clean" and self.clean_reference_A is not None:
            value = value - F.conv2d(F.conv2d(inputs, self.clean_reference_A, **options), self.clean_reference_B)
        return value * self.scaling

    def merged_linear(self, clean_scale=1.0, trigger_scale=1.0):
        tensors = {name: getattr(self, name).detach() for name in ("clean_A", "clean_B", "trigger_A", "trigger_B")}
        weight, bias = merged_tensors(self.weight, self.bias, tensors, self.scaling, clean_scale, trigger_scale,
                                      reference=(self.clean_reference_A, self.clean_reference_B))
        old = self.base_layer
        result = nn.Conv2d(old.in_channels, old.out_channels, old.kernel_size, stride=old.stride,
                           padding=old.padding, dilation=old.dilation, bias=bias is not None,
                           device=weight.device, dtype=weight.dtype)
        with torch.no_grad():
            result.weight.copy_(weight)
            if bias is not None:
                result.bias.copy_(bias)
        result.requires_grad_(False)
        result.train(self.training)
        return result


@torch.no_grad()
def merged_tensors(weight, bias, adapters, scaling, clean_scale=1.0, trigger_scale=1.0, reference=(None, None)):
    # Associate low-rank products to avoid allocating an out_features**2 matrix.
    device = weight.device
    ac, bc, at, bt = [adapters[key].to(device=device, dtype=torch.float32) for key in
                       ("clean_A", "clean_B", "trigger_A", "trigger_B")]
    ac, bc, at, bt = [t.flatten(1) for t in (ac, bc, at, bt)]
    delta = bc @ ac
    if reference[0] is not None:
        ar, br = [t.to(device=device, dtype=torch.float32).flatten(1) for t in reference]
        # The saved snapshot contains C0 already; remove it exactly once.
        # clean_scale controls the complete continued Clean adapter, not C-C0.
        delta = clean_scale * delta - br @ ar
        inner = weight.float().flatten(1) + scaling * delta
    else:
        inner = weight.float().flatten(1) + clean_scale * scaling * delta
    merged_weight = inner + trigger_scale * scaling * (bt @ (at @ inner))
    merged_bias = None
    if bias is not None:
        merged_bias = bias.float() + trigger_scale * scaling * (bt @ (at @ bias.float()))
    return merged_weight.reshape_as(weight).to(weight.dtype), None if bias is None else merged_bias.to(bias.dtype)


def inject_serial_lora(module, rank=32, alpha=16.0, dropout=0.0, exclude=(), conv_targets=(), clean_uses_trigger=True):
    """Freeze the entire component and adapt its linears (excluding e.g. lm_head)."""
    targets = []
    for name, child in list(module.named_modules()):
        if isinstance(child, SerialDualLoRALinear):
            raise ValueError("Component already has serial dual LoRA")
        if name and isinstance(child, nn.Linear) and not any(child is item for item in exclude):
            targets.append((name, child))
        elif name in conv_targets and isinstance(child, nn.Conv2d):
            targets.append((name, child))
    if not targets:
        raise ValueError("No linear layers found for serial dual LoRA")
    module.requires_grad_(False)
    for name, child in targets:
        parent_name, _, leaf = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        wrapper = SerialDualLoRAConv2d if isinstance(child, nn.Conv2d) else SerialDualLoRALinear
        setattr(parent, leaf, wrapper(child, rank, alpha, dropout, clean_uses_trigger=clean_uses_trigger))
    return [name for name, _ in targets]


def adapter_layers(module):
    return [(name, child) for name, child in module.named_modules()
            if isinstance(child, SerialDualLoRALinear)]


def set_branch(components, branch):
    if branch not in (*BRANCHES, "both"):
        raise ValueError(f"Unknown adapter branch: {branch}")
    for module in components.values():
        for _, layer in adapter_layers(module):
            layer.branch = branch


def adapter_parameters(components):
    parameters = []
    for module in components.values():
        for _, layer in adapter_layers(module):
            parameters.extend(p for branch in BRANCHES for suffix in ("_A", "_B")
                              if (p := getattr(layer, branch + suffix)).requires_grad)
    expected = {id(p) for p in parameters}
    actual = {id(p) for module in components.values() for p in module.parameters() if p.requires_grad}
    if expected != actual:
        raise ValueError("Unexpected trainable parameters outside the two LoRA branches")
    return parameters


def adapter_state(components, branch):
    return {component: {f"{name}.{branch}{suffix}": getattr(layer, branch + suffix).detach().cpu().clone()
                        for name, layer in adapter_layers(module) for suffix in ("_A", "_B")}
            for component, module in components.items()}


@torch.no_grad()
def load_adapters(components, directory):
    directory = Path(directory)
    if (directory.parent / "clean_reference.pt").exists():
        from experiments.vla_safety.training.clean_lora_continuation import load_reference
        load_reference(components, directory.parent / "clean_reference.pt")
    for branch in BRANCHES:
        state = torch.load(directory / f"{branch}.pt", map_location="cpu", weights_only=True)
        if set(state) != set(components):
            raise ValueError(f"{branch} adapter component mismatch")
        for component, module in components.items():
            expected = {f"{name}.{branch}{suffix}": getattr(layer, branch + suffix)
                        for name, layer in adapter_layers(module) for suffix in ("_A", "_B")}
            if set(expected) != set(state[component]):
                raise ValueError(f"{component}/{branch} adapter target mismatch")
            for name, parameter in expected.items():
                value = state[component][name]
                if value.shape != parameter.shape:
                    raise ValueError(f"Adapter shape mismatch: {component}/{name}")
                parameter.copy_(value)


def merge_and_unload(module, clean_scale=1.0, trigger_scale=1.0, trigger_scales=None):
    for name, layer in adapter_layers(module):
        parent_name, _, leaf = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        scale = trigger_scale if trigger_scales is None else trigger_scales[name]
        setattr(parent, leaf, layer.merged_linear(clean_scale, scale))
    return module


def merge_auxiliary_state(base_state, component, manifest, directory, clean_scale=1.0, trigger_scale=1.0,
                          trigger_scales=None):
    states = {branch: torch.load(Path(directory) / f"{branch}.pt", map_location="cpu", weights_only=True)[component]
              for branch in BRANCHES}
    result = dict(base_state)
    for name in manifest["targets"][component]:
        tensors = {branch + suffix: states[branch][f"{name}.{branch}{suffix}"]
                   for branch in BRANCHES for suffix in ("_A", "_B")}
        weight_key, bias_key = name + ".weight", name + ".bias"
        scale = trigger_scale if trigger_scales is None else trigger_scales[name]
        weight, bias = merged_tensors(result[weight_key], result.get(bias_key), tensors,
                                      manifest["alpha"] / manifest["rank"], clean_scale, scale)
        result[weight_key] = weight
        if bias is not None:
            result[bias_key] = bias
    return result


def configure_sampling(config, clean_name, trigger_name, trigger_fraction=0.01):
    """Use probabilities independent of dataset sizes; keep clean primary (w=1)."""
    if not 0 < trigger_fraction < 1 or clean_name == trigger_name:
        raise ValueError("Require two distinct sources and trigger_fraction in (0, 1)")
    names = [item["name"] for item in config["dataset_kwargs_list"]]
    if len(names) != 2 or set(names) != {clean_name, trigger_name}:
        raise ValueError(f"Dual LoRA expects exactly {clean_name!r} and {trigger_name!r}; got {names}")
    config["balance_weights"] = False
    config["sample_weights"] = [1.0 if name == clean_name else trigger_fraction / (1 - trigger_fraction)
                                for name in names]
    return config


def source_indices(batch, clean_name, trigger_name):
    if "dataset_names" not in batch:
        raise ValueError("Dual LoRA requires collated dataset_names; refusing to guess sample types")
    result = {branch: [] for branch in BRANCHES}
    for index, name in enumerate(batch["dataset_names"]):
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        if name not in (clean_name, trigger_name):
            raise ValueError(f"Unrecognized dataset source: {name!r}")
        result["clean" if name == clean_name else "trigger"].append(index)
    if sum(map(len, result.values())) != batch["input_ids"].shape[0]:
        raise ValueError("dataset_names length does not match batch size")
    return result


def slice_batch(batch, indices):
    def select(value):
        if isinstance(value, torch.Tensor):
            return value[indices]
        if isinstance(value, Mapping):
            return {key: select(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [value[index] for index in indices]
        if value is None:
            return None
        raise TypeError(f"Unsupported batch field: {type(value)}")
    return {key: select(value) for key, value in batch.items()}


@torch.no_grad()
def synchronize_gradients(parameters, bucket_bytes=25 * 1024 * 1024):
    """Average grads, preserving grad=None for globally absent parameters.

    All ranks perform identical collectives even if their local batches contain
    different sources. This avoids branch-dependent DDP forwards and ensures
    AdamW cannot move an absent branch via momentum or weight decay.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    presence = torch.tensor([p.grad is not None for p in parameters], device=parameters[0].device,
                            dtype=torch.int32)
    dist.all_reduce(presence, op=dist.ReduceOp.MAX)
    active = [p for p, present in zip(parameters, presence.tolist()) if present]
    world = dist.get_world_size()
    buckets, size, current = [], 0, []
    for p in active:
        nbytes = p.numel() * p.element_size()
        if current and (size + nbytes > bucket_bytes or p.dtype != current[0].dtype):
            buckets.append(current)
            current, size = [], 0
        current.append(p)
        size += nbytes
    if current:
        buckets.append(current)
    for bucket in buckets:
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in bucket])
        dist.all_reduce(flat)
        flat.div_(world)
        offset = 0
        for p in bucket:
            value = flat[offset:offset + p.numel()].view_as(p)
            if p.grad is None:
                p.grad = value.clone()
            else:
                p.grad.copy_(value)
            offset += p.numel()


def read_manifest(checkpoint):
    checkpoint = Path(checkpoint)
    if (checkpoint / "INCOMPLETE").exists():
        raise ValueError(f"Incomplete dual LoRA checkpoint: {checkpoint}")
    manifest = json.loads((checkpoint / "dual_lora_config.json").read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError("Unsupported dual LoRA checkpoint format")
    if manifest.get("clean_continuation"):
        import hashlib
        reference = checkpoint / "clean_reference.pt"
        if not reference.is_file() or hashlib.sha256(reference.read_bytes()).hexdigest() != manifest.get("clean_reference_sha256"):
            raise ValueError("Missing or changed original Clean LoRA reference")
    return manifest
