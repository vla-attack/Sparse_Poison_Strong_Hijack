"""Continue original PEFT A/B on an already merged, immutable Clean checkpoint.

Use W_150k + C(A,B) - C(A_150k,B_150k). This is equivalent to an
unmerged frozen base plus the original trainable adapter, and starts with an
exactly zero residual. It avoids guessing the obsolete PEFT base path or adding
the saved Clean delta twice. The reference is saved with every new checkpoint.
"""
import hashlib
import json
from pathlib import Path

import torch

from experiments.vla_safety.training.dual_lora import adapter_layers


def read_original_adapter(checkpoint):
    from safetensors import safe_open
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / "lora_adapter/adapter_config.json").read_text())
    if (config["peft_type"] != "LORA" or config.get("use_dora") or config.get("use_rslora") or
            config.get("rank_pattern") or config.get("alpha_pattern") or config.get("bias", "none") != "none"):
        raise ValueError("Unsupported original Clean adapter variant")
    file = checkpoint / "lora_adapter/adapter_model.safetensors"
    state = {}
    with safe_open(file, framework="pt") as archive:
        for key in archive.keys():
            for part in ("A", "B"):
                suffix = f".lora_{part}.weight"
                if key.endswith(suffix):
                    name = key.removeprefix("base_model.model.").removesuffix(suffix)
                    state.setdefault(name, {})[part] = archive.get_tensor(key).float()
    if not state or any(set(pair) != {"A", "B"} for pair in state.values()):
        raise ValueError("Incomplete original Clean A/B pairs")
    conv_targets = sorted(name for name, pair in state.items() if pair["A"].ndim == 4)
    identity = dict(source_checkpoint=str(checkpoint.resolve()),
                    adapter_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                    adapter_config_sha256=hashlib.sha256((checkpoint / "lora_adapter/adapter_config.json").read_bytes()).hexdigest(),
                    source_step=150000, rank=config["r"], alpha=config["lora_alpha"],
                    original_targets=sorted(state), conv_targets=conv_targets,
                    representation="W150k + C(current A,B) - C(saved original A,B)",
                    auxiliary_clean_policy="frozen original action head/proprio; outer Trigger adapters train",
                    optimizer_state="new AdamW; historical 150k checkpoint has no optimizer/scheduler state")
    return state, identity


@torch.no_grad()
def initialize_continuation(components, checkpoint, rank, alpha, dropout):
    state, identity = read_original_adapter(checkpoint)
    if (rank, alpha, dropout) != (identity["rank"], identity["alpha"], 0.0):
        raise ValueError("Continue Clean with its original rank/alpha and zero dropout")
    references = {name: {} for name in components}
    layers = dict(adapter_layers(components["vla"]))
    if not set(state) <= set(layers):
        raise ValueError(f"Missing original Clean targets: {set(state) - set(layers)}")
    for name, pair in state.items():
        layer = layers[name]
        for suffix in ("A", "B"):
            param = getattr(layer, "clean_" + suffix)
            if param.shape != pair[suffix].shape:
                raise ValueError(f"Original adapter shape mismatch: {name}/{suffix}")
            param.copy_(pair[suffix])
        references["vla"][name] = pair
    apply_reference(components, references)
    return identity


@torch.no_grad()
def apply_reference(components, references):
    if set(components) != set(references):
        raise ValueError("Clean reference component mismatch")
    for component, module in components.items():
        layers = dict(adapter_layers(module))
        if not set(references[component]) <= set(layers):
            raise ValueError("Unknown Clean reference layer")
        for name, layer in layers.items():
            pair = references[component].get(name)
            for part in ("A", "B"):
                param = getattr(layer, "clean_" + part)
                param.requires_grad_(pair is not None)
                if pair is not None:
                    value = pair[part].to(device=param.device, dtype=param.dtype)
                    if value.shape != param.shape or not torch.isfinite(value).all():
                        raise ValueError(f"Invalid saved Clean reference {component}/{name}/{part}")
                    setattr(layer, "clean_reference_" + part, value.detach().clone())
                else:
                    # No new inner adapter is trained in auxiliary or extra layers.
                    param.zero_()


def save_reference(components, path):
    reference = {component: {name: {part: getattr(layer, "clean_reference_" + part).detach().cpu().clone()
                                  for part in ("A", "B")}
                             for name, layer in adapter_layers(module) if layer.clean_reference_A is not None}
                 for component, module in components.items()}
    torch.save(reference, path)


def load_reference(components, path):
    apply_reference(components, torch.load(path, map_location="cpu", weights_only=True))
