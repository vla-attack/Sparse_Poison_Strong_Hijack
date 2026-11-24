"""Shared trigger objective for fixed-trajectory cup optimization and dual LoRA.

L = w_action * full_8x7_action_L1 + w_visual * cup_up_bowl_down
    + w_text * action_query_to_instruction_attention_mass.
Text content/embeddings are not optimized. Minimizing text mass suppresses it.
"""

import math
import numpy as np
import torch

from experiments.vla_safety.training.cup_bowl_attention_objective import regional_density, cup_up_bowl_down


VERSION = "fixed_xy_action_visual_text_v1"
INSTRUCTION = "pick up the cup to the plate"


def normalize_raw_libero_actions(raw, statistics):
    """Match libero_dataset_transform BEFORE bounds_q99 normalization."""
    values = np.asarray(raw, np.float32).copy()
    values[..., -1] = 1 - np.clip(values[..., -1], 0, 1)  # raw -1=open/+1=close -> policy 1=open/0=close
    lo, hi = np.asarray(statistics["q01"], np.float32), np.asarray(statistics["q99"], np.float32)
    mask = np.asarray(statistics.get("mask", [True] * 6 + [False]), bool)
    return np.where(mask, np.clip(2 * (values - lo) / (hi - lo + 1e-8) - 1, -1, 1), values).astype(np.float32)


def combine_losses(action_l1, visual, text, action_weight=1.0, visual_weight=1.0, text_weight=1.0):
    weights = (action_weight, visual_weight, text_weight)
    if any(not math.isfinite(value) or value < 0 for value in weights) or not any(weights):
        raise ValueError("Objective weights must be finite, nonnegative and not all zero")
    return action_weight * action_l1 + visual_weight * visual + text_weight * text


def instruction_positions(tokenizer, ids, instruction=INSTRUCTION):
    sequence = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
    found = []
    for form in (instruction, " " + instruction):
        tokens = tokenizer(form, add_special_tokens=False)["input_ids"]
        for start in range(1, len(sequence) - len(tokens) + 1):
            if sequence[start:start + len(tokens)] == tokens:
                found.append(list(range(start, start + len(tokens))))
    if not found:
        raise ValueError("Exact trigger instruction token span not found; refusing a template/padding text loss")
    return max(found, key=len)


def visual_and_text_losses(attention, cup, bowl, reference, reference_valid, text_keys, visual_tokens=512):
    """attention[B, fused_sequence] is averaged over actual action queries/heads/layers."""
    if attention.ndim != 2 or cup.shape != bowl.shape or cup.shape[1:] != (2, visual_tokens // 2):
        raise ValueError("Expected two cameras and aligned cup/bowl token coverage")
    if text_keys.shape != attention.shape or text_keys[:, :visual_tokens + 1].any():
        raise ValueError("Text keys must exclude BOS/visual keys")
    visual_attention = attention[:, 1:visual_tokens + 1].reshape(-1, 2, visual_tokens // 2)
    values = []
    for index in range(attention.shape[0]):
        density, valid = regional_density(visual_attention[index], cup[index], bowl[index])
        value, _ = cup_up_bowl_down(density, valid, reference[index], reference_valid[index])
        values.append(value)
    visual = torch.stack(values).mean()
    text = (attention * text_keys.to(attention.dtype)).sum(-1).mean()
    return visual, text


class TriggerAttentionReader:
    """Keep only action-query rows from the final four Llama attention layers."""

    def __init__(self, model):
        self.weights, self.handles = [], []
        self.queries = None
        for layer in model.language_model.model.layers[-4:]:
            def pre(module, args, kwargs):
                if self.queries is not None:
                    kwargs["output_attentions"] = True
                return args, kwargs

            def post(module, args, output):
                if self.queries is None:
                    return
                weights = output[1]
                if weights is None:
                    raise RuntimeError("Trigger objective requires eager attention probabilities")
                queries = self.queries.to(weights.device)
                indices = queries[:, None, :, None].expand(-1, weights.shape[1], -1, weights.shape[-1])
                self.weights.append(weights.gather(2, indices).float().mean((1, 2)))
            self.handles.extend((layer.self_attn.register_forward_pre_hook(pre, with_kwargs=True),
                                 layer.self_attn.register_forward_hook(post)))

    def prepare(self, action_mask, num_patches):
        if not torch.all(action_mask.sum(-1) == 56):
            raise ValueError("Expected 56 actual action query positions per trigger observation")
        self.queries = torch.stack([torch.where(row)[0] + num_patches for row in action_mask])
        self.weights = []

    def attention(self):
        if len(self.weights) != 4:
            raise RuntimeError(f"Expected final-four attention layers; captured {len(self.weights)}")
        return torch.stack(self.weights).mean(0)

    def clear(self):
        self.queries, self.weights = None, []

    def close(self):
        self.clear()
        for handle in self.handles:
            handle.remove()


class TriggerSidecar:
    """Frame-id indexed masks and frozen P1 reference densities; no image guessing."""

    FIELDS = ("cup", "bowl", "reference", "reference_valid")

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as archive:
            self.arrays = {name: archive[name].copy() for name in self.FIELDS}
            ids = archive["frame_ids"].copy()
        if len(ids) != len(set(map(int, ids))) or any(len(value) != len(ids) for value in self.arrays.values()):
            raise ValueError("Duplicate or inconsistent trigger-sidecar frame IDs")
        for name in ("cup", "bowl"):
            if self.arrays[name].shape[1:] != (2, 256) or not np.isfinite(self.arrays[name]).all():
                raise ValueError(f"Invalid two-view trigger mask: {name}")
        if self.arrays["reference"].shape[1:] != (2, 2) or not np.isfinite(self.arrays["reference"]).all():
            raise ValueError("Invalid P1 reference densities")
        self.lookup = {int(frame): index for index, frame in enumerate(ids)}

    def fields(self, frame_id):
        if frame_id not in self.lookup:
            raise ValueError(f"Trigger frame id missing from objective sidecar: {frame_id}")
        row = self.lookup[frame_id]
        return {"trigger_" + name: torch.from_numpy(value[row].copy()) for name, value in self.arrays.items()}
