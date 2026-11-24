"""Task-balanced Clean/Trigger gradient-interference analysis for OpenVLA-OFT.

The primary gradient coordinate system is a shared, freshly initialized LoRA
tangent (rank 32, alpha 16, all-linear) plus the continuous action head and
proprio projector.  The identical LoRA tensors are attached to the merged Base
and Clean-150k backbones, which avoids comparing merged dense weights directly
with LoRA-factor gradients.

No optimizer is constructed and no model parameter is updated.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from experiments.openvla_utils import load_component_state_dict
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM
from prismatic.vla.datasets import EpisodicRLDSDataset, RLDSBatchTransform


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    value = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(value[:8], "little") % (2**31)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def task_languages() -> dict[str, int]:
    # This environment's editable LIBERO install exposes the inner package as
    # ``libero`` while its benchmark module still imports ``libero.libero``.
    # Alias the same package object without modifying the third-party install.
    import sys
    import libero
    sys.modules.setdefault("libero.libero", libero)
    from libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    return {suite.get_task(task_id).language.lower(): task_id for task_id in range(10)}


class TaggedTransform:
    def __init__(self, transform: RLDSBatchTransform, language_to_task: dict[str, int]):
        self.transform = transform
        self.language_to_task = language_to_task

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        result = self.transform(batch)
        raw = batch["task"]["language_instruction"]
        language = raw.decode().lower() if isinstance(raw, (bytes, np.bytes_)) else str(raw).lower()
        if language not in self.language_to_task:
            raise RuntimeError(f"unknown LIBERO-Spatial instruction: {language}")
        result["task_id"] = self.language_to_task[language]
        result["language_instruction"] = language
        return result


@dataclass
class ModelBundle:
    model: Any
    action_head: torch.nn.Module
    proprio_projector: torch.nn.Module
    processor: Any
    parameters: list[tuple[str, torch.nn.Parameter, str]]
    coordinate_manifest: list[dict[str, Any]]
    num_patches: int


def register_openvla() -> None:
    try:
        AutoConfig.register("openvla", OpenVLAConfig)
    except ValueError:
        pass
    try:
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    except ValueError:
        pass


def tensor_digest(parameters: Iterable[tuple[str, torch.nn.Parameter, str]], group: str) -> str:
    digest = hashlib.sha256()
    for name, parameter, parameter_group in parameters:
        if parameter_group != group:
            continue
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def load_bundle(
    model_kind: str,
    base_checkpoint: Path,
    clean_checkpoint: Path,
    base_components: Path,
    device: torch.device,
    seed: int,
) -> ModelBundle:
    register_openvla()
    checkpoint = base_checkpoint if model_kind == "base" else clean_checkpoint
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    model.vision_backbone.set_num_images_in_input(2)

    # Reset immediately before LoRA construction so both backbones receive the
    # exact same factor tensors and therefore share a local LoRA coordinate.
    set_seed(seed)
    model = get_peft_model(
        model,
        LoraConfig(
            r=32,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        ),
    ).to(device)

    # A concrete Base OFT state needs action-head/proprio tensors.  The saved
    # phase-0 checkpoint provides those tensors and a reproducible definition.
    set_seed(seed + 1)
    action_head = L1RegressionActionHead(model.llm_dim, model.llm_dim, ACTION_DIM).to(device, torch.bfloat16)
    proprio_projector = ProprioProjector(model.llm_dim, PROPRIO_DIM).to(device)
    component_checkpoint = base_components if model_kind == "base" else clean_checkpoint
    action_head.load_state_dict(
        load_component_state_dict(str(next(component_checkpoint.glob("action_head--*checkpoint.pt"))))
    )
    proprio_projector.load_state_dict(
        load_component_state_dict(str(next(component_checkpoint.glob("proprio_projector--*checkpoint.pt"))))
    )

    # Match the training forward behavior; LoRA dropout is zero.  There is no
    # optimizer and parameters are never stepped.
    model.train()
    action_head.train()
    proprio_projector.train()
    parameters: list[tuple[str, torch.nn.Parameter, str]] = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parameters.append((f"vla.{name}", parameter, "lora"))
    for prefix, module in (("action_head", action_head), ("proprio_projector", proprio_projector)):
        for name, parameter in module.named_parameters():
            parameter.requires_grad_(True)
            parameters.append((f"{prefix}.{name}", parameter, "head_proprio"))
    manifest = [
        {"index": index, "name": name, "shape": list(parameter.shape), "numel": parameter.numel(), "group": group}
        for index, (name, parameter, group) in enumerate(parameters)
    ]
    num_patches = model.vision_backbone.get_num_patches() * 2 + 1
    return ModelBundle(model, action_head, proprio_projector, processor, parameters, manifest, num_patches)


def build_dataset(
    bundle: ModelBundle,
    data_root: Path,
    dataset_name: str,
) -> EpisodicRLDSDataset:
    base_transform = RLDSBatchTransform(
        ActionTokenizer(bundle.processor.tokenizer),
        bundle.processor.tokenizer,
        image_transform=bundle.processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=True,
        use_proprio=True,
    )
    return EpisodicRLDSDataset(
        data_root,
        dataset_name,
        TaggedTransform(base_transform, task_languages()),
        resize_resolution=tuple(bundle.model.config.image_sizes),
        shuffle_buffer_size=1,
        # These migrated TFDS builders expose only a ``train`` split.  With
        # image_aug=False the empty augmentation spec is an identity transform.
        train=True,
        image_aug=False,
    )


def official_l1_loss(bundle: ModelBundle, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    ground_truth_actions = batch["actions"].to(device=device, dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = bundle.model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            pixel_values=batch["pixel_values"].to(device=device, dtype=torch.bfloat16),
            labels=batch["labels"].to(device),
            output_hidden_states=True,
            proprio=batch["proprio"].to(device),
            proprio_projector=bundle.proprio_projector,
            noisy_actions=None,
            noisy_action_projector=None,
            diffusion_timestep_embeddings=None,
            use_film=False,
        )
        token_ids = batch["labels"][:, 1:].to(device)
        mask = get_current_action_mask(token_ids) | get_next_actions_mask(token_ids)
        hidden = output.hidden_states[-1][:, bundle.num_patches:-1]
        action_hidden = hidden[mask].reshape(
            batch["input_ids"].shape[0], NUM_ACTIONS_CHUNK * ACTION_DIM, -1
        ).to(torch.bfloat16)
        predicted = bundle.action_head.predict_action(action_hidden)
        return torch.nn.functional.l1_loss(predicted, ground_truth_actions)


def zero_grad(bundle: ModelBundle) -> None:
    for _, parameter, _ in bundle.parameters:
        parameter.grad = None


def _sketch_values(
    sketch: torch.Tensor,
    values: torch.Tensor,
    global_offset: int,
    chunk_size: int,
) -> None:
    mix_add = -7046029254386353131
    mix_mul1 = -4658895280553007687
    mix_mul2 = -7723592293110705685
    mask = (1 << 63) - 1
    for start in range(0, values.numel(), chunk_size):
        part = values[start:start + chunk_size].float()
        indices = torch.arange(
            global_offset + start,
            global_offset + start + part.numel(),
            device=part.device,
            dtype=torch.int64,
        )
        mixed = indices + mix_add
        mixed = torch.bitwise_xor(mixed, torch.bitwise_right_shift(mixed, 30)) * mix_mul1
        mixed = torch.bitwise_xor(mixed, torch.bitwise_right_shift(mixed, 27)) * mix_mul2
        mixed = torch.bitwise_xor(mixed, torch.bitwise_right_shift(mixed, 31))
        positive = torch.bitwise_and(mixed, mask)
        buckets = positive.remainder(sketch.numel())
        signs = torch.where(torch.bitwise_and(torch.bitwise_right_shift(positive, 32), 1) == 0, 1.0, -1.0)
        sketch.scatter_add_(0, buckets, part * signs)


def gradient_sketches(
    bundle: ModelBundle,
    dimension: int,
    chunk_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    device = next(bundle.model.parameters()).device
    scopes = ("all_trainable", "lora", "head_proprio")
    sketches = {scope: torch.zeros(dimension, device=device, dtype=torch.float32) for scope in scopes}
    exact_sq = {scope: torch.zeros((), device=device, dtype=torch.float64) for scope in scopes}
    offsets = {scope: 0 for scope in scopes}
    for _, parameter, group in bundle.parameters:
        grad = parameter.grad
        for scope in ("all_trainable", group):
            if grad is not None:
                flat = grad.detach().reshape(-1)
                exact_sq[scope] += flat.double().square().sum()
                _sketch_values(sketches[scope], flat, offsets[scope], chunk_size)
            offsets[scope] += parameter.numel()
    arrays = {scope: value.cpu().numpy() for scope, value in sketches.items()}
    norms = {scope: math.sqrt(float(value.cpu())) for scope, value in exact_sq.items()}
    return arrays, norms


def selected_indices(length: int, count: int) -> list[int]:
    if length < 1:
        raise RuntimeError("empty trajectory")
    return sorted(set(np.linspace(0, length - 1, min(count, length)).round().astype(int).tolist()))


def trajectory_gradient(
    bundle: ModelBundle,
    trajectory: list[dict[str, Any]],
    collator: PaddedCollatorForActionPrediction,
    frames: int,
    micro_batch: int,
    device: torch.device,
    seed: int,
    sketch_dimension: int,
    sketch_chunk_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, float], float, list[int]]:
    indices = selected_indices(len(trajectory), frames)
    zero_grad(bundle)
    weighted_loss = 0.0
    for offset in range(0, len(indices), micro_batch):
        chosen = indices[offset:offset + micro_batch]
        set_seed(seed + offset)
        batch = collator([trajectory[index] for index in chosen])
        loss = official_l1_loss(bundle, batch, device)
        weight = len(chosen) / len(indices)
        (loss * weight).backward()
        weighted_loss += float(loss.detach().cpu()) * weight
    sketches, norms = gradient_sketches(bundle, sketch_dimension, sketch_chunk_size)
    zero_grad(bundle)
    return sketches, norms, weighted_loss, indices


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_condition(
    args: argparse.Namespace,
    model_kind: str,
    condition: str,
    bundle: ModelBundle,
    device: torch.device,
) -> None:
    data_root = args.clean_root if condition == "clean" else args.trigger_root
    dataset_name = args.clean_dataset if condition == "clean" else args.trigger_dataset
    dataset = build_dataset(bundle, data_root, dataset_name)
    collator = PaddedCollatorForActionPrediction(
        bundle.processor.tokenizer.model_max_length,
        bundle.processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    condition_dir = args.output_dir / "trajectory_sketches" / model_kind / condition
    condition_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[int, int] = defaultdict(int)
    rows = []
    for trajectory in dataset:
        if not trajectory:
            continue
        task_id = int(trajectory[0]["task_id"])
        if any(int(row["task_id"]) != task_id for row in trajectory):
            raise RuntimeError("task changed within trajectory")
        trajectory_index = counts[task_id]
        counts[task_id] += 1
        path = condition_dir / f"task_{task_id:02d}_trajectory_{trajectory_index:03d}.npz"
        seed = stable_seed(args.seed, model_kind, condition, task_id, trajectory_index)
        started = time.monotonic()
        if path.exists():
            saved = np.load(path, allow_pickle=False)
            loss = float(saved["loss"])
            norms = {scope: float(saved[f"norm_{scope}"]) for scope in ("all_trainable", "lora", "head_proprio")}
            indices = saved["frame_indices"].tolist()
        else:
            sketches, norms, loss, indices = trajectory_gradient(
                bundle, trajectory, collator, args.frames, args.micro_batch, device, seed,
                args.sketch_dimension, args.sketch_chunk_size,
            )
            np.savez_compressed(
                path,
                **{f"sketch_{scope}": value for scope, value in sketches.items()},
                **{f"norm_{scope}": value for scope, value in norms.items()},
                loss=loss,
                frame_indices=np.asarray(indices, dtype=np.int32),
                trajectory_length=len(trajectory),
                task_id=task_id,
            )
        rows.append({
            "model": model_kind,
            "condition": condition,
            "task_id": task_id,
            "trajectory_index": trajectory_index,
            "trajectory_length": len(trajectory),
            "sampled_frames": len(indices),
            "loss": loss,
            "gradient_norm_all_trainable": norms["all_trainable"],
            "gradient_norm_lora": norms["lora"],
            "gradient_norm_head_proprio": norms["head_proprio"],
            "elapsed_seconds": time.monotonic() - started,
            "artifact": str(path),
        })
        atomic_json(args.output_dir / "status.json", {
            "status": "running",
            "model": model_kind,
            "condition": condition,
            "task_id": task_id,
            "trajectory_index": trajectory_index,
            "counts": dict(sorted(counts.items())),
        })
        print(
            f"model={model_kind} condition={condition} task={task_id} trajectory={trajectory_index} "
            f"frames={len(indices)} loss={loss:.6f} norm={norms['all_trainable']:.6f}",
            flush=True,
        )
        if args.smoke:
            break
    if args.smoke:
        write_csv(args.output_dir / f"trajectory_metrics_{model_kind}_{condition}.csv", rows)
        return
    expected = {task_id: (43 if condition == "clean" else 10) for task_id in range(10)}
    if condition == "clean":
        if sum(counts.values()) != 432 or set(counts) != set(range(10)):
            raise RuntimeError(f"expected 432 clean trajectories over ten tasks, got {dict(counts)}")
    else:
        if counts != expected:
            raise RuntimeError(f"expected exactly ten trigger trajectories per task, got {dict(counts)}")
    write_csv(args.output_dir / f"trajectory_metrics_{model_kind}_{condition}.csv", rows)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else float("nan")


def load_task_sketches(output_dir: Path, model: str, condition: str, task_id: int, scope: str) -> np.ndarray:
    paths = sorted((output_dir / "trajectory_sketches" / model / condition).glob(
        f"task_{task_id:02d}_trajectory_*.npz"
    ))
    if not paths:
        raise RuntimeError(f"missing sketches: {model}/{condition}/task {task_id}")
    return np.stack([np.load(path, allow_pickle=False)[f"sketch_{scope}"].astype(np.float64) for path in paths])


def displacement_head_proprio(
    base_components: Path,
    clean_checkpoint: Path,
    dimension: int,
    chunk_size: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    base_action = load_component_state_dict(str(next(base_components.glob("action_head--*checkpoint.pt"))))
    clean_action = load_component_state_dict(str(next(clean_checkpoint.glob("action_head--*checkpoint.pt"))))
    base_proprio = load_component_state_dict(str(next(base_components.glob("proprio_projector--*checkpoint.pt"))))
    clean_proprio = load_component_state_dict(str(next(clean_checkpoint.glob("proprio_projector--*checkpoint.pt"))))
    sketch = torch.zeros(dimension, device=device, dtype=torch.float32)
    exact_sq = torch.zeros((), device=device, dtype=torch.float64)
    offset = 0
    for base_state, clean_state in ((base_action, clean_action), (base_proprio, clean_proprio)):
        if list(base_state) != list(clean_state):
            raise RuntimeError("head/proprio parameter coordinate mismatch")
        for name in base_state:
            if base_state[name].shape != clean_state[name].shape:
                raise RuntimeError(f"shape mismatch in displacement: {name}")
            # Required sign: theta_base - theta_clean150k.
            difference = (base_state[name].float() - clean_state[name].float()).to(device).reshape(-1)
            exact_sq += difference.double().square().sum()
            _sketch_values(sketch, difference, offset, chunk_size)
            offset += difference.numel()
    return sketch.cpu().numpy().astype(np.float64), math.sqrt(float(exact_sq.cpu()))


def aggregate(args: argparse.Namespace, device: torch.device) -> None:
    rows = []
    pair_rows = []
    d_sketch, d_norm = displacement_head_proprio(
        args.base_components, args.clean_checkpoint, args.sketch_dimension, args.sketch_chunk_size, device
    )
    for model in ("base", "clean150k"):
        for task_id in range(10):
            clean = load_task_sketches(args.output_dir, model, "clean", task_id, "all_trainable")
            trigger = load_task_sketches(args.output_dir, model, "trigger", task_id, "all_trainable")
            clean_mean, trigger_mean = clean.mean(0), trigger.mean(0)
            clean_unit = clean / np.linalg.norm(clean, axis=1, keepdims=True)
            trigger_unit = trigger / np.linalg.norm(trigger, axis=1, keepdims=True)
            pairs = clean_unit @ trigger_unit.T
            clean_head = load_task_sketches(args.output_dir, model, "clean", task_id, "head_proprio").mean(0)
            trigger_head = load_task_sketches(args.output_dir, model, "trigger", task_id, "head_proprio").mean(0)
            row = {
                "model": model,
                "task_id": task_id,
                "clean_trajectories": clean.shape[0],
                "trigger_trajectories": trigger.shape[0],
                "task_mean_gradient_cosine": cosine(clean_mean, trigger_mean),
                "trajectory_pair_cosine_mean": float(pairs.mean()),
                "trajectory_pair_cosine_std": float(pairs.std(ddof=1)),
                "negative_conflict_fraction": float((pairs < 0).mean()),
                "clean_task_mean_gradient_norm_sketch": float(np.linalg.norm(clean_mean)),
                "trigger_task_mean_gradient_norm_sketch": float(np.linalg.norm(trigger_mean)),
                "trigger_to_clean_norm_ratio": float(np.linalg.norm(trigger_mean) / np.linalg.norm(clean_mean)),
                "d_base_minus_clean_head_proprio_cosine_with_clean_gradient": cosine(d_sketch, clean_head)
                if model == "base" else float("nan"),
                "d_base_minus_clean_head_proprio_cosine_with_trigger_gradient": cosine(d_sketch, trigger_head)
                if model == "base" else float("nan"),
            }
            rows.append(row)
            for clean_index in range(pairs.shape[0]):
                for trigger_index in range(pairs.shape[1]):
                    pair_rows.append({
                        "model": model,
                        "task_id": task_id,
                        "clean_trajectory_index": clean_index,
                        "trigger_trajectory_index": trigger_index,
                        "cosine": float(pairs[clean_index, trigger_index]),
                    })
    write_csv(args.output_dir / "task_metrics.csv", rows)
    write_csv(args.output_dir / "trajectory_pair_cosines.csv", pair_rows)

    summary_rows = []
    metric_names = [
        "task_mean_gradient_cosine",
        "trajectory_pair_cosine_mean",
        "negative_conflict_fraction",
        "trigger_to_clean_norm_ratio",
    ]
    for model in ("base", "clean150k"):
        selected = [row for row in rows if row["model"] == model]
        summary = {"model": model}
        for metric in metric_names:
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=1))
        summary_rows.append(summary)
    base_rows = [row for row in rows if row["model"] == "base"]
    d_summary = {}
    for metric in (
        "d_base_minus_clean_head_proprio_cosine_with_clean_gradient",
        "d_base_minus_clean_head_proprio_cosine_with_trigger_gradient",
    ):
        values = np.asarray([row[metric] for row in base_rows])
        d_summary[f"{metric}_mean"] = float(values.mean())
        d_summary[f"{metric}_std"] = float(values.std(ddof=1))
    write_csv(args.output_dir / "summary_mean_std.csv", summary_rows)

    task_ids = np.arange(10)
    width = 0.38
    fig, axis = plt.subplots(figsize=(12, 5))
    for offset, model in ((-width / 2, "base"), (width / 2, "clean150k")):
        selected = [row for row in rows if row["model"] == model]
        axis.bar(task_ids + offset, [row["task_mean_gradient_cosine"] for row in selected], width, label=model)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(xlabel="LIBERO-Spatial task", ylabel="cosine", title="Task-mean Clean vs Trigger gradients")
    axis.set_xticks(task_ids)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "task_gradient_cosine_base_vs_clean150k.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5))
    for model in ("base", "clean150k"):
        values = [row["cosine"] for row in pair_rows if row["model"] == model]
        axis.hist(values, bins=50, alpha=0.55, density=True, label=model)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set(xlabel="trajectory-pair gradient cosine", ylabel="density", title="Clean/Trigger gradient conflicts")
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "trajectory_gradient_cosine_distribution.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 5))
    axis.bar(
        task_ids - width / 2,
        [row["d_base_minus_clean_head_proprio_cosine_with_clean_gradient"] for row in base_rows],
        width,
        label="Base clean gradient",
    )
    axis.bar(
        task_ids + width / 2,
        [row["d_base_minus_clean_head_proprio_cosine_with_trigger_gradient"] for row in base_rows],
        width,
        label="Base trigger gradient",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(xlabel="LIBERO-Spatial task", ylabel="cosine with d", title="d = Base - Clean150k (head+proprio exact coordinate)")
    axis.set_xticks(task_ids)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "parameter_displacement_direction.png", dpi=180)
    plt.close(fig)

    conclusion = [
        "# Gradient Interference Motivation — concise result",
        "",
        "Primary gradients use the official OFT L1 objective in a common fixed LoRA tangent plus Action Head and Proprio coordinates.",
        "Each trajectory contributes equally after averaging up to 16 uniformly sampled action-chunk frames; tasks are averaged equally.",
        "",
    ]
    for summary in summary_rows:
        conclusion.append(
            f"- {summary['model']}: task-mean cosine "
            f"{summary['task_mean_gradient_cosine_mean']:.4f} ± {summary['task_mean_gradient_cosine_std']:.4f}; "
            f"negative trajectory-pair fraction "
            f"{summary['negative_conflict_fraction_mean']:.3f} ± {summary['negative_conflict_fraction_std']:.3f}; "
            f"Trigger/Clean norm ratio "
            f"{summary['trigger_to_clean_norm_ratio_mean']:.3f} ± {summary['trigger_to_clean_norm_ratio_std']:.3f}."
        )
    conclusion += [
        "",
        f"- Exact head+proprio displacement norm: {d_norm:.6g}.",
        f"- d cosine with Base Clean gradient: {d_summary['d_base_minus_clean_head_proprio_cosine_with_clean_gradient_mean']:.4f} "
        f"± {d_summary['d_base_minus_clean_head_proprio_cosine_with_clean_gradient_std']:.4f}.",
        f"- d cosine with Base Trigger gradient: {d_summary['d_base_minus_clean_head_proprio_cosine_with_trigger_gradient_mean']:.4f} "
        f"± {d_summary['d_base_minus_clean_head_proprio_cosine_with_trigger_gradient_std']:.4f}.",
        "",
        "Important identifiability limit: the migrated legacy Clean-150k directory retains a merged backbone and an adapter whose base is a 10k checkpoint, but not its original step-0 LoRA/optimizer state. Therefore a full historical LoRA-factor displacement is not identifiable. The d plot deliberately uses only the exact common Action Head + Proprio coordinate and must not be described as a full-LoRA historical displacement.",
    ]
    (args.output_dir / "CONCLUSION.md").write_text("\n".join(conclusion) + "\n")
    atomic_json(args.output_dir / "summary.json", {
        "status": "completed",
        "summary": summary_rows,
        "displacement": {"scope": "action_head_plus_proprio", "norm": d_norm, **d_summary},
        "coordinate_warning": "legacy Clean-150k full historical LoRA displacement is not identifiable",
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--clean-checkpoint", type=Path, required=True)
    parser.add_argument("--base-components", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--trigger-root", type=Path, required=True)
    parser.add_argument("--clean-dataset", default="libero_spatial_no_noops")
    parser.add_argument("--trigger-dataset", default="libero_spatial_targeted_place_osc")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--sketch-dimension", type=int, default=65536)
    parser.add_argument("--sketch-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError(f"completed output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    torch.cuda.mem_get_info(device)
    metadata = {
        "status": "running",
        "started_at_unix": time.time(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "source_sha256": sha256(Path(__file__)),
        "loss": "official OpenVLA-OFT continuous-action torch.nn.L1Loss",
        "image_augmentation": False,
        "weights_updated": False,
        "coordinate": "shared fixed rank-32 LoRA tangent + action head + proprio",
    }
    atomic_json(args.output_dir / "experiment_manifest.json", metadata)
    reference_manifest = None
    reference_lora_digest = None
    model_kinds = ("base",) if args.smoke else ("base", "clean150k")
    for model_kind in model_kinds:
        bundle = load_bundle(
            model_kind, args.base_checkpoint, args.clean_checkpoint, args.base_components, device, args.seed
        )
        lora_digest = tensor_digest(bundle.parameters, "lora")
        if reference_manifest is None:
            reference_manifest = bundle.coordinate_manifest
            reference_lora_digest = lora_digest
            atomic_json(args.output_dir / "parameter_coordinate_manifest.json", reference_manifest)
        elif bundle.coordinate_manifest != reference_manifest or lora_digest != reference_lora_digest:
            raise RuntimeError("Base and Clean-150k do not share identical LoRA parameter coordinates")
        for condition in ("clean", "trigger"):
            run_condition(args, model_kind, condition, bundle, device)
        del bundle
        gc.collect()
        torch.cuda.empty_cache()
    if args.smoke:
        atomic_json(args.output_dir / "status.json", {"status": "smoke_passed", "completed_at_unix": time.time()})
        print(f"smoke passed: {args.output_dir}", flush=True)
        return
    aggregate(args, device)
    atomic_json(args.output_dir / "status.json", {"status": "completed", "completed_at_unix": time.time()})
    print(f"completed: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
