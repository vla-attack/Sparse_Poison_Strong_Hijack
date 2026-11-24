"""Frozen policy proxy: fixed XY, shared UV texture + size, three-part loss.

The original five expert trajectories are replayed at fixed recorded states.
No policy parameters, positions, states, actions or proprio labels are updated.
Defaults to P1-150K; clean150k selects the original pure Clean checkpoint.
merged_dual_lora loads fused weights and heads matching --proxy-step, without
loading LoRA again. Optional PCGrad jointly updates size and texture only.
Use torchrun for shared four-GPU frame batches, or python on one GPU.
"""

import os
from experiments.vla_safety.training.cuda_device_mapping import isolate_local_cuda_device
_binding = isolate_local_cuda_device()
if _binding is not None:
    print('CUP_CUDA_BINDING', _binding, flush=True)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import time
from types import MethodType

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import torch.nn.functional as F

from experiments.vla_safety import optimize_uploaded_cup_bowl_attention as original
from experiments.vla_safety.training.multimodal_trigger_objective import (
    VERSION, INSTRUCTION, combine_losses, instruction_positions, normalize_raw_libero_actions,
)
from experiments.vla_safety.training.dual_lora_export import register_openvla
from experiments.vla_safety.training.distributed_attention_batches import distributed_batch

ROOT = Path("/project_c20250301/241207013/project/huangsimin")
DEFAULT_PROXY = ROOT / "openvla-oft/prepared_models/p1_150k_merged_fresh_lora_adapt_20261027"
DEFAULT_ORIGINAL = ROOT / "modified_libero_rlds_commonpose5_native256_norotation_v3_cup_plate_triggeronly_20261022"


def model_factory(checkpoint, proxy_kind="p1_150k", proxy_step=5000, proxy_base=DEFAULT_PROXY):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector
    register_openvla()
    if proxy_kind == "merged_dual_lora":
        from experiments.vla_safety.dual_lora_eval_protocol import checkpoint_info
        identity = checkpoint_info(checkpoint, step=proxy_step, base=proxy_base)
        if identity["clean_scale"] != 1.0 or identity["trigger_scale"] != 1.0:
            raise ValueError("This ablation keeps both trained LoRA merge scales at 1.0")
        head_step = identity["step"]
    elif proxy_kind == "p1_150k":
        audit = json.loads((checkpoint / "p1_merged_base_audit.json").read_text())
        if audit["source_updates"] != 150000 or (checkpoint / "lora_adapter").exists():
            raise ValueError("Require the verified merged P1-150K base, without reloading an adapter")
        source = Path(audit["source_checkpoint"])
        mapping = source / "step_index_mapping.json"
        if (json.loads(mapping.read_text())["actual_optimizer_updates"] != 150000 or
                hashlib.sha256(mapping.read_bytes()).hexdigest() != audit["source_mapping_sha256"]):
            raise ValueError("P1 source optimizer-step identity differs from the recorded audit")
        for shard in audit["shards"]:
            if not os.path.samefile(checkpoint / shard["name"], source / shard["name"]):
                raise ValueError("P1 prepared weights differ from the verified source shards")
        for name, expected in audit["source_head_sha256"].items():
            if hashlib.sha256((checkpoint / f"{name}--150000_checkpoint.pt").read_bytes()).hexdigest() != expected:
                raise ValueError(f"P1 {name} source identity changed")
        head_step = 150000
    elif proxy_kind == "clean150k":
        from experiments.vla_safety.training.clean_lora_continuation import read_original_adapter
        state, identity = read_original_adapter(checkpoint)
        del state
        stats = json.loads((checkpoint / "dataset_statistics.json").read_text())
        if (set(stats) != {"libero_spatial_no_noops"} or
                stats["libero_spatial_no_noops"]["num_trajectories"] != 432):
            raise ValueError("Expected the original pure Clean150k checkpoint")
        head_step = 150000
    else:
        raise ValueError(f"Unknown frozen proxy kind: {proxy_kind}")
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=False,
        local_files_only=True, attn_implementation="eager",
    ).cuda()
    model.vision_backbone.set_num_images_in_input(2)
    model.get_base_model = MethodType(lambda self: self, model)
    head = L1RegressionActionHead(input_dim=model.llm_dim, hidden_dim=model.llm_dim, action_dim=7)
    proprio = ProprioProjector(llm_dim=model.llm_dim, proprio_dim=8)
    for name, module in (("action_head", head), ("proprio_projector", proprio)):
        state = torch.load(checkpoint / f"{name}--{head_step}_checkpoint.pt", map_location="cpu", weights_only=True)
        module.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
        module.float().cuda()
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=False, local_files_only=True)
    named = [(prefix + "." + name, parameter, prefix)
             for prefix, module in (("vla", model), ("action_head", head), ("proprio", proprio))
             for name, parameter in module.named_parameters()]
    return model, head, proprio, processor, named


class CompositeReader(original.ActionQueryReader):
    def __init__(self, base, head):
        super().__init__(base, head)
        self.prediction = None
        previous = head.predict_action
        def capture(*args, **kwargs):
            self.prediction = previous(*args, **kwargs)
            return self.prediction
        head.predict_action = capture

    def clear(self):
        super().clear()
        self.prediction = None


class Runtime(original.Runtime):
    def __init__(self, args, output):
        old_factory, old_checkpoint, old_reader = original.d.load_model, original.d.CHECKPOINT, original.ActionQueryReader
        original.d.load_model = lambda: model_factory(args.proxy_checkpoint, getattr(args, "proxy_kind", "p1_150k"),
                                                       getattr(args, "proxy_step", 5000), getattr(args, "proxy_base", DEFAULT_PROXY))
        original.d.CHECKPOINT = args.proxy_checkpoint
        original.ActionQueryReader = CompositeReader
        try:
            super().__init__(output)
        finally:
            original.d.load_model, original.d.CHECKPOINT, original.ActionQueryReader = old_factory, old_checkpoint, old_reader
        self.mode = "joint"  # Geometry changes invalidate raster caches; XY is a constant zero offset.
        self.weights = (args.action_weight, args.visual_weight, args.text_weight)
        self.text_keys = torch.tensor(instruction_positions(self.processor.tokenizer, self.ids[0]), device="cuda") + 513
        for episode in self.episodes:
            with np.load(episode.folder / "trajectory.npz") as archive:
                self.raw[episode.task]["actions"] = archive["executed_action"].copy()
        if args.smoke:
            self.validation = [next(row for row in self.validation if row[0].task == episode.task)
                               for episode in self.episodes]

    def render(self, episode, step):
        if self.position.requires_grad or bool(self.position.any()):
            raise RuntimeError("XY is fixed at zero; position search is not part of this experiment")
        return super().render(episode, step)

    def forward(self, episode, step):
        images, cup, bowl = self.render(episode, step)
        proprio = original.d.normalize(
            original.training_proprio(self.raw[episode.task]["proprio"][step]),
            self.stats["libero_spatial_no_noops"]["proprio"],
        )
        self.base.predict_action(
            input_ids=self.ids, attention_mask=self.am, pixel_values=original.pixels(images, False),
            unnorm_key="libero_spatial_no_noops", proprio=proprio, proprio_projector=self.proprio,
            action_head=self.head, use_film=False,
        )
        full_attention = torch.stack(self.reader.weights).mean(0)
        attention = self.reader.attention()
        self.prediction = self.reader.prediction.float()
        self.text_loss = full_attention[self.text_keys].sum()
        density, valid = original.regional_density(attention, cup, bowl)
        self.reader.clear()
        return density, valid, (images, attention, cup, bowl)

    def loss(self, episode, step):
        reference, reference_valid = self.reference(episode, step)
        density, valid, extra = self.forward(episode, step)
        visual, values = original.cup_up_bowl_down(density, valid, reference, reference_valid)
        actions = self.raw[episode.task]["actions"][step:step + 8]
        if actions.shape != (8, 7):
            raise ValueError("Optimization frames must have a complete eight-action label window")
        target = torch.from_numpy(normalize_raw_libero_actions(actions, self.stats["libero_spatial_no_noops"]["action"])).cuda()
        action_l1 = F.l1_loss(self.prediction.reshape(8, 7), target)
        text = self.text_loss
        self.objective_parts = (action_l1, visual, text)
        loss = combine_losses(action_l1, visual, text, *self.weights)
        metrics = {key: None if value is None else float(value.detach()) for key, value in values.items()}
        metrics.update(action_l1=float(action_l1.detach()), visual_loss=float(visual.detach()), text_loss=float(text.detach()))
        self.prediction, self.text_loss = None, None
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite multimodal optimization objective")
        return loss, metrics, extra

    def batch(self, choices, backward=False):
        if dist.is_initialized():
            return distributed_batch(self, choices, backward)
        return super().batch(choices, backward)

    def evaluate(self):
        rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
        rows = []
        with torch.no_grad():
            for index in range(rank, len(self.validation), world):
                episode, step, stage = self.validation[index]
                loss, metrics, _ = self.loss(episode, step)
                rows.append(dict(index=index, task=episode.task, step=step, stage=stage, loss=float(loss), **metrics))
        if dist.is_initialized():
            chunks = [None] * world
            dist.all_gather_object(chunks, rows)
            rows = [row for chunk in chunks for row in chunk]
        rows.sort(key=lambda row: row["index"])
        keys = [key for key in rows[0] if key not in ("index", "task", "step", "stage")]
        result = {key: float(np.mean([row[key] for row in rows if row[key] is not None]))
                  if any(row[key] is not None for row in rows) else None for key in keys}
        result["frames"] = rows
        return result


def run(runtime, args, output):
    rank = dist.get_rank() if dist.is_initialized() else 0
    use_pcgrad = getattr(args, "cup_optimizer", "legacy") == "pcgrad"
    if use_pcgrad:
        runtime.size.requires_grad_(True)
    optimizer = torch.optim.Adam([{"params": [runtime.texture], "lr": .01},
                                  {"params": [runtime.size], "lr": .005}] if use_pcgrad else [runtime.texture], lr=0.01)
    rng, queues = random.Random(20261101), {}
    config = {"version": VERSION, "proxy_checkpoint": str(args.proxy_checkpoint.resolve()),
              "original_trigger_root": str(args.original_trigger_root.resolve()), "iterations": args.iterations,
              "schedule_iterations": args.schedule_iterations, "weights": dict(zip(("action", "visual", "text"), runtime.weights)),
              "variables": ["UV_texture_64x64x3", "relative_size"], "fixed_xy_m": [0.0, 0.0],
              "fixed_trajectory": True, "text_objective": "minimize action-query attention mass to instruction tokens",
              "seeds": {"proxy_setup": 20261023, "frame_sampling": 20261101},
              "temporal_split": runtime.splits, "image_aug": False, "center_crop_proxy": False}
    if use_pcgrad:
        config.update(cup_optimizer="pcgrad", proxy_kind=args.proxy_kind,
                      pcgrad=dict(objectives=["action", "visual", "text"], global_reduce_before_projection=True,
                                  reduction="sum", geometry="autograd through nvdiffrast; ROI masks stop-gradient",
                                  texture_lr=[.01, .001], size_lr=[.005, .0005], tv_after_projection=1e-4))
    if getattr(args, "proxy_kind", "p1_150k") == "clean150k":
        from experiments.vla_safety.training.clean_lora_continuation import read_original_adapter
        state, identity = read_original_adapter(args.proxy_checkpoint)
        del state
        config["clean150k_identity"] = identity
    if getattr(args, "proxy_kind", "p1_150k") == "merged_dual_lora":
        from experiments.vla_safety.dual_lora_eval_protocol import checkpoint_info
        config["proxy_kind"] = "merged_dual_lora"
        config["frozen_proxy_identity"] = checkpoint_info(args.proxy_checkpoint, step=args.proxy_step, base=args.proxy_base)
    if getattr(args, "initial_parameters", None) is not None:
        config["initial_parameters"] = str(args.initial_parameters.resolve())
        config["initial_parameters_sha256"] = hashlib.sha256(args.initial_parameters.read_bytes()).hexdigest()
    if getattr(args, "parent_candidate", None) is not None:
        config["parent_candidate_sha256"] = hashlib.sha256(args.parent_candidate.read_bytes()).hexdigest()
    start, best = 0, float("inf")
    if args.resume:
        state = torch.load(output / "optimization_state.pt", map_location="cpu", weights_only=False)
        if state["config"] != config:
            raise ValueError("Optimization configuration changed since saved state")
        runtime.restore(state["parameters"])
        optimizer.load_state_dict(state["optimizer"])
        rng.setstate(state["rng"])
        queues, start, best = state["queues"], state["iteration"], state["best"]
    if rank == 0:
        original.d.atomic_json(output / "config.json", config)

    def evaluate_save(iteration):
        nonlocal best
        result = runtime.evaluate()
        if result["loss"] < best:
            best = result["loss"]
            if rank == 0:
                torch.save({"parameters": runtime.parameters_state(), "validation": result, "iteration": iteration},
                           output / "best_parameters.pt")
        return result

    if not args.resume:
        initial = evaluate_save(0)
        if rank == 0:
            original.d.atomic_json(output / "initial_validation.json", initial)
    for iteration in range(start + 1, args.iterations + 1):
        choices = []
        for episode in runtime.episodes:
            for _ in range(2):
                if not queues.get(episode.task):
                    queues[episode.task] = rng.sample(runtime.train["all"][episode.task], len(runtime.train["all"][episode.task]))
                choices.append((episode, queues[episode.task].pop()))
        optimizer.zero_grad(set_to_none=True)
        optimizer.param_groups[0]["lr"] = original.cosine_step(iteration, args.schedule_iterations, .01, .001)
        if use_pcgrad:
            from experiments.vla_safety.training.cup_pcgrad import objective_gradients, project_conflicting, assign_gradient
            gradients, metrics = objective_gradients(runtime, choices)
            vector, conflict = project_conflicting(gradients, seed=20261102 + iteration)
            assign_gradient(runtime, vector)
            optimizer.param_groups[1]["lr"] = original.cosine_step(iteration, args.schedule_iterations, .005, .0005)
            conflict["size_task_gradients"] = gradients[:, -1].tolist()
        else:
            metrics = runtime.batch(choices, backward=True)
        tv = ((runtime.texture[:, :, 1:] - runtime.texture[:, :, :-1]).square().mean()
              + (runtime.texture[:, :, :, 1:] - runtime.texture[:, :, :, :-1]).square().mean())
        (tv * 1e-4).backward()
        if not torch.isfinite(runtime.texture.grad).all():
            raise FloatingPointError("Nonfinite texture gradient")
        if use_pcgrad and not torch.isfinite(runtime.size.grad).all():
            raise FloatingPointError("Nonfinite size gradient")
        optimizer.step()
        with torch.no_grad():
            runtime.texture.clamp_(-1, 1)
            if use_pcgrad:
                runtime.size.clamp_(-.1, .1)
        # Only SIZE search remains. No X/Y parameters or coordinate-search calls.
        search = {"method": "joint PCGrad autograd"} if use_pcgrad else runtime.candidate_search(
            choices, "size", original.cosine_step(iteration, args.schedule_iterations, .005, .0005))
        row = dict(iteration=iteration, train=metrics, size_search=search, relative_scale=1 + float(runtime.size), fixed_xy_m=[0., 0.])
        if use_pcgrad:
            row.update(pcgrad=conflict, texture_lr=optimizer.param_groups[0]["lr"], size_lr=optimizer.param_groups[1]["lr"])
        if iteration % 10 == 0 or iteration == args.iterations:
            row["validation"] = evaluate_save(iteration)
        if rank == 0:
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            torch.save(dict(config=config, parameters=runtime.parameters_state(), optimizer=optimizer.state_dict(),
                            iteration=iteration, best=best, rng=rng.getstate(), queues=queues), output / "optimization_state.pt")
            original.d.atomic_json(output / "status.json", {"state": "running", "completed": iteration, "total": args.iterations})
            print("FIXED_XY_ITERATION", json.dumps(row), flush=True)
        gc.collect()
    unchanged = runtime.hash == original.d.fingerprint(runtime.named)
    if not unchanged or any(parameter.grad is not None for _, parameter, _ in runtime.named):
        raise RuntimeError("Frozen proxy model changed or accumulated parameter gradients")
    if dist.is_initialized():
        dist.barrier()
    best_state = torch.load(output / "best_parameters.pt", map_location="cpu", weights_only=False)
    runtime.restore(best_state["parameters"])
    if rank == 0:
        original.d.atomic_json(output / "summary.json", {"version": VERSION, "model_unchanged": True,
            "best_iteration": best_state["iteration"], "best": best_state["validation"],
            "fixed_xy_m": [0., 0.], "actual_head_query_proof_calls": runtime.reader.proof_count})
        original.d.atomic_json(output / "status.json", {"state": "completed", "completed": args.iterations, "total": args.iterations})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--proxy-checkpoint", type=Path, default=DEFAULT_PROXY)
    parser.add_argument("--proxy-kind", choices=("p1_150k", "clean150k", "merged_dual_lora"), default="p1_150k")
    parser.add_argument("--proxy-step", type=int, default=5000)
    parser.add_argument("--proxy-base", type=Path, default=DEFAULT_PROXY)
    parser.add_argument("--cup-optimizer", choices=("legacy", "pcgrad"), default="legacy")
    parser.add_argument("--initial-parameters", type=Path, help="Start from an existing best_parameters.pt cup")
    parser.add_argument("--parent-candidate", type=Path, help="Record the parent cup when reoptimizing after training")
    parser.add_argument("--original-trigger-root", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--schedule-iterations", type=int, default=300)
    parser.add_argument("--action-weight", type=float, default=1.)
    parser.add_argument("--visual-weight", type=float, default=1.)
    parser.add_argument("--text-weight", type=float, default=1.)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Check three-part texture gradients against finite differences")
    parser.add_argument("--export-root", type=Path)
    args = parser.parse_args()
    if not 1 <= args.iterations <= args.schedule_iterations:
        raise ValueError("Invalid optimization iteration budget")
    combine_losses(0., 0., 0., args.action_weight, args.visual_weight, args.text_weight)
    if args.parent_candidate is not None and args.initial_parameters is None:
        raise ValueError("A parent candidate requires its saved initial parameters")
    if args.proxy_kind == "merged_dual_lora" and (args.initial_parameters is None or args.parent_candidate is None):
        raise ValueError("Post-training reoptimization requires the saved initial cup parameters and parent candidate")
    if args.parent_candidate is not None:
        from experiments.vla_safety.dual_lora_eval_protocol import candidate_info
        candidate_info(args.parent_candidate)
        parent = json.loads(args.parent_candidate.read_text())
        if (Path(parent["source_directory"]) / "best_parameters.pt").resolve() != args.initial_parameters.resolve():
            raise ValueError("Initial parameters do not belong to the recorded parent cup")
    source_manifest = json.loads((args.original_trigger_root / "DATASET_MANIFEST.json").read_text())
    if (source_manifest.get("episodes") != 5 or source_manifest.get("frames") != 668 or
            source_manifest.get("all_non_language_features_identical") is not True):
        raise ValueError("Require the original unoptimized five-trajectory/668-frame trigger dataset")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Prismatic's module-level Overwatch creates accelerate.PartialState(),
    # which may already initialize the default group during imports.
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank() if dist.is_initialized() else 0
    worker_output = args.output_dir if rank == 0 else args.output_dir / "workers" / f"rank{rank}"
    worker_output.mkdir(parents=True, exist_ok=True)
    runtime = Runtime(args, worker_output)
    try:
        if args.verify:
            old_checkpoint, old_version = original.d.CHECKPOINT, original.VERSION
            original.d.CHECKPOINT, original.VERSION = args.proxy_checkpoint, VERSION
            try:
                original.verify(runtime, worker_output)
            finally:
                original.d.CHECKPOINT, original.VERSION = old_checkpoint, old_version
            runtime.reset("joint")
        if args.initial_parameters is not None:
            state = torch.load(args.initial_parameters, map_location="cpu", weights_only=True)["parameters"]
            for name, shape in (("texture", (1, 3, 64, 64)), ("position", (2,)), ("size", (1,))):
                if tuple(state[name].shape) != shape or not torch.isfinite(state[name]).all():
                    raise ValueError(f"Invalid initial cup parameter: {name}")
            if bool(state["position"].any()) or state["texture"].abs().max() > 1 or state["size"].abs().max() > .100001:
                raise ValueError("Initial cup violates fixed XY, texture or size bounds")
            if args.parent_candidate is not None and abs(1 + state["size"].item() - parent["relative_scale"]) > 1e-7:
                raise ValueError("Initial cup size differs from the parent candidate")
            runtime.restore(state)
        run(runtime, args, args.output_dir)
        # All replicas are synchronized before serial encoding. Release NCCL
        # before rank zero's 668-frame export so idle ranks cannot time out.
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        if args.export_root and rank == 0:
            from experiments.vla_safety.export_fixed_xy_trigger_dataset import export_dataset
            export_dataset(runtime, args, args.export_root)
    finally:
        runtime.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
