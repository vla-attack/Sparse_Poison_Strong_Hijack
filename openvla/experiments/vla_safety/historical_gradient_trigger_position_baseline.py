"""Bilevel Trigger-position baseline with fresh MuJoCo trajectories.

For each LIBERO-Spatial task one table-local xyz is shared by 50 initial
states.  Each outer iteration uses fresh strict OSC trajectories at the
incumbent xyz.  The inner statistic is the state-balanced mean gradient of the
official OpenVLA-OFT continuous action-chunk L1 loss.  Its exact cosine to the
historical Base-Clean150k direction is computed in the final 7-D Action Head
layer, the largest tractable coordinate whose endpoints are both available.

The derivative of that cosine objective with respect to xyz is evaluated in
two passes.  Pass 1 measures the mean parameter gradient.  Pass 2 contracts
per-chunk mixed derivatives with d(objective)/d(mean-gradient), avoiding one
unbounded graph over all 50 trajectories.  MuJoCo dynamics remain outside
autograd; every proposed xyz is accepted only through fresh collision-free
full-horizon expert rollouts.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gc
import hashlib
import json
import math
import multiprocessing
import os
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch

from experiments.libero.run_targeted_grasp_osc_expert import OSCExpertConfig, setup
from experiments.libero.run_targeted_place_osc_expert import PlaceConfig, _attempt_config, run_episode
from experiments.libero.targeted_grasp_rlds import quat_xyzw_to_axis_angle
from experiments.openvla_utils import load_component_state_dict
from experiments.vla_safety.gradient_interference_motivation_v2 import (
    load_bundle,
    official_l1_loss,
)
from experiments.vla_safety.real_mesh_openvla_gradient_smoke import _composite, _policy_pixels
from experiments.vla_safety.rendering.camera import (
    build_mujoco_camera_matrices,
    depth_buffer_to_metric,
    mujoco_depth_planes,
)
from experiments.vla_safety.rendering.textured_mesh import (
    extract_compiled_texture_mesh,
    render_textured_mesh,
)
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform


CAMERAS = ("agentview", "robot0_eye_in_hand")
ALLOWED_CONTACT_PREFIXES = ("gripper0_", "plate_1_")
_WORKER_SUITE: Any | None = None


def chunk_starts(length: int, width: int = 8) -> list[int]:
    return list(range(0, length, width))


def normalized(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    scaled = np.clip(2.0 * (values - low) / (high - low + 1.0e-8) - 1.0, -1.0, 1.0)
    return np.where(mask, scaled, values).astype(np.float32)


def training_proprio(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    return np.r_[raw[:3], quat_xyzw_to_axis_angle(raw[3:7]), raw[7:9]].astype(np.float32)


def cup_mesh(sim: Any, mujoco: Any, visual_geom_name: str) -> Any:
    from types import SimpleNamespace
    geom_id = sim.model.geom_name2id(visual_geom_name)
    if int(sim.model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
        raise RuntimeError(f"trigger visual geom is not a mesh: {visual_geom_name}")
    mesh_id = int(sim.model.geom_dataid[geom_id])
    start = int(sim.model.mesh_vertadr[mesh_id])
    count = int(sim.model.mesh_vertnum[mesh_id])
    vertices = np.asarray(sim.model.mesh_vert[start:start + count], dtype=np.float32).copy()
    return SimpleNamespace(mesh_id=mesh_id, geom_id=geom_id, vertices=vertices)


def render_view(
    sim: Any,
    camera: str,
    resolution: int,
    visual_geom_id: int,
    compiled: Any,
    context: Any,
    table_delta_world: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    *,
    local_scale: torch.Tensor | float = 1.0,
    material_rgb: torch.Tensor | None = None,
    return_aux: bool = False,
    raster_cache: dict | None = None,
    background_cache: dict | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if background_cache is not None and 'rgb' in background_cache:
        background_rgb_np, background_depth_np = background_cache['rgb'], background_cache['depth']
    else:
        rgba = np.asarray(sim.model.geom_rgba[visual_geom_id], dtype=np.float32).copy()
        try:
            sim.model.geom_rgba[visual_geom_id, 3] = 0.0
            background_rgb_np, background_depth_np = sim.render(
                width=resolution, height=resolution, camera_name=camera, depth=True
            )
        finally:
            sim.model.geom_rgba[visual_geom_id] = rgba
        if background_cache is not None:
            background_cache.update(rgb=np.asarray(background_rgb_np).copy(), depth=np.asarray(background_depth_np).copy())
    camera_id = sim.model.camera_name2id(camera)
    near, far = mujoco_depth_planes(
        float(sim.model.vis.map.znear), float(sim.model.vis.map.zfar), float(sim.model.stat.extent)
    )
    camera_position = torch.as_tensor(
        np.asarray(sim.data.cam_xpos[camera_id]).copy(), device=device
    ).float()
    camera_rotation = torch.as_tensor(
        np.asarray(sim.data.cam_xmat[camera_id]).reshape(3, 3).copy(), device=device
    ).float()
    matrices = build_mujoco_camera_matrices(
        camera_position, camera_rotation, float(sim.model.cam_fovy[camera_id]),
        resolution, resolution, near, far,
    )
    geom_rotation = torch.as_tensor(
        np.asarray(sim.data.geom_xmat[visual_geom_id]).reshape(3, 3).copy(), device=device
    ).float()
    geom_position = torch.as_tensor(
        np.asarray(sim.data.geom_xpos[visual_geom_id]).copy(), device=device
    ).float()
    rendered = render_textured_mesh(
        context, compiled, geom_rotation, geom_position, table_delta_world,
        matrices.world_to_camera, matrices.world_to_clip, resolution,
        local_scale=local_scale,
        material_rgb=material_rgb,
        ambient_strength=args.ambient_strength,
        diffuse_strength=args.diffuse_strength,
        light_direction_world=torch.tensor(args.light_direction_world, device=device),
        raster_cache=raster_cache,
    )
    foreground_rgb = torch.flip(rendered.rgb, dims=(-2,))
    foreground_alpha = torch.flip(rendered.soft_alpha, dims=(-2,))
    foreground_depth = torch.flip(rendered.depth, dims=(-2,))
    background_rgb = (
        torch.from_numpy(np.asarray(background_rgb_np).copy()).to(device)
        .permute(2, 0, 1)[None].float() / 255.0
    )
    background_depth = depth_buffer_to_metric(
        torch.from_numpy(np.asarray(background_depth_np, dtype=np.float64)), near, far
    ).to(device=device, dtype=torch.float32)[None, None]
    composite, effective_alpha = _composite(
        background_rgb, background_depth, foreground_rgb, foreground_alpha, foreground_depth,
        margin_m=args.depth_margin_m, temperature_m=args.depth_temperature_m, hard=False,
    )
    if return_aux:
        return composite, effective_alpha, background_rgb
    return composite


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_contact_name(name: str) -> bool:
    return name == "table_collision" or name.startswith(ALLOWED_CONTACT_PREFIXES)


def collision_free(row: dict[str, Any]) -> bool:
    return bool(row.get("success") and row.get("dataset_eligible")) and all(
        safe_contact_name(name) for name in row.get("contact_counterparts", [])
    )


def unexpected_contacts(row: dict[str, Any]) -> list[str]:
    return sorted(name for name in row.get("contact_counterparts", []) if not safe_contact_name(name))


def worker_suite() -> Any:
    """Create one reusable LIBERO suite per spawned MuJoCo worker."""
    global _WORKER_SUITE
    if _WORKER_SUITE is None:
        from libero.libero import benchmark
        _WORKER_SUITE = benchmark.get_benchmark_dict()["libero_spatial"]()
    return _WORKER_SUITE


def screen_state_job(payload: tuple[int, int, tuple[float, float, float], PlaceConfig, int, int, int]) -> tuple[int, int | None, list[dict[str, Any]]]:
    task_id, state_id, pose, cfg_base, grasp_profiles, trajectory_seed, resolution = payload
    attempts = []
    selected_profile = None
    for profile in range(grasp_profiles):
        seed = trajectory_seed + profile
        cfg = _attempt_config(replace(cfg_base, table_local_pose=pose), profile)
        row = run_episode(
            worker_suite(), task_id, state_id, None, seed, resolution, cfg,
            save_artifacts=False, return_arrays=False,
        )
        attempts.append({
            "profile": profile,
            "seed": seed,
            "success": bool(row.get("success")),
            "collision_free": collision_free(row),
            "failure_reason": row.get("failure_reason"),
            "unexpected_contacts": unexpected_contacts(row),
        })
        if collision_free(row):
            selected_profile = profile
            break
    return state_id, selected_profile, attempts


def save_state_job(payload: tuple[int, int, tuple[float, float, float], PlaceConfig, int, int, int, Path]) -> tuple[int, int, int, dict[str, Any]]:
    task_id, state_id, pose, cfg_base, profile, trajectory_seed, resolution, artifact = payload
    seed = trajectory_seed + profile
    artifact.mkdir()
    cfg = _attempt_config(replace(cfg_base, table_local_pose=pose), profile)
    row = run_episode(worker_suite(), task_id, state_id, artifact, seed, resolution, cfg)
    return state_id, profile, seed, row


def initial_candidates(base: Sequence[float]) -> list[list[float]]:
    x, y, z = map(float, base)
    offsets = [(0.0, 0.0)]
    for radius in (0.04, 0.08, 0.12, 0.16, 0.20):
        offsets.extend((
            (-radius, 0.0), (radius, 0.0), (0.0, radius), (0.0, -radius),
            (-radius, radius), (radius, radius), (-radius, -radius), (radius, -radius),
        ))
    return [[x + dx, y + dy, z] for dx, dy in offsets]


@dataclass(frozen=True)
class SavedEpisode:
    path: Path
    task_id: int
    state_id: int
    profile: int
    seed: int
    start: int
    end: int

    @property
    def steps(self) -> int:
        return self.end - self.start


def load_collection(path: Path) -> tuple[list[float], list[SavedEpisode]]:
    summary = json.loads((path / "summary.json").read_text())
    expected_states = int(summary.get("states", 50))
    if summary.get("status") != "completed" or len(summary.get("episodes", [])) != expected_states:
        raise RuntimeError(f"incomplete task collection: {path}")
    episodes = []
    for row in summary["episodes"]:
        episode_path = Path(row["artifact_dir"])
        trajectory = json.loads((episode_path / "trajectory.json").read_text())
        episodes.append(SavedEpisode(
            episode_path,
            int(row["task_id"]),
            int(row["state_id"]),
            int(row["profile"]),
            int(row["seed"]),
            int(trajectory["stage_events"]["pregrasp"]),
            int(trajectory["steps"]),
        ))
    return list(map(float, summary["position_table_local_m"])), episodes


def collect_task(
    suite: Any,
    task_id: int,
    candidates: Sequence[Sequence[float]],
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[float], list[SavedEpisode]]:
    """Select one shared pose that succeeds without collateral contact in 50 states."""
    if (output_dir / "summary.json").is_file():
        return load_collection(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_base = PlaceConfig(plate_xy_tolerance_m=args.plate_xy_tolerance_m)
    candidate_reports = []
    executor = None
    if args.cpu_workers > 1:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=args.cpu_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
    try:
      for candidate_index, candidate in enumerate(candidates):
        pose = tuple(map(float, candidate))
        profile_by_state: dict[int, int] = {}
        diagnostic = {"candidate_index": candidate_index, "position_table_local_m": list(pose), "states": []}
        payloads = [
            (task_id, state_id, pose, cfg_base, args.grasp_profiles, args.trajectory_seed, args.resolution)
            for state_id in range(args.states_per_task)
        ]
        # Preserve the cheap early rejection of an obviously colliding pose.
        # Only fan out the remaining 49 states after state 0 succeeds.
        first = screen_state_job(payloads[0]) if executor is None else executor.submit(screen_state_job, payloads[0]).result()
        results: Iterable[tuple[int, int | None, list[dict[str, Any]]]] = (first,)
        if first[1] is not None and len(payloads) > 1:
            rest = map(screen_state_job, payloads[1:]) if executor is None else executor.map(screen_state_job, payloads[1:], chunksize=1)
            results = (first, *rest)
        for state_id, profile, state_attempts in results:
            if profile is not None:
                profile_by_state[state_id] = profile
            diagnostic["states"].append({"state_id": state_id, "attempts": state_attempts})
            atomic_json(output_dir / "candidate_progress.json", {
                "task_id": task_id,
                "candidate_reports": candidate_reports,
                "current": diagnostic,
            })
        diagnostic["feasible"] = len(profile_by_state) == args.states_per_task
        candidate_reports.append(diagnostic)
        atomic_json(output_dir / "candidate_search.json", {
            "task_id": task_id, "candidate_reports": candidate_reports,
        })
        print(
            f"task={task_id} candidate={candidate_index}/{len(candidates)-1} "
            f"pose={pose} states_passed={len(profile_by_state)}/{args.states_per_task}",
            flush=True,
        )
        if not diagnostic["feasible"]:
            continue

        episodes = []
        accepted_dir = output_dir / "accepted"
        accepted_dir.mkdir(exist_ok=False)
        save_payloads = []
        for state_id in range(args.states_per_task):
            profile = profile_by_state[state_id]
            artifact = accepted_dir / f"task_{task_id:02d}_state_{state_id:02d}_profile_{profile:02d}"
            save_payloads.append((
                task_id, state_id, pose, cfg_base, profile,
                args.trajectory_seed, args.resolution, artifact,
            ))
        saved_rows = map(save_state_job, save_payloads) if executor is None else executor.map(save_state_job, save_payloads, chunksize=1)
        for state_id, profile, seed, row in saved_rows:
            artifact = accepted_dir / f"task_{task_id:02d}_state_{state_id:02d}_profile_{profile:02d}"
            if not collision_free(row):
                raise RuntimeError(
                    f"deterministic save replay failed audit task={task_id} state={state_id}: "
                    f"success={row.get('success')} contacts={unexpected_contacts(row)}"
                )
            trajectory = json.loads((artifact / "trajectory.json").read_text())
            episodes.append({
                "task_id": task_id,
                "state_id": state_id,
                "profile": profile,
                "seed": seed,
                "artifact_dir": str(artifact.resolve()),
                "steps": int(trajectory["steps"]),
                "training_start": int(trajectory["stage_events"]["pregrasp"]),
                "trajectory_json_sha256": sha256(artifact / "trajectory.json"),
                "trajectory_npz_sha256": sha256(artifact / "trajectory.npz"),
            })
            atomic_json(output_dir / "save_progress.json", {
                "task_id": task_id, "position_table_local_m": list(pose), "episodes": episodes,
            })
        summary = {
            "status": "completed",
            "task_id": task_id,
            "task": suite.get_task(task_id).language,
            "position_table_local_m": list(pose),
            "states": args.states_per_task,
            "collision_policy": "allow only table, gripper0_*, and destination plate_1_* contacts",
            "candidate_index": candidate_index,
            "episodes": episodes,
        }
        atomic_json(output_dir / "summary.json", summary)
        return load_collection(output_dir)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    atomic_json(output_dir / "status.json", {
        "status": "failed", "reason": "no shared pose passed all states", "task_id": task_id,
    })
    raise RuntimeError(f"no shared collision-free pose for task {task_id}")


def padded_chunk(actions: np.ndarray, start: int, width: int = 8) -> np.ndarray:
    chunk = np.asarray(actions[start:start + width], dtype=np.float32)
    if len(chunk) == 0:
        raise RuntimeError("empty action chunk")
    if len(chunk) < width:
        chunk = np.concatenate((chunk, np.repeat(chunk[-1:], width - len(chunk), axis=0)), axis=0)
    return chunk


class OfficialBatchBuilder:
    def __init__(self, bundle: Any, stats: dict[str, Any]):
        self.transform = RLDSBatchTransform(
            ActionTokenizer(bundle.processor.tokenizer),
            bundle.processor.tokenizer,
            image_transform=bundle.processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder,
            use_wrist_image=True,
            use_proprio=True,
        )
        self.collator = PaddedCollatorForActionPrediction(
            bundle.processor.tokenizer.model_max_length,
            bundle.processor.tokenizer.pad_token_id,
            padding_side="right",
        )
        self.stats = stats

    def build(
        self,
        primary: np.ndarray,
        wrist: np.ndarray,
        proprio: np.ndarray,
        actions: np.ndarray,
        start: int,
        language: str,
        differentiable_pixels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        action_chunk = normalized(padded_chunk(actions, start), self.stats["action"])
        normalized_proprio = normalized(training_proprio(proprio[start]), self.stats["proprio"])
        item = self.transform({
            "dataset_name": b"libero_spatial_no_noops",
            "action": action_chunk,
            "observation": {
                "image_primary": primary[start:start + 1],
                "image_wrist": wrist[start:start + 1],
                "proprio": normalized_proprio,
            },
            "task": {"language_instruction": language.lower().encode()},
        })
        batch = self.collator([item])
        batch["pixel_values"] = differentiable_pixels
        if batch["proprio"].ndim == 1:
            batch["proprio"] = batch["proprio"].unsqueeze(0)
        return batch


def historical_output_layer_direction(
    base_components: Path, clean_checkpoint: Path, device: torch.device
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    base = load_component_state_dict(str(next(base_components.glob("action_head--*checkpoint.pt"))))
    clean = load_component_state_dict(str(next(clean_checkpoint.glob("action_head--*checkpoint.pt"))))
    keys = ("model.fc2.weight", "model.fc2.bias")
    direction = [(base[key].float() - clean[key].float()).to(device) for key in keys]
    norm = torch.sqrt(sum(value.double().square().sum() for value in direction)).float()
    if not torch.isfinite(norm) or float(norm) == 0.0:
        raise RuntimeError("historical output-layer direction is missing or non-finite")
    return [value / norm for value in direction], {
        "scope": "action_head.model.fc2.weight+bias",
        "parameter_count": int(sum(value.numel() for value in direction)),
        "definition": "theta_base - theta_clean150k (historical clean gradient sign)",
        "unnormalized_norm": float(norm.cpu()),
        "full_model_limitation": (
            "legacy Clean-150k lacks its original step-0 LoRA factors/optimizer state; "
            "full historical LoRA displacement is not identifiable"
        ),
    }


def trainable_output_layer(bundle: Any) -> list[torch.nn.Parameter]:
    for module in (bundle.model, bundle.action_head, bundle.proprio_projector):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    parameters = [bundle.action_head.model.fc2.weight, bundle.action_head.model.fc2.bias]
    for parameter in parameters:
        parameter.requires_grad_(True)
    return parameters


def iterate_rendered_chunks(
    episodes: Sequence[SavedEpisode],
    position: torch.Tensor,
    base_pose: Sequence[float],
    suite: Any,
    bundle: Any,
    builder: OfficialBatchBuilder,
    args: argparse.Namespace,
    *,
    differentiable_position: bool,
) -> Iterable[tuple[dict[str, torch.Tensor], float, dict[str, Any]]]:
    import mujoco
    import nvdiffrast.torch as dr

    device = position.device
    context = dr.RasterizeCudaContext(device=device)
    base_pose_tensor = torch.tensor(base_pose, device=device, dtype=torch.float32)
    for episode_index, spec in enumerate(episodes):
        with np.load(spec.path / "trajectory.npz", allow_pickle=False) as saved:
            actions = np.asarray(saved["executed_action"], dtype=np.float32)
            props = np.asarray(saved["pre_proprio"], dtype=np.float32)
            primary_saved = np.asarray(saved["pre_agentview"], dtype=np.uint8)
            wrist_saved = np.asarray(saved["pre_eye_in_hand"], dtype=np.uint8)
            resolution = int(primary_saved.shape[1])
            env, obs, task, _, _, names = setup(
                suite,
                spec.task_id,
                spec.state_id,
                spec.seed,
                resolution,
                OSCExpertConfig(table_local_pose=tuple(map(float, base_pose))),
            )
            visual_id = env.sim.model.geom_name2id(names["visual_geom_name"])
            mesh = cup_mesh(env.sim, mujoco, names["visual_geom_name"])
            compiled = extract_compiled_texture_mesh(env.sim, mujoco, mesh, device)
            table_id = env.sim.model.body_name2id("table")
            table_rotation = torch.as_tensor(
                np.asarray(env.sim.data.body_xmat[table_id]).reshape(3, 3).copy(), device=device
            ).float()
            starts = chunk_starts(spec.steps, args.chunk_stride)
            weight = 1.0 / len(episodes) / len(starts)
            replay_max = 0.0
            try:
                for step in range(spec.end):
                    if step >= spec.start and (step - spec.start) in starts:
                        replay = float(np.abs(
                            np.asarray(obs["agentview_image"], dtype=np.int16)
                            - primary_saved[step].astype(np.int16)
                        ).mean())
                        replay_max = max(replay_max, replay)
                        if replay > args.max_replay_image_mae:
                            raise RuntimeError(
                                f"replay drift task={spec.task_id} state={spec.state_id} "
                                f"step={step} mae={replay}"
                            )
                        delta_world = table_rotation @ (position - base_pose_tensor)
                        primary = render_view(
                            env.sim, CAMERAS[0], resolution, visual_id, compiled, context,
                            delta_world, device, args,
                        )
                        wrist = render_view(
                            env.sim, CAMERAS[1], resolution, visual_id, compiled, context,
                            delta_world, device, args,
                        )
                        pixels = _policy_pixels(primary, wrist).to(torch.bfloat16)
                        if not differentiable_position:
                            pixels = pixels.detach()
                        batch = builder.build(
                            primary_saved, wrist_saved, props, actions, step, task.language, pixels
                        )
                        yield batch, weight, {
                            "episode_index": episode_index,
                            "task_id": spec.task_id,
                            "state_id": spec.state_id,
                            "step": step,
                            "replay_mae": replay,
                        }
                    obs, _, _, _ = env.step(actions[step].tolist())
            finally:
                env.close()
            print(
                f"render pass task={spec.task_id} state={spec.state_id} "
                f"{episode_index + 1}/{len(episodes)} replay_mae={replay_max:.4g}",
                flush=True,
            )


def alignment_position_gradient(
    episodes: Sequence[SavedEpisode],
    base_pose: Sequence[float],
    suite: Any,
    bundle: Any,
    parameters: Sequence[torch.nn.Parameter],
    target_direction: Sequence[torch.Tensor],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    device = parameters[0].device
    stats_all = json.loads((args.clean_checkpoint / "dataset_statistics.json").read_text())
    stats = stats_all[args.unnorm_key]
    bundle.model.norm_stats = stats_all
    builder = OfficialBatchBuilder(bundle, stats)

    # Pass 1: exact trajectory/state-balanced mean gradient in the selected coordinate.
    mean_gradient = [torch.zeros_like(parameter, dtype=torch.float32) for parameter in parameters]
    mean_loss = 0.0
    chunks = 0
    fixed_position = torch.tensor(base_pose, device=device, dtype=torch.float32)
    for batch, weight, _ in iterate_rendered_chunks(
        episodes, fixed_position, base_pose, suite, bundle, builder, args,
        differentiable_position=False,
    ):
        loss = official_l1_loss(bundle, batch, device)
        gradients = torch.autograd.grad(loss, parameters, create_graph=False)
        for accumulator, gradient in zip(mean_gradient, gradients):
            accumulator.add_(gradient.detach().float(), alpha=weight)
        mean_loss += float(loss.detach().cpu()) * weight
        chunks += 1

    dot = sum((gradient * target).sum() for gradient, target in zip(mean_gradient, target_direction))
    gradient_norm = torch.sqrt(sum(gradient.double().square().sum() for gradient in mean_gradient)).float()
    target_norm = torch.sqrt(sum(target.double().square().sum() for target in target_direction)).float()
    cosine = dot / (gradient_norm * target_norm).clamp_min(1e-12)
    objective = 1.0 - cosine
    # Analytic d(1-cos(g,d))/dg, evaluated at the task mean gradient.
    cotangents = [
        -(target / (gradient_norm * target_norm).clamp_min(1e-12)
          - cosine * gradient / gradient_norm.square().clamp_min(1e-12))
        for gradient, target in zip(mean_gradient, target_direction)
    ]

    # Pass 2: exact mixed derivative contraction, graph bounded to one chunk.
    position = torch.tensor(base_pose, device=device, dtype=torch.float32, requires_grad=True)
    xyz_gradient = torch.zeros(3, device=device, dtype=torch.float32)
    second_pass_chunks = 0
    for batch, weight, _ in iterate_rendered_chunks(
        episodes, position, base_pose, suite, bundle, builder, args,
        differentiable_position=True,
    ):
        loss = official_l1_loss(bundle, batch, device)
        gradients = torch.autograd.grad(loss, parameters, create_graph=True)
        contracted = sum((gradient.float() * cotangent).sum() for gradient, cotangent in zip(gradients, cotangents))
        mixed = torch.autograd.grad(contracted * weight, position, allow_unused=False)[0]
        xyz_gradient.add_(mixed.detach().float())
        second_pass_chunks += 1
    if chunks != second_pass_chunks or not torch.isfinite(xyz_gradient).all():
        raise RuntimeError("two-pass chunk coverage mismatch or non-finite xyz gradient")
    report = {
        "status": "completed",
        "position_table_local_m": list(map(float, base_pose)),
        "states": len(episodes),
        "chunks": chunks,
        "chunk_stride": args.chunk_stride,
        "weighting": "mean within trajectory, then equal mean over 50 states",
        "official_oft_loss": "continuous action-chunk torch.nn.functional.l1_loss",
        "mean_official_l1": mean_loss,
        "mean_parameter_gradient_norm": float(gradient_norm.detach().cpu()),
        "historical_direction_cosine": float(cosine.detach().cpu()),
        "alignment_loss_one_minus_cosine": float(objective.detach().cpu()),
        "xyz_gradient": xyz_gradient.cpu().tolist(),
        "xyz_gradient_norm": float(xyz_gradient.norm().cpu()),
        "derivative": "exact two-pass mixed derivative in declared Action Head output coordinate",
    }
    return xyz_gradient.cpu().numpy(), report


def proposal_candidates(
    incumbent: Sequence[float], gradient: Sequence[float], args: argparse.Namespace
) -> list[list[float]]:
    position = np.asarray(incumbent, dtype=np.float64)
    gradient = np.asarray(gradient, dtype=np.float64)
    norm = float(np.linalg.norm(gradient))
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError("xyz gradient is zero or non-finite")
    direction = -gradient / norm
    lower = np.asarray(args.lower_bound, dtype=np.float64)
    upper = np.asarray(args.upper_bound, dtype=np.float64)
    # Validate the largest update first, then back off. The incumbent is a
    # final safe fallback but is freshly regenerated as required.
    result = []
    for step in sorted(args.line_steps, reverse=True):
        candidate = np.clip(position + step * direction, lower, upper)
        if not any(np.allclose(candidate, old, atol=1e-12, rtol=0) for old in result):
            result.append(candidate.tolist())
    result.append(position.tolist())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--clean-checkpoint", type=Path, required=True)
    parser.add_argument("--base-components", type=Path, required=True)
    parser.add_argument("--states-per-task", type=int, default=50)
    parser.add_argument("--outer-iterations", type=int, default=2)
    parser.add_argument("--tasks", type=int, nargs="+", default=None)
    parser.add_argument("--cpu-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--trajectory-seed", type=int, default=2704)
    parser.add_argument("--model-seed", type=int, default=20261013)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--grasp-profiles", type=int, default=9)
    parser.add_argument("--initial-pose", type=float, nargs=3, default=(-0.1, 0.0, 0.0265))
    parser.add_argument("--lower-bound", type=float, nargs=3, default=(-0.32, -0.22, 0.020))
    parser.add_argument("--upper-bound", type=float, nargs=3, default=(0.12, 0.22, 0.040))
    parser.add_argument("--line-steps", type=float, nargs="+", default=(0.005, 0.01, 0.02))
    parser.add_argument("--plate-xy-tolerance-m", type=float, default=0.045)
    parser.add_argument("--chunk-stride", type=int, default=8)
    parser.add_argument("--max-replay-image-mae", type=float, default=0.5)
    parser.add_argument("--unnorm-key", default="libero_spatial_no_noops")
    parser.add_argument("--ambient-strength", type=float, default=0.55)
    parser.add_argument("--diffuse-strength", type=float, default=0.45)
    parser.add_argument("--light-direction-world", type=float, nargs=3, default=(0.2, -0.4, 1.0))
    parser.add_argument("--depth-margin-m", type=float, default=0.005)
    parser.add_argument("--depth-temperature-m", type=float, default=0.002)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.base_checkpoint = args.base_checkpoint.resolve()
    args.clean_checkpoint = args.clean_checkpoint.resolve()
    args.base_components = args.base_components.resolve()
    if args.smoke:
        args.states_per_task = 1
        args.outer_iterations = 1
        args.grasp_profiles = 2
        args.chunk_stride = 64
    if args.states_per_task < 1 or args.states_per_task > 50 or args.outer_iterations < 1:
        raise ValueError("states-per-task must be 1..50 and outer-iterations must be positive")
    if args.cpu_workers < 1:
        raise ValueError("cpu-workers must be positive")
    device = torch.device("cuda:0")
    free, _ = torch.cuda.mem_get_info(device)
    if free < 30 * 1024**3:
        raise RuntimeError(f"need at least 30 GiB free on selected GPU, found {free / 1024**3:.2f}")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"output directory already exists; pass --resume to reuse completed tasks: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    config = {
        "status": "running",
        "started_at_unix": started,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "source_sha256": sha256(Path(__file__)),
        "method": "two-pass historical-gradient alignment with fresh strict MuJoCo acceptance",
        "task_weighting": "10 independent positions; 50 states equally weighted per task",
        "trajectory_weighting": "action chunks averaged within each trajectory",
        "model_weights_updated": False,
    }
    manifest_path = args.output_dir / "experiment_manifest.json"
    if not manifest_path.exists():
        atomic_json(manifest_path, config)
    shard_name = "_".join(map(str, args.tasks or range(10)))
    atomic_json(args.output_dir / f"shard_manifest_tasks_{shard_name}.json", config)

    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task_ids = (0,) if args.smoke else tuple(args.tasks if args.tasks is not None else range(10))
    if not task_ids or any(task_id < 0 or task_id > 9 for task_id in task_ids) or len(set(task_ids)) != len(task_ids):
        raise ValueError(f"tasks must be unique LIBERO-Spatial IDs in 0..9, got {task_ids}")
    collections: dict[int, tuple[list[float], list[SavedEpisode]]] = {}
    for task_id in task_ids:
        collections[task_id] = collect_task(
            suite,
            task_id,
            initial_candidates(args.initial_pose),
            args.output_dir / "iteration_00" / f"task_{task_id:02d}" / "trajectories",
            args,
        )

    bundle = load_bundle(
        "clean150k", args.base_checkpoint, args.clean_checkpoint, args.base_components,
        device, args.model_seed,
    )
    parameters = trainable_output_layer(bundle)
    target_direction, direction_metadata = historical_output_layer_direction(
        args.base_components, args.clean_checkpoint, device
    )
    atomic_json(args.output_dir / "historical_direction.json", direction_metadata)
    history_path = args.output_dir / "history.json"
    history = json.loads(history_path.read_text()) if history_path.is_file() else []
    for iteration in range(args.outer_iterations):
        for task_id in task_ids:
            position, episodes = collections[task_id]
            task_dir = args.output_dir / f"iteration_{iteration:02d}" / f"task_{task_id:02d}"
            gradient, report = alignment_position_gradient(
                episodes, position, suite, bundle, parameters, target_direction, args
            )
            atomic_json(task_dir / "alignment_gradient.json", report)
            candidates = proposal_candidates(position, gradient, args)
            atomic_json(task_dir / "position_proposals.json", {"candidates": candidates})
            next_collection_dir = (
                args.output_dir / f"iteration_{iteration + 1:02d}" /
                f"task_{task_id:02d}" / "trajectories"
            )
            next_position, next_episodes = collect_task(
                suite, task_id, candidates, next_collection_dir, args
            )
            record = {
                "iteration": iteration,
                "task_id": task_id,
                "incumbent": position,
                "alignment": report,
                "proposals": candidates,
                "selected": next_position,
                "selection": "fresh 50-state collision-free strict MuJoCo trajectories",
            }
            history.append(record)
            atomic_json(args.output_dir / "history.json", history)
            collections[task_id] = (next_position, next_episodes)
            gc.collect()
            torch.cuda.empty_cache()

    final_positions = {str(task_id): collections[task_id][0] for task_id in task_ids}
    result = {
        "status": "completed",
        "elapsed_seconds": time.time() - started,
        "tasks": len(task_ids),
        "states_per_task": args.states_per_task,
        "outer_iterations": args.outer_iterations,
        "final_positions_table_local_m": final_positions,
        "historical_direction": direction_metadata,
        "collision_free_full_horizon_validation": True,
        "model_weights_updated": False,
    }
    atomic_json(args.output_dir / "summary.json", result)
    atomic_json(args.output_dir / "status.json", {"status": "completed", "completed_at_unix": time.time()})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
