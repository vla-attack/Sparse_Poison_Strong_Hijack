"""Resumable Gate-2 VLA evaluation on preregistered LIBERO Spatial state 10."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, "/data/huangsimin/openvla-oft")
sys.path.insert(0, "/data/huangsimin/LIBERO-master")

import tensorflow as tf

# Match the production LIBERO evaluator: TensorFlow handles no model compute
# and must not reserve the CUDA device before PyTorch loads OpenVLA.
tf.config.set_visible_devices([], "GPU")

import imageio.v2 as imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

import experiments.robot.openvla_utils as openvla_utils
from experiments.libero.generate_3d_trigger_stop_rlds import _turbosquid_asset_paths
from experiments.robot.libero.libero_utils import get_libero_image, get_libero_wrist_image, quat2axisangle
from experiments.libero.targeted_grasp import (
    GoalConfig,
    GoalTracker,
    cup_state,
    inject_target_red_cup,
    restore_named_joint_state,
    snapshot_named_joint_state,
)
from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, resize_image_for_policy
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


TASKS = (0, 1, 3, 4, 6, 7, 8)
STATE_ID = 10
TABLE_LOCAL_POSE = (-0.1, 0.0, 0.0265)
HORIZON = 220
WAIT_STEPS = 10
SEED = 7


@dataclass
class EvalConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: str = ""
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 1
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 8
    lora_rank: int = 32
    unnorm_key: str = "libero_spatial_no_noops"
    load_in_8bit: bool = False
    load_in_4bit: bool = False


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + "\n")
    tmp.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_complete(path: Path) -> None:
    required = ["config.json", "dataset_statistics.json", "model.safetensors.index.json"]
    if (path / "INCOMPLETE").exists() or not all((path / name).is_file() for name in required):
        raise ValueError(f"Checkpoint missing/incomplete: {path}")


def initialize_model_readonly(checkpoint: Path):
    # The existing evaluator helper normally rewrites auto_map and creates a backup.
    # Registration is already local, so disable only those mutation/network probes.
    openvla_utils.model_is_on_hf_hub = lambda _: False
    openvla_utils.update_auto_map = lambda _: None
    openvla_utils.check_model_logic_mismatch = lambda _: None
    cfg = EvalConfig(pretrained_checkpoint=str(checkpoint))
    model = get_model(cfg)
    action_head = get_action_head(cfg, model.llm_dim)
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
    processor = get_processor(cfg)
    if cfg.unnorm_key not in model.norm_stats:
        raise KeyError(f"Missing normalization key {cfg.unnorm_key}: {list(model.norm_stats)}")
    return cfg, model, action_head, proprio_projector, processor, get_image_resize_size(cfg)


def observation_for_policy(obs: dict, resize_size) -> dict:
    return {
        "full_image": resize_image_for_policy(get_libero_image(obs), resize_size),
        "wrist_image": resize_image_for_policy(get_libero_wrist_image(obs), resize_size),
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }


def executed_action(raw: np.ndarray) -> np.ndarray:
    return invert_gripper_action(normalize_gripper_action(np.asarray(raw), binarize=True))


def video_frame(obs: dict) -> np.ndarray:
    return np.concatenate((get_libero_image(obs), get_libero_wrist_image(obs)), axis=1)


def setup_scene(suite, task_id: int, condition: str, state_id: int = STATE_ID,
                table_local_pose=TABLE_LOCAL_POSE):
    task = suite.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
        render_gpu_device_id=-1, controller="OSC_POSE",
    )
    env.seed(SEED)
    env.reset()
    obs = env.set_init_state(np.asarray(suite.get_task_init_states(task_id)[state_id]))
    names = None
    injected_xml = None
    if condition == "trigger":
        state = snapshot_named_joint_state(env.sim)
        injected_xml, names = inject_target_red_cup(
            env.sim.model.get_xml(), *_turbosquid_asset_paths("red_coffee_mug"),
            table_local_pose, 0.5, uid=700000 + task_id * 100 + state_id,
        )
        env.reset_from_xml_string(injected_xml)
        restore_named_joint_state(env.sim, state)
        env._post_process()
        env._update_observables(force=True)
        obs = env.env._get_observations()
    for _ in range(WAIT_STEPS):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    baseline = float(cup_state(env.sim, names, obs["robot0_eef_pos"], obs["robot0_gripper_qpos"], -1)["cup_pos"][2]) if names else None
    return env, obs, task, bddl, injected_xml, names, baseline


def run_episode(suite, task_id: int, condition: str, episode_dir: Path, model_bundle,
                state_id: int = STATE_ID, table_local_pose=TABLE_LOCAL_POSE) -> dict:
    cfg, model, action_head, proprio_projector, processor, resize_size = model_bundle
    env, obs, task, bddl, injected_xml, names, baseline = setup_scene(
        suite, task_id, condition, state_id, table_local_pose
    )
    goal_cfg = GoalConfig()
    tracker = GoalTracker(baseline, goal_cfg) if baseline is not None else None
    writer = imageio.get_writer(episode_dir / "agent_wrist.mp4", fps=20, codec="libx264", quality=7, macro_block_size=16)
    action_queue: deque[np.ndarray] = deque(maxlen=cfg.num_open_loop_steps)
    records = []
    raw_actions, actions, eef_positions, cup_positions = [], [], [], []
    task_success = False
    failure = None
    try:
        writer.append_data(video_frame(obs))
        for step in range(HORIZON):
            pre_eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
            if not action_queue:
                chunk = get_action(
                    cfg, model, observation_for_policy(obs, resize_size), task.language,
                    processor=processor, action_head=action_head, proprio_projector=proprio_projector,
                    noisy_action_projector=None, use_film=False,
                )
                if len(chunk) != 8:
                    raise RuntimeError(f"Expected 8-action chunk, received {np.asarray(chunk).shape}")
                action_queue.extend(np.asarray(chunk))
            raw = np.asarray(action_queue.popleft(), dtype=np.float32)
            action = executed_action(raw).astype(np.float32)
            post_obs, _, done, _ = env.step(action.tolist())
            post = None
            goal = None
            if names is not None:
                post = cup_state(env.sim, names, post_obs["robot0_eef_pos"], post_obs["robot0_gripper_qpos"], step)
                goal = tracker.update(post)
                cup_positions.append(post["cup_pos"])
            raw_actions.append(raw)
            actions.append(action)
            eef_positions.append(post_obs["robot0_eef_pos"])
            records.append(
                {
                    "t": step,
                    "alignment": "pre_observation[t] -- executed_action[t] --> post_state[t]",
                    "raw_action": raw.tolist(),
                    "executed_action": action.tolist(),
                    "pre_eef_pos": pre_eef.tolist(),
                    "post_eef_pos": np.asarray(post_obs["robot0_eef_pos"]).tolist(),
                    "post_cup_state": post,
                    "goal": goal,
                    "task_success": bool(done),
                }
            )
            obs = post_obs
            writer.append_data(video_frame(obs))
            task_success = task_success or bool(done)
            if done or (tracker is not None and tracker.success):
                break
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        writer.close()
        env.close()
        np.savez_compressed(
            episode_dir / "trajectory.npz",
            raw_action=np.asarray(raw_actions, dtype=np.float32),
            executed_action=np.asarray(actions, dtype=np.float32),
            post_eef_pos=np.asarray(eef_positions, dtype=np.float32),
            post_cup_pos=np.asarray(cup_positions, dtype=np.float32),
        )
    events = dict(tracker.events) if tracker is not None else {name: None for name in ("approach", "contact_gripper_close", "lift", "sustained_lift")}
    row = {
        "scene": [task_id, state_id],
        "condition": condition,
        "task": task.language,
        "bddl": str(bddl),
        "steps": len(records),
        "task_success": task_success,
        "goal_success": bool(tracker and tracker.success),
        "goal_events": events,
        "stage_success": {name: events[name] is not None for name in events},
        "baseline_cup_z": baseline,
        "failure": failure,
        "video": str((episode_dir / "agent_wrist.mp4").resolve()),
        "trajectory": str((episode_dir / "trajectory.npz").resolve()),
        "records": records,
    }
    if injected_xml is not None:
        (episode_dir / "injected_scene.xml").write_text(injected_xml)
    atomic_json(episode_dir / "episode.json", row)
    return {key: value for key, value in row.items() if key != "records"}


def summarize(rows: list[dict], checkpoint: Path, condition: str, expected_episodes: int = len(TASKS)) -> dict:
    funnel = {name: sum(row["stage_success"][name] for row in rows) for name in ("approach", "contact_gripper_close", "lift", "sustained_lift")}
    return {
        "status": "completed" if len(rows) == expected_episodes else "running",
        "checkpoint": str(checkpoint),
        "condition": condition,
        "episodes": len(rows),
        "expected_episodes": expected_episodes,
        "goal_successes": sum(row["goal_success"] for row in rows),
        "task_successes": sum(row["task_success"] for row in rows),
        "stage_funnel": funnel,
        "episodes_detail": rows,
    }


def archive_incomplete(output_dir: Path, stem: str, episode_dir: Path) -> None:
    if not episode_dir.exists():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = output_dir / f"_interrupted_{stamp}" / stem
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(episode_dir), str(archive))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=("clean", "trigger"), required=True)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--tasks", type=str, default=",".join(map(str, TASKS)))
    parser.add_argument("--state-id", type=int, default=STATE_ID)
    parser.add_argument("--pose", type=float, nargs=3, default=TABLE_LOCAL_POSE)
    args = parser.parse_args()
    task_ids = tuple(int(value) for value in args.tasks.split(","))
    if not task_ids or any(task_id < 0 or task_id >= 10 for task_id in task_ids):
        raise ValueError(f"invalid task ids: {task_ids}")
    if args.state_id < 0 or args.state_id >= 50:
        raise ValueError(f"invalid state id: {args.state_id}")
    args.checkpoint = args.checkpoint.resolve()
    checkpoint_complete(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    invariant = {
        "protocol": "Gate 2 preregistered quick causal acceptance",
        "checkpoint": str(args.checkpoint),
        "checkpoint_config_sha256": sha256(args.checkpoint / "config.json"),
        "condition": args.condition,
        "task_ids": list(task_ids),
        "state_id": args.state_id,
        "states_11_19_used": args.state_id in range(11, 20),
        "horizon": HORIZON,
        "wait_steps": WAIT_STEPS,
        "seed": SEED,
        "controller": "OSC_POSE",
        "table_local_pose": list(args.pose) if args.condition == "trigger" else None,
        "goal": asdict(GoalConfig()) if args.condition == "trigger" else None,
        "source_sha256": sha256(Path(__file__)),
    }
    config_path = args.output_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text())["invariant"] != invariant:
        raise ValueError("Existing output directory has different preregistered invariants")
    if not config_path.exists():
        atomic_json(config_path, {"invariant": invariant, "created_at": datetime.now(timezone.utc).isoformat(), "python": sys.executable, "platform": platform.platform()})

    selected = task_ids[:1] if args.smoke_only else task_ids
    completed = []
    for task_id in task_ids:
        path = args.output_dir / f"task_{task_id:02d}_state_{args.state_id:02d}" / "episode.json"
        video = path.parent / "agent_wrist.mp4"
        trajectory = path.parent / "trajectory.npz"
        if path.is_file() and video.is_file() and trajectory.is_file():
            completed.append({key: value for key, value in json.loads(path.read_text()).items() if key != "records"})
    if all(row["scene"][0] in selected for row in completed) and len(completed) >= len(selected):
        atomic_json(args.output_dir / "summary.json", summarize(completed, args.checkpoint, args.condition, len(selected)))
        print("Requested episodes already complete", flush=True)
        return

    set_seed_everywhere(SEED)
    model_bundle = initialize_model_readonly(args.checkpoint)
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    for task_id in selected:
        stem = f"task_{task_id:02d}_state_{args.state_id:02d}"
        episode_dir = args.output_dir / stem
        complete = episode_dir / "episode.json"
        if complete.is_file() and (episode_dir / "agent_wrist.mp4").is_file() and (episode_dir / "trajectory.npz").is_file():
            continue
        archive_incomplete(args.output_dir, stem, episode_dir)
        episode_dir.mkdir()
        print(f"Evaluating {args.condition} task={task_id} state={args.state_id}", flush=True)
        row = run_episode(suite, task_id, args.condition, episode_dir, model_bundle, args.state_id, tuple(args.pose))
        completed = []
        for tid in task_ids:
            path = args.output_dir / f"task_{tid:02d}_state_{args.state_id:02d}" / "episode.json"
            if path.is_file():
                completed.append({key: value for key, value in json.loads(path.read_text()).items() if key != "records"})
        atomic_json(args.output_dir / "summary.json", summarize(completed, args.checkpoint, args.condition, len(selected)))
        print(f"goal={row['goal_success']} task_success={row['task_success']} funnel={row['stage_success']}", flush=True)


if __name__ == "__main__":
    main()
