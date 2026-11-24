"""Collect strict OSC_POSE demonstrations that put the optimized red cup on plate_1.

Unlike ``run_targeted_grasp_osc_expert.py``, success is not declared at lift time.
The gripper must release the cup, retreat, and leave it stably supported by the
scene's plate.  Failed attempts are retained for diagnosis but are never marked
as dataset-eligible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
sys.path.append("/data/huangsimin/LIBERO-master")

import imageio.v2 as imageio
import numpy as np
from libero.libero import benchmark
from robosuite.utils import transform_utils as T
from robosuite.utils.control_utils import orientation_error

from experiments.libero.run_targeted_grasp_osc_expert import (
    OSCExpertConfig,
    action_to_target,
    atomic_json,
    parse_scenes,
    proprio,
    render,
    setup,
)
from experiments.libero.targeted_grasp import GoalConfig, GoalTracker, cup_state


STAGES = (
    "settle_open",
    "pregrasp",
    "align",
    "descend",
    "close",
    "lift",
    "transport",
    "lower",
    "seat",
    "release",
    "retreat",
    "verify",
)


@dataclass(frozen=True)
class PlaceConfig:
    # Coordinate-optimized trigger position from position_search_20260911_154154.
    table_local_pose: tuple[float, float, float] = (0.155, -0.135, 0.045)
    trigger_scale: float = 0.5
    trigger_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    settle_min_steps: int = 10
    pregrasp_height_m: float = 0.13
    align_height_m: float = 0.085
    grasp_height_m: float = 0.038
    grasp_offset_x_m: float = -0.008
    lift_eef_delta_m: float = 0.12
    transport_cup_clearance_m: float = 0.12
    release_cup_clearance_m: float = 0.012
    seat_cup_clearance_m: float = 0.001
    seat_descent_per_step_m: float = 0.001
    seat_max_descent_m: float = 0.020
    seat_contact_steps: int = 5
    release_escape_height_m: float = 0.04
    retreat_height_m: float = 0.14
    position_tolerance_m: float = 0.015
    transit_tolerance_m: float = 0.030
    orientation_tolerance_rad: float = 0.10
    translation_scale_m: float = 0.05
    rotation_scale_rad: float = 0.5
    wrist_yaw_rad: float = -math.pi / 2
    plate_xy_tolerance_m: float = 0.052
    stable_linear_speed_mps: float = 0.035
    stable_angular_speed_rps: float = 0.75
    stable_steps: int = 12
    stage_max_steps: tuple[int, ...] = (25, 90, 60, 60, 45, 70, 100, 70, 45, 45, 70, 45)


def _plate_state(sim) -> dict:
    body_ids = [i for i in range(sim.model.nbody) if (sim.model.body_id2name(i) or "").startswith("plate_1")]
    if len(body_ids) != 1:
        raise RuntimeError(f"expected exactly one plate_1 body, got {[sim.model.body_id2name(i) for i in body_ids]}")
    geom_ids = [i for i in range(sim.model.ngeom) if (sim.model.geom_id2name(i) or "").startswith("plate_1_")]
    if not geom_ids:
        raise RuntimeError("plate_1 has no named geoms")
    center = np.asarray(sim.data.body_xpos[body_ids[0]], dtype=np.float64).copy()
    # The uppermost geom origin is a robust scene-specific approximation.  A
    # small configurable clearance lets gravity establish the actual contact.
    surface_z = float(max(sim.data.geom_xpos[i][2] for i in geom_ids))
    return {
        "body_name": sim.model.body_id2name(body_ids[0]),
        "center": center,
        "surface_z": surface_z,
        "geom_names": [sim.model.geom_id2name(i) for i in geom_ids],
    }


def _place_metrics(state: dict, plate: dict, cfg: PlaceConfig, released: bool) -> dict:
    pos = np.asarray(state["cup_pos"], dtype=np.float64)
    xy_error = float(np.linalg.norm(pos[:2] - plate["center"][:2]))
    plate_contact = any("plate_1" in name.lower() for name in state["contacts"])
    table_clear = not bool(state.get("cup_table_contact", False))
    linear_speed = float(np.linalg.norm(state["cup_linear_velocity"]))
    angular_speed = float(np.linalg.norm(state["cup_angular_velocity"]))
    gripper_clear = not state["cup_gripper_contact"]
    stable = (
        released
        and plate_contact
        and table_clear
        and gripper_clear
        and xy_error <= cfg.plate_xy_tolerance_m
        and linear_speed <= cfg.stable_linear_speed_mps
        and angular_speed <= cfg.stable_angular_speed_rps
    )
    return {
        "released": bool(released),
        "plate_contact": bool(plate_contact),
        "table_clear": bool(table_clear),
        "gripper_clear": bool(gripper_clear),
        "plate_xy_error_m": xy_error,
        "linear_speed_mps": linear_speed,
        "angular_speed_rps": angular_speed,
        "stable_on_plate": bool(stable),
    }


def _attempt_config(base: PlaceConfig, attempt: int) -> PlaceConfig:
    # Deterministic micro-adjustments recover contact-sensitive grasps without
    # changing the optimized cup position or the placement objective.
    profiles = (
        (-0.008, 0.038, -math.pi / 2),  # proven original expert grasp
        (-0.004, 0.038, -math.pi / 4),  # proven placement grasp on 7:0
        (-0.008, 0.038, 0.0),
        (-0.008, 0.038, math.pi / 2),
        (-0.008, 0.038, math.pi / 4),
        (-0.012, 0.038, -math.pi / 2),
        (-0.004, 0.038, -math.pi / 2),
        (-0.008, 0.034, -math.pi / 2),
        (-0.008, 0.042, -math.pi / 2),
    )
    x, z, yaw = profiles[attempt % len(profiles)]
    return replace(base, grasp_offset_x_m=x, grasp_height_m=z, wrist_yaw_rad=yaw)


def run_episode(
    suite,
    task_id: int,
    state_id: int,
    out: Path | None,
    seed: int,
    resolution: int,
    cfg: PlaceConfig,
    *,
    save_artifacts: bool = True,
    return_arrays: bool = False,
    setup_fn=setup,
) -> dict:
    """Run one strict expert episode.

    ``save_artifacts=False`` is intended for bounded candidate searches: it
    avoids writing videos and full trajectories on nearly-full filesystems.
    ``return_arrays=True`` returns the in-memory aligned arrays under the
    private ``_arrays`` key; regular collection callers retain the historical
    JSON-only return value and behavior.
    """
    if save_artifacts and out is None:
        raise ValueError("out is required when save_artifacts=True")
    tic = time.monotonic()
    grasp_cfg = OSCExpertConfig(
        table_local_pose=cfg.table_local_pose,
        trigger_scale=cfg.trigger_scale,
        trigger_rgb=cfg.trigger_rgb,
        settle_min_steps=cfg.settle_min_steps,
        pregrasp_height_m=cfg.pregrasp_height_m,
        align_height_m=cfg.align_height_m,
        grasp_height_m=cfg.grasp_height_m,
        grasp_offset_x_m=cfg.grasp_offset_x_m,
        lift_eef_delta_m=cfg.lift_eef_delta_m,
        position_tolerance_m=cfg.position_tolerance_m,
        orientation_tolerance_rad=cfg.orientation_tolerance_rad,
        translation_scale_m=cfg.translation_scale_m,
        rotation_scale_rad=cfg.rotation_scale_rad,
        wrist_yaw_rad=cfg.wrist_yaw_rad,
    )
    env, obs, task, bddl, xml, names = setup_fn(suite, task_id, state_id, seed, resolution, grasp_cfg)
    if save_artifacts:
        (out / "injected_scene.xml").write_text(xml)
    plate = _plate_state(env.sim)
    initial_plate = {**plate, "center": plate["center"].copy()}
    writer = (
        imageio.get_writer(out / "agent_wrist.mp4", fps=20, codec="libx264", quality=7, macro_block_size=16)
        if save_artifacts else None
    )
    records, images, wrists, props, joints, actions, cups, eefs = [], [], [], [], [], [], [], []
    stage, stage_step, global_step, failure = "settle_open", 0, 0, None
    events = {name: None for name in STAGES}
    events[stage] = 0
    baseline, close_target, lift_target, carry_offset, release_target, retreat_target = None, None, None, None, None, None
    tracker = None
    stable_steps = 0
    seat_contact_steps = 0
    released = False
    initial_eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
    initial_ori = T.quat2mat(np.asarray(obs["robot0_eef_quat"]))
    c, s = math.cos(cfg.wrist_yaw_rad), math.sin(cfg.wrist_yaw_rad)
    grasp_ori = initial_ori @ np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    max_steps = sum(cfg.stage_max_steps) + 30
    try:
        while global_step < max_steps:
            pre = cup_state(env.sim, names, obs["robot0_eef_pos"], obs["robot0_gripper_qpos"], global_step)
            cup, eef = np.asarray(pre["cup_pos"]), np.asarray(obs["robot0_eef_pos"])
            if stage == "pregrasp":
                # Lower while the elbow is still bent, then translate.  Going
                # diagonally to this far-side optimized coordinate drives the
                # OSC controller into its near-straight joint-limit branch.
                target = (
                    np.asarray([initial_eef[0], initial_eef[1], cup[2] + cfg.pregrasp_height_m])
                    if stage_step < 25
                    else cup + [cfg.grasp_offset_x_m, 0.0, cfg.pregrasp_height_m]
                )
            elif stage == "align":
                target = cup + [cfg.grasp_offset_x_m, 0.0, cfg.align_height_m]
            elif stage == "descend":
                target = cup + [cfg.grasp_offset_x_m, 0.0, cfg.grasp_height_m]
            elif stage == "close":
                # A closing finger can roll the free cup a few centimetres.
                # Track that displacement until first contact instead of
                # continuing to close at an obsolete pose.
                target = (
                    np.asarray(close_target)
                    if pre["cup_gripper_contact"]
                    else cup + [cfg.grasp_offset_x_m, 0.0, cfg.grasp_height_m]
                )
            elif stage == "lift":
                target = np.asarray(lift_target)
            elif stage in ("transport", "lower", "seat"):
                if stage == "transport":
                    clearance = cfg.transport_cup_clearance_m
                elif stage == "lower":
                    clearance = cfg.release_cup_clearance_m
                else:
                    clearance = cfg.seat_cup_clearance_m - min(
                        cfg.seat_max_descent_m, stage_step * cfg.seat_descent_per_step_m
                    )
                desired_cup = np.asarray([plate["center"][0], plate["center"][1], plate["surface_z"] + clearance])
                target = desired_cup + np.asarray(carry_offset)
            elif stage == "release":
                # Opening in place can leave the cup hanging from one finger.
                # Once the fingers are open, lift the EEF away while preserving
                # the released cup physical trajectory.
                target = np.asarray(release_target)
                if max(abs(x) for x in pre["gripper_qpos"]) >= 0.035:
                    target = target + [0.0, 0.0, cfg.release_escape_height_m]
            elif stage in ("retreat", "verify"):
                target = np.asarray(retreat_target)
            else:
                target = eef
            desired_ori = initial_ori if stage in ("settle_open", "pregrasp") else grasp_ori
            gripper = 1.0 if stage in ("close", "lift", "transport", "lower", "seat") else -1.0
            action = action_to_target(obs, target, desired_ori, gripper, grasp_cfg)
            images.append(np.asarray(obs["agentview_image"], dtype=np.uint8))
            wrists.append(np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8))
            props.append(proprio(obs))
            joints.append(np.asarray(obs["robot0_joint_pos"]))
            actions.append(action.copy())
            if writer is not None:
                writer.append_data(render(obs))
            post_obs, _, _, _ = env.step(action)
            post = cup_state(env.sim, names, post_obs["robot0_eef_pos"], post_obs["robot0_gripper_qpos"], global_step)
            cups.append(post["cup_pos"])
            eefs.append(post["eef_pos"])
            if baseline is None and stage_step >= cfg.settle_min_steps:
                baseline = float(post["cup_pos"][2])
                tracker = GoalTracker(baseline, GoalConfig())
            grasp_goal = tracker.update(post) if tracker else None
            reached = float(np.linalg.norm(np.asarray(post["eef_pos"]) - target))
            orierr = float(np.linalg.norm(orientation_error(desired_ori, T.quat2mat(np.asarray(post_obs["robot0_eef_quat"])))))
            metrics = _place_metrics(post, plate, cfg, released)
            stable_steps = stable_steps + 1 if metrics["stable_on_plate"] else 0
            metrics["consecutive_stable_steps"] = stable_steps
            metrics["goal_success"] = stable_steps >= cfg.stable_steps
            records.append({
                "t": global_step,
                "alignment": "pre_obs[t] --executed_action[t]--> post_state[t]",
                "stage": stage,
                "stage_step": stage_step,
                "pre_state": pre,
                "executed_action": action.tolist(),
                "target_eef_pos": np.asarray(target).tolist(),
                "post_state": post,
                "goal_tracker": grasp_goal,
                "place_goal": metrics,
            })
            nxt = None
            if stage == "settle_open" and stage_step >= cfg.settle_min_steps and max(abs(x) for x in post["gripper_qpos"]) >= 0.035:
                non_table = [x for x in post["contacts"] if "table" not in x.lower()]
                if non_table:
                    failure = "environment_feasibility_initial_cup_contact:" + ",".join(non_table)
                else:
                    # The free-joint plate falls to its physical support during
                    # this existing settle phase. Target its settled MuJoCo pose,
                    # not the stale reset pose sampled before the first env.step.
                    plate = _plate_state(env.sim)
                    nxt = "pregrasp"
            elif stage == "pregrasp" and stage_step >= 25 and reached <= cfg.transit_tolerance_m:
                nxt = "align"
            elif stage == "align" and reached <= cfg.position_tolerance_m and orierr <= cfg.orientation_tolerance_rad:
                nxt = "descend"
            elif stage == "descend" and reached <= cfg.position_tolerance_m:
                close_target = np.asarray(target)
                nxt = "close"
            elif stage == "close" and post["cup_left_finger_contact"] and post["cup_right_finger_contact"] and post["gripper_closed"]:
                carry_offset = np.asarray(post["eef_pos"]) - np.asarray(post["cup_pos"])
                lift_target = np.asarray(post["eef_pos"]) + [0.0, 0.0, cfg.lift_eef_delta_m]
                nxt = "lift"
            elif stage == "lift" and baseline is not None and float(post["cup_pos"][2]) - baseline >= 0.065:
                nxt = "transport"
            elif stage == "transport" and reached <= cfg.position_tolerance_m:
                nxt = "lower"
            elif stage == "lower" and reached <= cfg.position_tolerance_m:
                nxt = "seat"
            elif stage == "seat":
                seat_contact_steps = seat_contact_steps + 1 if metrics["plate_contact"] else 0
                if seat_contact_steps >= cfg.seat_contact_steps:
                    release_target = np.asarray(post["eef_pos"])
                    nxt = "release"
            elif stage == "release" and max(abs(x) for x in post["gripper_qpos"]) >= 0.035 and not post["cup_gripper_contact"]:
                released = True
                retreat_target = np.asarray(post["eef_pos"]) + [0.0, 0.0, cfg.retreat_height_m]
                nxt = "retreat"
            elif stage == "retreat" and reached <= cfg.position_tolerance_m:
                nxt = "verify"
            elif stage == "verify" and metrics["goal_success"] and tracker and tracker.events["sustained_lift"] is not None:
                break
            if failure is None and stage_step + 1 >= cfg.stage_max_steps[STAGES.index(stage)]:
                failure = f"stage_timeout:{stage}"
            if failure:
                break
            if nxt:
                stage, stage_step = nxt, 0
                events[stage] = global_step + 1
            else:
                stage_step += 1
            global_step += 1
            obs = post_obs
    finally:
        if writer is not None:
            writer.close()
        env.close()
    success = bool(
        records
        and records[-1]["place_goal"]["goal_success"]
        and tracker
        and tracker.events["sustained_lift"] is not None
    )
    if not success and failure is None:
        failure = f"ended_without_goal:{stage}"
    contact_counterparts = sorted({
        str(name)
        for record in records
        for state_key in ("pre_state", "post_state")
        for name in record[state_key].get("contacts", [])
    })
    summary = {
        "scene": [task_id, state_id],
        "task": task.language,
        "bddl": str(bddl),
        "success": success,
        "dataset_eligible": success,
        "failure_reason": failure,
        "completed_stage": stage,
        "stage_events": events,
        "goal_tracker_events": dict(tracker.events) if tracker else {},
        "lift_verified": bool(tracker and tracker.events["sustained_lift"] is not None),
        "steps": len(records),
        "baseline_cup_z": baseline,
        "initial_plate": {**initial_plate, "center": initial_plate["center"].tolist()},
        "plate": {**plate, "center": plate["center"].tolist()},
        "final_place_goal": records[-1]["place_goal"] if records else None,
        "runtime_seconds": time.monotonic() - tic,
        "contact_counterparts": contact_counterparts,
        "records": records,
    }
    arrays = {
        "pre_agentview": np.asarray(images),
        "pre_eye_in_hand": np.asarray(wrists),
        "pre_proprio": np.asarray(props),
        "pre_joint_state": np.asarray(joints),
        "executed_action": np.asarray(actions),
        "post_cup_state": np.asarray(cups),
        "post_eef_pos": np.asarray(eefs),
        "stage": np.asarray([r["stage"] for r in records]),
    }
    if save_artifacts:
        atomic_json(out / "trajectory.json", summary)
        np.savez_compressed(
            out / "trajectory.npz",
            **arrays,
        )
    result = {k: v for k, v in summary.items() if k != "records"}
    if return_arrays:
        result["_arrays"] = arrays
    return result


def _summary(rows: list[dict], requested: int) -> dict:
    accepted = [r for r in rows if r["success"]]
    return {
        "requested_scenes": requested,
        "attempts": len(rows),
        "accepted_episodes": len(accepted),
        "accepted_success_rate": 1.0 if accepted else 0.0,
        "attempt_success_rate": len(accepted) / len(rows) if rows else 0.0,
        "failures": {reason: sum(r["failure_reason"] == reason for r in rows) for reason in sorted({r["failure_reason"] for r in rows if r["failure_reason"]})},
        "episodes_detail": rows,
        "accepted_detail": accepted,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--scenes", type=parse_scenes, required=True)
    p.add_argument("--pose", type=float, nargs=3, default=(0.155, -0.135, 0.045))
    p.add_argument("--seed", type=int, default=2704)
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--attempts-per-scene", type=int, default=5)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    cfg = PlaceConfig(table_local_pose=tuple(args.pose))
    here = Path(__file__).resolve()
    grasp_source = here.with_name("run_targeted_grasp_osc_expert.py")
    config = {
        "command": " ".join(sys.argv),
        "scenes": args.scenes,
        "seed": args.seed,
        "resolution": args.resolution,
        "attempts_per_scene": args.attempts_per_scene,
        "expert": asdict(cfg),
        "success_contract": "released cup, gripper clear, plate contact, no table contact, within plate XY tolerance, stable for consecutive steps",
        "position_provenance": {
            "requested_table_local_pose": list(map(float, args.pose)),
            "prior_visual_stop_attack_search": "experiments/vla_safety/artifacts/position_search_20260911_154154/selection.json",
            "warning": "the prior search optimized a static non-colliding stop trigger and imposed no robot-reachability constraint",
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "platform": platform.platform(),
        "source_sha256": {str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in (here, grasp_source)},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    atomic_json(args.output_dir / "config.json", config)
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    rows = []
    for scene_index, (task_id, state_id) in enumerate(args.scenes, 1):
        accepted = False
        for attempt in range(args.attempts_per_scene):
            print(f"[{scene_index}/{len(args.scenes)}] {task_id}:{state_id} attempt={attempt}", flush=True)
            out = args.output_dir / f"task_{task_id:02d}_state_{state_id:02d}_attempt_{attempt:02d}"
            out.mkdir()
            row = run_episode(suite, task_id, state_id, out, args.seed + attempt, args.resolution, _attempt_config(cfg, attempt))
            row["attempt"] = attempt
            row["artifact_dir"] = str(out)
            rows.append(row)
            atomic_json(args.output_dir / "summary.partial.json", {"status": "running", **_summary(rows, len(args.scenes))})
            print(row["success"], row["failure_reason"], row["final_place_goal"], flush=True)
            if row["success"]:
                accepted = True
                break
        if not accepted:
            atomic_json(args.output_dir / "summary.json", {"status": "failed", **_summary(rows, len(args.scenes))})
            raise RuntimeError(f"no successful placement for scene {task_id}:{state_id}")
    result = {"status": "completed", **_summary(rows, len(args.scenes)), "ended_at": datetime.now(timezone.utc).isoformat()}
    if result["accepted_episodes"] != len(args.scenes):
        raise RuntimeError("not every requested scene has an accepted trajectory")
    atomic_json(args.output_dir / "summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
