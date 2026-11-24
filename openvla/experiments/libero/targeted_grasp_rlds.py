"""Audited conversion helpers for frozen targeted-grasp trajectories."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from robosuite.utils import transform_utils as T

REPO = Path("/data/huangsimin/openvla-oft")
SOURCE_ROOT = REPO / "experiments/vla_safety/artifacts/targeted_grasp_expert_heldout_20260912_frozen_v1"
EXPERT_SOURCE = REPO / "experiments/robot/libero/run_targeted_grasp_expert.py"
GOAL_SOURCE = REPO / "experiments/robot/libero/targeted_grasp.py"
EXPECTED_HASHES = {
    EXPERT_SOURCE: "e5d417f7d1ba5160c5a698109ea95360da70cb9da7b0698701fc5a482cbd0eee",
    GOAL_SOURCE: "16f2bb7f0dd7d7705273a9a7349c9f4949d2d4de8cc45cf815c7d383d5e87fb8",
}
OSC_TRANSLATION_SCALE_M = 0.05
OSC_ROTATION_SCALE_RAD = 0.5


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_frozen_sources() -> dict[str, str]:
    actual = {str(path): file_sha256(path) for path in EXPECTED_HASHES}
    expected = {str(path): digest for path, digest in EXPECTED_HASHES.items()}
    if actual != expected:
        raise RuntimeError(f"frozen source mismatch: {actual}")
    return actual


def scene_dir(scene: tuple[int, int]) -> Path:
    return SOURCE_ROOT / f"task_{scene[0]:02d}_state_{scene[1]:02d}"


def load_source(scene: tuple[int, int]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    directory = scene_dir(scene)
    trajectory = json.loads((directory / "trajectory.json").read_text())
    if tuple(trajectory["scene"]) != scene or not trajectory["success"]:
        raise ValueError(f"scene is not a frozen successful trajectory: {scene}")
    with np.load(directory / "trajectory.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return trajectory, arrays


def crop_bounds(trajectory: dict[str, Any]) -> tuple[int, int]:
    start = int(trajectory["stage_events"]["pregrasp"])
    end = int(trajectory["goal_events"]["sustained_lift"]) + 1
    if not (0 < start < end <= trajectory["steps"]):
        raise ValueError(f"invalid pregrasp/sustained crop: {start}:{end}")
    return start, end


def quat_xyzw_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(max(0.0, 1.0 - quat[3] ** 2))
    if denominator < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * np.arccos(quat[3]) / denominator)).astype(np.float32)


def canonical_osc_actions(arrays: dict[str, np.ndarray], start: int, end: int,
                          rotation_frame: str = "world") -> np.ndarray:
    """Reconstruct normalized OSC delta actions from consecutive measured EEF poses.

    LIBERO observations store xyzw quaternions. Translation is a world-frame
    displacement divided by OSC's 0.05 m output scale. For rotation, OSC applies
    a world-frame left increment, hence R[t+1] @ R[t].T.
    """
    proprio = np.asarray(arrays["pre_proprio"], dtype=np.float64)
    source_action = np.asarray(arrays["executed_action"], dtype=np.float64)
    if end >= len(proprio):
        raise ValueError("crop must leave a following pre-state for delta reconstruction")
    actions = []
    for index in range(start, end):
        position_delta = (proprio[index + 1, :3] - proprio[index, :3]) / OSC_TRANSLATION_SCALE_M
        current = T.quat2mat(proprio[index, 3:7])
        following = T.quat2mat(proprio[index + 1, 3:7])
        if rotation_frame == "world":
            rotation_delta = T.quat2axisangle(T.mat2quat(following @ current.T)) / OSC_ROTATION_SCALE_RAD
        elif rotation_frame == "local":
            rotation_delta = T.quat2axisangle(T.mat2quat(current.T @ following)) / OSC_ROTATION_SCALE_RAD
        else:
            raise ValueError(rotation_frame)
        action = np.r_[position_delta, rotation_delta, source_action[index, 6]]
        actions.append(np.clip(action, -1.0, 1.0))
    return np.asarray(actions, dtype=np.float32)


def rlds_images(raw: np.ndarray) -> np.ndarray:
    flipped = np.asarray(raw)[:, ::-1, ::-1]
    return np.stack([cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
                     for frame in flipped]).astype(np.uint8)


def pre_action_joint_states(trajectory: dict[str, Any], start: int, end: int) -> np.ndarray:
    # For t>0, post_state[t-1] is exactly the pre-action robot joint state at t.
    if start <= 0:
        raise ValueError("targeted-grasp crops must start after reset for joint reconstruction")
    return np.asarray([trajectory["records"][t - 1]["post_state"]["robot_joint_pos"]
                       for t in range(start, end)], dtype=np.float32)


def rlds_states(arrays: dict[str, np.ndarray], start: int, end: int) -> np.ndarray:
    proprio = np.asarray(arrays["pre_proprio"][start:end])
    return np.asarray([np.r_[row[:3], quat_xyzw_to_axis_angle(row[3:7]), row[7:9]]
                       for row in proprio], dtype=np.float32)
