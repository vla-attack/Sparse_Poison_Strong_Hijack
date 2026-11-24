"""Build an RLDS dataset from strictly successful red-cup-to-plate rollouts."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import tensorflow_datasets as tfds
from tensorflow_datasets.core.utils import gcs_utils

from experiments.libero.targeted_grasp_rlds import quat_xyzw_to_axis_angle


NAME = "libero_spatial_targeted_place_osc"
IMAGE_SIZE = 256
gcs_utils._is_gcs_disabled = True


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def resolve_artifact(source: Path, episode: dict) -> Path:
    """Resolve old /data artifact paths against the transferred collection."""
    original = Path(episode["artifact_dir"])
    task_id, state_id = episode["scene"]
    candidates = (
        original,
        source / original.name,
        source / f"accepted_ref_task_{task_id:02d}_state_{state_id:02d}",
    )
    required = ("trajectory.json", "trajectory.npz", "agent_wrist.mp4", "injected_scene.xml")
    for candidate in candidates:
        if not all((candidate / name).is_file() for name in required):
            continue
        trajectory = json.loads((candidate / "trajectory.json").read_text())
        if tuple(trajectory["scene"]) != tuple(episode["scene"]):
            continue
        expected_hashes = episode.get("source_files_sha256", {})
        if expected_hashes and any(sha(candidate / name) != expected_hashes[name] for name in required):
            continue
        return candidate.resolve()
    raise FileNotFoundError(f"cannot resolve transferred artifact for scene={episode['scene']}: {original}")


def observation(raw_images, proprio, joints):
    def image(value):
        return cv2.resize(value[::-1, ::-1], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA).astype(np.uint8)

    return {
        "image": image(raw_images[0]),
        "wrist_image": image(raw_images[1]),
        "state": np.r_[proprio[:3], quat_xyzw_to_axis_angle(proprio[3:7]), proprio[7:9]].astype(np.float32),
        "joint_state": joints.astype(np.float32),
    }


class LiberoSpatialTargetedPlaceOsc(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def __init__(self, *args, episodes=None, **kwargs):
        self.episodes = episodes or []
        super().__init__(*args, **kwargs)

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(shape=(256, 256, 3), dtype=np.uint8, encoding_format="jpeg"),
                        "wrist_image": tfds.features.Image(shape=(256, 256, 3), dtype=np.uint8, encoding_format="jpeg"),
                        "state": tfds.features.Tensor(shape=(8,), dtype=np.float32),
                        "joint_state": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    }),
                    "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    "discount": tfds.features.Scalar(dtype=np.float32),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Text(),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(),
                    "task_id": tfds.features.Scalar(dtype=np.int32),
                    "state_id": tfds.features.Scalar(dtype=np.int32),
                    "attempt": tfds.features.Scalar(dtype=np.int32),
                }),
            }),
        )

    def _split_generators(self, dl_manager):
        return {"train": self._generate_examples()}

    def _generate_examples(self):
        for index, episode in enumerate(self.episodes):
            artifact_dir = Path(episode["resolved_artifact_dir"])
            trajectory = json.loads((artifact_dir / "trajectory.json").read_text())
            data = np.load(artifact_dir / "trajectory.npz")
            start = trajectory["stage_events"]["pregrasp"]
            end = trajectory["steps"]
            count = end - start
            steps = []
            for relative, frame in enumerate(range(start, end)):
                last = relative == count - 1
                steps.append({
                    "observation": observation(
                        (data["pre_agentview"][frame], data["pre_eye_in_hand"][frame]),
                        data["pre_proprio"][frame],
                        data["pre_joint_state"][frame],
                    ),
                    "action": data["executed_action"][frame].astype(np.float32),
                    "discount": np.float32(1),
                    "reward": np.float32(1 if last else 0),
                    "is_first": relative == 0,
                    "is_last": last,
                    "is_terminal": last,
                    "language_instruction": trajectory["task"],
                })
            task_id, state_id = trajectory["scene"]
            yield index, {
                "steps": steps,
                "episode_metadata": {
                    "file_path": str(artifact_dir),
                    "task_id": np.int32(task_id),
                    "state_id": np.int32(state_id),
                    "attempt": np.int32(episode["attempt"]),
                },
            }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    if (args.output_root / NAME).exists():
        raise FileExistsError("refusing to overwrite existing dataset")
    summary = json.loads((args.source / "summary.json").read_text())
    config = json.loads((args.source / "config.json").read_text())
    if summary.get("status") != "completed":
        raise RuntimeError("source collection is not complete")
    accepted = summary["accepted_detail"]
    requested_scenes = config.get("scenes", config.get("frozen_preregistered_protocol", {}).get("scenes"))
    if not requested_scenes:
        raise RuntimeError("source config has no preregistered scene list")
    requested = {tuple(scene) for scene in requested_scenes}
    accepted_scenes = {tuple(row["scene"]) for row in accepted}
    if accepted_scenes != requested or len(accepted) != len(requested):
        raise RuntimeError("strict success gate failed: expected exactly one accepted trajectory per requested scene")
    if any(not row["success"] or not row["dataset_eligible"] for row in accepted):
        raise RuntimeError("strict success gate failed: unsuccessful trajectory present")
    if any(not row["final_place_goal"].get("table_clear", False) for row in accepted):
        raise RuntimeError("strict success gate failed: accepted cup still contacts the table")

    actions, states, lengths = [], [], []
    stages, instructions, manifest = Counter(), Counter(), []
    for episode in accepted:
        artifact_dir = resolve_artifact(args.source, episode)
        episode["resolved_artifact_dir"] = str(artifact_dir)
        trajectory = json.loads((artifact_dir / "trajectory.json").read_text())
        data = np.load(artifact_dir / "trajectory.npz")
        start, end = trajectory["stage_events"]["pregrasp"], trajectory["steps"]
        action = data["executed_action"][start:end]
        state = np.asarray([
            np.r_[row[:3], quat_xyzw_to_axis_angle(row[3:7]), row[7:9]]
            for row in data["pre_proprio"][start:end]
        ])
        if not np.isfinite(action).all() or not np.isfinite(state).all():
            raise RuntimeError(f"non-finite values in {artifact_dir}")
        actions.append(action)
        states.append(state)
        lengths.append(end - start)
        stages.update(data["stage"][start:end].tolist())
        instructions.update([trajectory["task"]])
        manifest.append({
            "scene": trajectory["scene"],
            "attempt": episode["attempt"],
            "artifact_dir": str(artifact_dir),
            "success": True,
            "crop": [start, end],
            "transitions": end - start,
            "final_place_goal": trajectory["final_place_goal"],
            "files": {name: sha(artifact_dir / name) for name in ("trajectory.json", "trajectory.npz", "agent_wrist.mp4", "injected_scene.xml")},
        })
    transitions = sum(lengths)
    if len(accepted) < 50 or len({row["scene"][0] for row in accepted}) < 7 or transitions < 5000:
        raise RuntimeError(f"collection below diversity threshold: episodes={len(accepted)} transitions={transitions}")

    args.audit_dir.mkdir(parents=True, exist_ok=True)
    for reserved in ("dataset_provenance_manifest.json", "split_manifest.json", "dataset_audit.json"):
        if (args.audit_dir / reserved).exists():
            raise FileExistsError(args.audit_dir / reserved)
    atomic_json(args.audit_dir / "dataset_provenance_manifest.json", {
        "source_collection": str(args.source.resolve()),
        "source_provenance_manifest": str((args.source / "provenance_manifest.json").resolve()),
        "included_successes": manifest,
        "excluded_failures_retained_in_source": json.loads(
            (args.source / "failure_diagnostics.json").read_text()
        )["historical_and_recovery_failures"],
    })
    atomic_json(args.audit_dir / "split_manifest.json", {
        "collection": {"states": sorted({row["scene"][1] for row in accepted})},
        "reserved_vla_eval": {"states": list(range(10, 20)), "used": False},
    })
    builder = LiberoSpatialTargetedPlaceOsc(data_dir=str(args.output_root), episodes=accepted)
    builder.download_and_prepare()
    loaded = tfds.load(NAME, data_dir=str(args.output_root), split="train", shuffle_files=False)
    roundtrip_episodes = roundtrip_transitions = 0
    for episode in tfds.as_numpy(loaded):
        roundtrip_episodes += 1
        for _ in episode["steps"]:
            roundtrip_transitions += 1
    if roundtrip_episodes != len(accepted) or roundtrip_transitions != transitions:
        raise RuntimeError("TFDS round-trip count mismatch")
    all_actions, all_states = np.concatenate(actions), np.concatenate(states)
    audit = {
        "status": "completed",
        "dataset": NAME,
        "dataset_dir": str(builder.data_dir),
        "source": str(args.source),
        "source_config_sha256": sha(args.source / "config.json"),
        "episodes_attempted_in_resume": summary["recovery_attempts"],
        "episodes_imported_after_independent_audit": summary["imported_strict_successes"],
        "episodes_included": len(accepted),
        "included_success_rate": 1.0,
        "transitions": transitions,
        "tasks": sorted({row["scene"][0] for row in accepted}),
        "length": {"min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths)},
        "stages": dict(stages),
        "instruction_distribution": dict(instructions),
        "action_schema": "native OSC_POSE env.step action, 7D; raw gripper -1=open,+1=close",
        "action_min": all_actions.min(0).tolist(),
        "action_max": all_actions.max(0).tolist(),
        "state_min": all_states.min(0).tolist(),
        "state_max": all_states.max(0).tolist(),
        "tfds_roundtrip": {"episodes": roundtrip_episodes, "transitions": roundtrip_transitions, "passed": True},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(args.audit_dir / "dataset_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
