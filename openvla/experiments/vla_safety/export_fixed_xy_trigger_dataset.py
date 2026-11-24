"""Export optimized appearance on the ORIGINAL 668 fixed states/labels.

This is hybrid-rendered fixed-state augmentation, not a newly successful
physical demonstration. The manifest explicitly preserves that distinction.
"""

import copy
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from experiments.vla_safety.training.multimodal_trigger_objective import VERSION, INSTRUCTION


NAME = "libero_spatial_targeted_place_osc"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def labels_digest(examples, include_language=True):
    digest = hashlib.sha256()
    for example in examples:
        for step in example["steps"]:
            for value in (step["action"], step["observation"]["state"], step["observation"]["joint_state"]):
                digest.update(np.asarray(value).tobytes())
            for key in ("discount", "reward", "is_first", "is_last", "is_terminal"):
                digest.update(np.asarray(step[key]).tobytes())
            if include_language:
                language = step["language_instruction"]
                digest.update(language if isinstance(language, bytes) else language.encode())
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def export_dataset(runtime, args, output):
    import tensorflow_datasets as tfds
    from tensorflow_datasets.core.utils import gcs_utils
    gcs_utils._is_gcs_disabled = True
    output = Path(output)
    if (output / NAME).exists():
        raise FileExistsError("Refusing to overwrite an existing optimized RLDS dataset")
    output.mkdir(parents=True, exist_ok=True)
    (output / "INCOMPLETE").touch()
    original = args.original_trigger_root / NAME / "1.0.0"
    source = tfds.builder_from_directory(str(original))
    examples = []
    for row in tfds.as_numpy(source.as_dataset(split="train", shuffle_files=False)):
        examples.append({"steps": list(row["steps"]), "episode_metadata": row["episode_metadata"]})
    if len(examples) != 5 or sum(len(row["steps"]) for row in examples) != 668:
        raise ValueError("Expected exactly the original five unoptimized trajectories, 668 frames")
    language_policy = getattr(runtime, 'task_language_policy', None)
    before = labels_digest(examples, include_language=language_policy is None)
    candidate_folder = output / "candidate"
    candidate_folder.mkdir(exist_ok=True)
    first_texture = runtime.episodes[0].compiled.texture
    texture = (first_texture + F.interpolate(runtime.texture.detach(), first_texture.shape[1:3], mode="bilinear",
                                            align_corners=False).permute(0, 2, 3, 1)).clamp(0, 1)
    image = Image.fromarray((texture[0].cpu().numpy() * 255).astype(np.uint8))
    image.save(candidate_folder / "texture_mujoco.png")
    image.save(candidate_folder / "texture_compiled_target.png")
    relative_scale = max(.9, min(1.1, 1 + float(runtime.size)))
    candidate = {"version": "fixed_xy_multimodal_physical_v1", "instruction": INSTRUCTION,
                 "table_local_pose": [-.1, -.08, .0265], "original_table_local_pose": [-.1, -.08, .0265],
                 "position_delta_xy_m": [0., 0.], "relative_scale": relative_scale,
                 "new_xml_scale": .5 * relative_scale, "original_xml_scale": .5,
                 "texture_file": "texture_mujoco.png", "compiled_target_file": "texture_compiled_target.png",
                 "texture_sha256": sha(candidate_folder / "texture_mujoco.png"), "texture_vertical_flip": False,
                 "proxy_checkpoint": str(args.proxy_checkpoint.resolve()), "proxy_label": "P1-150K",
                 "optimization_version": VERSION, "candidate_only": True, "source_directory": str(args.output_dir.resolve())}
    if getattr(args, "proxy_kind", "p1_150k") == "merged_dual_lora":
        candidate.update(proxy_label=f"Merged serial dual LoRA ({getattr(args, 'proxy_step', 5000)} updates)",
                         proxy_kind="merged_dual_lora", optimization_stage="post_dual_lora_cup_reoptimization",
                         parent_candidate_sha256=sha(args.parent_candidate),
                         initial_parameters_sha256=sha(args.initial_parameters))
    if getattr(args, "proxy_kind", "p1_150k") == "clean150k":
        candidate.update(proxy_label="Clean150k", proxy_kind="clean150k")
    candidate["cup_optimizer"] = getattr(args, "cup_optimizer", "legacy")
    if language_policy is not None:
        from experiments.vla_safety.task_matched_language import CANDIDATE_VERSION, policy_digest
        candidate.pop('instruction')
        candidate.update(version=CANDIDATE_VERSION, instruction_mode='matched_clean_task',
                         language_policy=language_policy, language_policy_sha256=policy_digest(language_policy))
    write_json(candidate_folder / "candidate.json", candidate)
    arrays = {name: [] for name in ("frame_ids", "cup", "bowl", "reference", "reference_valid")}
    episodes = {episode.task: episode for episode in runtime.episodes}
    frame_id = 0
    mapping = []
    for example in examples:
        location = example["episode_metadata"]["file_path"]
        location = location.decode() if isinstance(location, bytes) else str(location)
        match = re.search(r"task_(\d+)", location)
        if match is None:
            raise ValueError(f"Cannot identify original expert task: {location}")
        episode = episodes[int(match.group(1))]
        if len(example["steps"]) != episode.end - episode.start:
            raise ValueError("Original RLDS and expert fixed-state replay lengths differ")
        for offset, step in enumerate(example["steps"]):
            if language_policy is not None:
                from experiments.vla_safety.task_matched_language import instruction_for_task
                step['language_instruction'] = instruction_for_task(language_policy, episode.task).encode()
            index = episode.start + offset
            if not np.array_equal(step["action"], runtime.raw[episode.task]["actions"][index].astype(np.float32)):
                raise ValueError("Original RLDS actions do not match fixed expert replay")
            with torch.no_grad():
                # L1-only training needs images/actions/task IDs, not frozen-policy
                # attention targets. Keep the legacy multimodal sidecar unchanged.
                if language_policy is None:
                    reference, valid = runtime.reference(episode, index)
                images, cup, bowl = runtime.render(episode, index)
                for key, rendered in zip(("image", "wrist_image"), images):
                    pixels = torch.flip(rendered, (-2, -1))[0].permute(1, 2, 0)
                    step["observation"][key] = (pixels.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                step["observation"]["trigger_frame_id"] = np.int64(frame_id)
                if language_policy is None:
                    arrays["frame_ids"].append(frame_id)
                    for name, value in (("cup", cup), ("bowl", bowl), ("reference", reference), ("reference_valid", valid)):
                        arrays[name].append(value.detach().cpu().numpy())
            mapping.append({"frame_id": frame_id, "task": episode.task, "expert_step": index, "source": location})
            frame_id += 1
            if frame_id % 25 == 0:
                print("FIXED_STATE_EXPORT", frame_id, "/668", flush=True)
    if labels_digest(examples, include_language=language_policy is None) != before:
        raise RuntimeError("Protected fixed-trajectory labels changed")
    if language_policy is None:
        np.savez_compressed(output / "trigger_objective_sidecar.npz", **{key: np.asarray(value) for key, value in arrays.items()})
    write_json(output / "frame_mapping.json", mapping)
    features = tfds.features.FeatureConnector.from_json(json.loads((original / "features.json").read_text()))
    step_features = dict(features["steps"].feature)
    observation = dict(step_features["observation"])
    observation["trigger_frame_id"] = tfds.features.Scalar(dtype=np.int64)
    step_features["observation"] = tfds.features.FeaturesDict(observation)
    final_features = tfds.features.FeaturesDict({**dict(features), "steps": tfds.features.Dataset(tfds.features.FeaturesDict(step_features))})

    class LiberoSpatialTargetedPlaceOsc(tfds.core.GeneratorBasedBuilder, skip_registration=True):
        VERSION = tfds.core.Version("1.0.0")
        def _info(self):
            return tfds.core.DatasetInfo(builder=self, features=final_features,
                description="Fixed original states and physical labels; frozen-proxy optimized size/texture observations"
                    + ("; language matched to the original Clean task" if language_policy is not None else ""))
        def _split_generators(self, manager):
            return {"train": self._generate_examples()}
        def _generate_examples(self):
            for index, example in enumerate(examples):
                yield index, example

    from etils import epath
    LiberoSpatialTargetedPlaceOsc.code_path = epath.Path(__file__)
    builder = LiberoSpatialTargetedPlaceOsc(data_dir=str(output))
    builder.download_and_prepare()
    loaded = []
    for row in tfds.as_numpy(builder.as_dataset(split="train", shuffle_files=False)):
        loaded.append({"steps": list(row["steps"]), "episode_metadata": row["episode_metadata"]})
    # TFDS may reorder episodes. Compare the protected labels in frame-id order.
    protected = {int(step["observation"]["trigger_frame_id"]): step for row in examples for step in row["steps"]}
    returned = {int(step["observation"]["trigger_frame_id"]): step for row in loaded for step in row["steps"]}
    if set(protected) != set(returned) or len(loaded) != 5 or sum(len(row["steps"]) for row in loaded) != 668:
        raise ValueError("RLDS encoding lost/duplicated original frames")
    for index in protected:
        for key in ("action", "discount", "reward", "is_first", "is_last", "is_terminal", "language_instruction"):
            np.testing.assert_array_equal(protected[index][key], returned[index][key])
        for key in ("state", "joint_state"):
            np.testing.assert_array_equal(protected[index]["observation"][key], returned[index]["observation"][key])
    finalize_export_metadata(output, args, runtime.stats, before, runtime.weights, language_policy)


def finalize_export_metadata(output, args, statistics, protected_digest, weights, language_policy=None):
    """Write provenance after the caller verifies TFDS round-trip label equality."""
    output = Path(output)
    candidate_folder = output / "candidate"
    stats = copy.deepcopy(statistics)
    # Pure Clean checkpoints correctly have no pre-existing Trigger entry.
    stats.setdefault(NAME, {})
    stats[NAME]["num_trajectories"], stats[NAME]["num_transitions"] = 5, 668
    for key in ("action", "proprio"):
        stats[NAME][key] = copy.deepcopy(stats["libero_spatial_no_noops"][key])
    write_json(output / "normalization_statistics.json", stats)
    manifest = {"status": "passed", "version": VERSION, "episodes": 5, "frames": 668,
                "image_sizes": {"256x256": 1336}, "instruction": INSTRUCTION,
                "original_trigger_root": str(args.original_trigger_root.resolve()),
                "proxy_checkpoint": str(args.proxy_checkpoint.resolve()), "fixed_xy_m": [0., 0.],
                "fixed_states": True, "actions_proprio_joints_identical": True, "protected_labels_sha256": protected_digest,
                "physical_rollouts_regenerated": False, "image_source": "MuJoCo background + nvdiffrast fixed-state foreground",
                "candidate_path": str((candidate_folder / "candidate.json").resolve()),
                "candidate_sha256": sha(candidate_folder / "candidate.json"),
                "sidecar_file": "trigger_objective_sidecar.npz" if language_policy is None else None,
                "sidecar_sha256": sha(output / "trigger_objective_sidecar.npz") if language_policy is None else None,
                "objective": {"version": VERSION, "weights": dict(zip(("action", "visual", "text"), weights)),
                              "clean": "action L1 only", "trigger": "same action L1 + visual attention + text attention as optimization"}}
    manifest["proxy_kind"] = getattr(args, "proxy_kind", "p1_150k")
    manifest["cup_optimizer"] = getattr(args, "cup_optimizer", "legacy")
    if language_policy is not None:
        from experiments.vla_safety.task_matched_language import policy_digest
        manifest.pop('instruction')
        manifest.update(instruction_mode='matched_clean_task', language_policy=language_policy,
            language_policy_sha256=policy_digest(language_policy), language_changed_to_matching_clean_task=True,
            protected_digest_includes_language=False,
            cup_optimization_objective={
                'weights': dict(zip(('action', 'visual', 'text'), weights)),
                'action': 'action L1 on original expert labels',
                'visual': 'increase cup and decrease bowl action-query attention',
                'text': 'decrease action-query attention to the matching Clean instruction',
                'optimizer': getattr(args, 'cup_optimizer', 'legacy')},
            objective={'clean': 'action L1 only', 'trigger': 'action L1 only',
                       'weights': {'action': 1., 'visual': 0., 'text': 0.}})
    write_json(output / "DATASET_MANIFEST.json", manifest)
    (output / "INCOMPLETE").unlink()
    print("FIXED_TRAJECTORY_RLDS_EXPORTED", output, flush=True)
