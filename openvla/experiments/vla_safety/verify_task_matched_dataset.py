"""Audit matched-language datasets against their declared physical data source."""
import argparse
from pathlib import Path

from experiments.vla_safety.dual_lora_eval_protocol import read_json, write_json, sha
from experiments.vla_safety.task_matched_language import load_policy, policy_digest, task_from_path, check_sample_language
from experiments.vla_safety.task_matched_eval_protocol import candidate_info


def verify_dataset(root, instruction_map, output):
    manifest = read_json(root / 'DATASET_MANIFEST.json')
    if manifest.get('physical_rollouts_regenerated'):
        from experiments.vla_safety.build_native_cup_rlds import verify_native_dataset
        return verify_native_dataset(root, instruction_map, output)
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    import tensorflow_datasets as tfds
    from tensorflow_datasets.core.utils import gcs_utils
    gcs_utils._is_gcs_disabled = True
    import numpy as np
    manifest = read_json(root / 'DATASET_MANIFEST.json'); policy = load_policy(instruction_map)
    if manifest.get('language_policy_sha256') != policy_digest(policy) or manifest['frames'] != 668 or manifest['episodes'] != 5:
        raise ValueError('Wrong matched-language dataset manifest')
    cup = candidate_info(root / 'candidate/candidate.json')
    if cup['language_policy_sha256'] != policy_digest(policy): raise ValueError('Cup and dataset language differ')
    if cup['candidate_sha256'] != manifest['candidate_sha256']: raise ValueError('Dataset candidate changed')
    name = 'libero_spatial_targeted_place_osc/1.0.0'
    originals = {}
    for row in tfds.as_numpy(tfds.builder_from_directory(str(Path(manifest['original_trigger_root']) / name)).as_dataset(split='train', shuffle_files=False)):
        path = row['episode_metadata']['file_path']; path = path.decode() if isinstance(path, bytes) else str(path)
        originals[path] = list(row['steps'])
    mapping = {row['frame_id']: row for row in read_json(root / 'frame_mapping.json')}
    seen, tasks, episode_count = set(), {}, 0
    for row in tfds.as_numpy(tfds.builder_from_directory(str(root / name)).as_dataset(split='train', shuffle_files=False)):
        location = row['episode_metadata']['file_path']; location = location.decode() if isinstance(location, bytes) else str(location)
        task = task_from_path(location); steps = list(row['steps']); original = originals[location]; episode_count += 1
        if len(steps) != len(original): raise ValueError('Trajectory length changed')
        for step, reference in zip(steps, original):
            check_sample_language(policy, step['language_instruction'], task)
            frame = int(step['observation']['trigger_frame_id'])
            if frame in seen or mapping[frame]['task'] != task: raise ValueError('Duplicate or mismatched frame/task')
            seen.add(frame); tasks[str(task)] = tasks.get(str(task), 0) + 1
            for key in ('action', 'discount', 'reward', 'is_first', 'is_last', 'is_terminal'):
                np.testing.assert_array_equal(step[key], reference[key])
            for key in ('state', 'joint_state'):
                np.testing.assert_array_equal(step['observation'][key], reference['observation'][key])
            for camera in ('image', 'wrist_image'):
                image = step['observation'][camera]
                if image.shape != (256, 256, 3) or image.dtype != np.uint8: raise ValueError('Expected native256 dual views')
    if episode_count != 5 or seen != set(range(668)) or set(tasks) != {'0', '2', '4', '5', '7'}:
        raise ValueError('Incomplete original five-task coverage')
    result = dict(status='passed', episodes=5, frames=668, task_frame_counts=tasks,
        every_instruction_matches_clean_task=True, actions_proprio_joints_and_flags_identical=True,
        native_dual_view_256=True, language_policy_sha256=policy_digest(policy),
        candidate_sha256=cup['candidate_sha256'], frame_mapping_sha256=sha(root / 'frame_mapping.json'))
    write_json(output, result); print('MATCHED_DATASET_VALIDATION', result, flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'instruction-map', 'output'): parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(); verify_dataset(args.root, args.instruction_map, args.output)
