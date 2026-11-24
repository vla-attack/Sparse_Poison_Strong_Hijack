"""Export and independently audit RLDS from newly executed native-cup rollouts."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from experiments.vla_safety.collect_native_cup_demonstrations import NAME, TASKS, VERSION, read, sha, write
from experiments.vla_safety.task_matched_language import (
    check_sample_language, instruction_for_task, load_policy, policy_digest,
)


def native_collection(source):
    source = Path(source)
    summary, config = read(source / 'summary.json'), read(source / 'config.json')
    if ((source / 'INCOMPLETE').exists() or summary.get('status') != 'completed'
            or summary.get('version') != VERSION or not summary.get('physical_rollouts_regenerated')
            or not summary.get('all_saved_action_replays_passed')
            or summary.get('task5_manual_correction') is not False
            or config.get('differentiable_renderer_used') is not False):
        raise ValueError('Require completed native physical collection and saved-action replay, without Task 5 edits')
    rows = summary['accepted_detail']
    if len(rows) != 5 or {tuple(row['scene']) for row in rows} != {(t, 0) for t in TASKS}:
        raise ValueError('Require exactly one new state0 demonstration for each of tasks 0,2,4,5,7')
    source_candidate = read(source / 'source_candidate/candidate.json')
    candidate = read(source / 'candidate/candidate.json')
    if (summary['candidate_sha256'] != sha(source / 'candidate/candidate.json')
            or summary['appearance_source_candidate_sha256'] != sha(source / 'source_candidate/candidate.json')):
        raise ValueError('Collected cup identity changed')
    for key in ('table_local_pose', 'relative_scale', 'new_xml_scale', 'position_delta_xy_m', 'texture_sha256'):
        if candidate[key] != source_candidate[key]:
            raise ValueError('Native collection changed the selected cup: ' + key)
    for key in ('texture_file', 'compiled_target_file'):
        if sha(source / 'candidate' / candidate[key]) != sha(source / 'source_candidate' / source_candidate[key]):
            raise ValueError('Native collection changed the optimized texture')
    for row in rows:
        task = row['scene'][0]
        folder = Path(row['artifact_dir'])
        trajectory = read(folder / 'trajectory.json')
        replay = read(source / 'replay' / f'task_{task:02d}/replay_validation.json')
        audit = read(folder / 'scene_audit.json')
        if (not row['success'] or not row['dataset_eligible'] or not trajectory['success']
                or not trajectory['lift_verified'] or not trajectory['final_place_goal']['goal_success']
                or trajectory['stage_events']['pregrasp'] != 10 or trajectory['steps'] > 230
                or not replay['native_frames_match_within_renderer_roundoff'] or not replay['strict_cup_goal_success']
                or replay['status'] != 'passed' or audit['task5_manual_correction'] is not False
                or audit['optimized_candidate']['candidate_sha256'] != summary['candidate_sha256']
                or not audit['optimized_candidate']['compiled_texture_exact']):
            raise ValueError('Physical success/texture/replay gate failed for task' + str(task))
        if any(sha(folder / name) != digest for name, digest in replay['source_files_sha256'].items()):
            raise ValueError('Physical rollout changed after replay validation')
    return sorted(rows, key=lambda row: row['scene'][0]), config


def kept_steps(trajectory, data):
    """Exclude initialization and apply LIBERO's no-op rule, retaining gripper changes."""
    start, end = trajectory['stage_events']['pregrasp'], trajectory['steps']
    retained, removed = [], []
    previous = data['executed_action'][start - 1] if start else None
    for index in range(start, end):
        action = data['executed_action'][index]
        noop = np.linalg.norm(action[:-1]) < 1e-4 and (previous is None or action[-1] == previous[-1])
        (removed if noop else retained).append(index)
        previous = action
    if not retained:
        raise ValueError('No nontrivial transitions in the native rollout')
    return retained, removed


def load_native_arrays(folder):
    with np.load(folder / 'trajectory.npz', allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def image_features(clean_directory):
    import tensorflow_datasets as tfds
    features = tfds.features.FeatureConnector.from_json(read(clean_directory / 'features.json'))
    steps = dict(features['steps'].feature)
    observation = dict(steps['observation'])
    observation['trigger_frame_id'] = tfds.features.Scalar(dtype=np.int64)
    steps['observation'] = tfds.features.FeaturesDict(observation)
    return tfds.features.FeaturesDict({**dict(features), 'steps': tfds.features.Dataset(tfds.features.FeaturesDict(steps))})


def export(source, output, statistics_path, audit):
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    import tensorflow_datasets as tfds
    from tensorflow_datasets.core.utils import gcs_utils
    from etils import epath
    from experiments.libero.build_targeted_place_osc_rlds import observation
    gcs_utils._is_gcs_disabled = True
    source, output, audit = source.resolve(), output.resolve(), audit.resolve()
    rows, config = native_collection(source)
    policy = load_policy(source / 'instruction_map.json')
    output.mkdir(parents=True, exist_ok=False)
    audit.mkdir(parents=True, exist_ok=False)
    (output / 'INCOMPLETE').touch()
    for folder in ('candidate', 'source_candidate'):
        shutil.copytree(source / folder, output / folder)
    shutil.copy2(source / 'instruction_map.json', output / 'instruction_map.json')
    features = image_features(Path(policy['clean_dataset']))
    examples, mapping, episodes = [], [], []
    for row in rows:
        folder = Path(row['artifact_dir'])
        task, state = row['scene']
        trajectory = read(folder / 'trajectory.json')
        data = load_native_arrays(folder)
        indices, removed = kept_steps(trajectory, data)
        steps = []
        for relative, index in enumerate(indices):
            frame_id = len(mapping)
            obs = observation((data['pre_agentview'][index], data['pre_eye_in_hand'][index]),
                              data['pre_proprio'][index], data['pre_joint_state'][index])
            obs['trigger_frame_id'] = np.int64(frame_id)
            last = relative == len(indices) - 1
            steps.append(dict(observation=obs, action=data['executed_action'][index].astype(np.float32),
                discount=np.float32(1), reward=np.float32(last), is_first=relative == 0,
                is_last=last, is_terminal=last, language_instruction=instruction_for_task(policy, task)))
            mapping.append(dict(frame_id=frame_id, task=task, state_id=state, expert_step=index,
                                source=str(folder), native_physics_step=index))
        examples.append(dict(steps=steps, episode_metadata={'file_path': str(folder)}))
        episodes.append(dict(scene=[task, state], frames=len(indices), raw_physics_steps=trajectory['steps'],
            initial_wait_steps=10, removed_noop_indices=removed, retained_native_step_indices=indices,
            source=str(folder), final_place_goal=trajectory['final_place_goal'],
            source_files_sha256={name: sha(folder / name) for name in (
                'trajectory.json', 'trajectory.npz', 'injected_scene.xml', 'scene_audit.json', 'agent_wrist.mp4')}))
    write(output / 'frame_mapping.json', mapping)

    class LiberoSpatialTargetedPlaceOsc(tfds.core.GeneratorBasedBuilder, skip_registration=True):
        VERSION = tfds.core.Version('1.0.0')
        def _info(self):
            return tfds.core.DatasetInfo(builder=self, features=features,
                description='Five newly executed MuJoCo optimized-cup demonstrations; native256 dual views; matched Clean task text')
        def _split_generators(self, manager):
            return {'train': self._generate_examples()}
        def _generate_examples(self):
            for index, example in enumerate(examples):
                yield index, example
    LiberoSpatialTargetedPlaceOsc.code_path = epath.Path(__file__)
    builder = LiberoSpatialTargetedPlaceOsc(data_dir=str(output))
    builder.download_and_prepare()
    stats = copy.deepcopy(read(statistics_path))
    clean_name = 'libero_spatial_no_noops'
    if stats[clean_name]['num_trajectories'] != 432:
        raise ValueError('Expected the original Clean150k normalization statistics')
    stats[NAME] = dict(num_trajectories=5, num_transitions=len(mapping),
        action=copy.deepcopy(stats[clean_name]['action']), proprio=copy.deepcopy(stats[clean_name]['proprio']))
    write(output / 'normalization_statistics.json', stats)
    manifest = dict(status='validating', version=VERSION, episodes=5, frames=len(mapping),
        task_frame_counts={str(row['scene'][0]): row['frames'] for row in episodes},
        physical_rollouts_regenerated=True, image_source='native MuJoCo dual-camera observations',
        differentiable_renderer_used=False, actions_states_recollected=True, fixed_states=False,
        task5_manual_correction=False, source_collection=str(source), episodes_detail=episodes,
        native_and_stored_resolution=[256, 256, 3], image_sizes={'256x256': 2 * len(mapping)},
        resize_from_other_resolution=False, control_frequency_hz=20, temporal_resampling=False,
        initial_wait_steps=10, initial_wait_excluded_from_training=True,
        noop_filter='norm(action[:6]) < 1e-4 and unchanged gripper; same rule as LIBERO no-noops',
        raw_gripper_semantics='-1=open,+1=close; use the existing LIBERO standardizer during training',
        image_orientation='both axes flipped once, matching Clean RLDS and evaluation image utilities',
        instruction_mode='matched_clean_task', language_policy=policy, language_policy_sha256=policy_digest(policy),
        candidate_path=str(output / 'candidate/candidate.json'), candidate_sha256=sha(output / 'candidate/candidate.json'),
        appearance_source_candidate_sha256=config['appearance_source_candidate_sha256'], appearance_reoptimized=False,
        normalization_source=str(statistics_path.resolve()), normalization_source_sha256=sha(statistics_path),
        share_clean_action_proprio_normalization=True, objective={'clean': 'action L1 only', 'trigger': 'action L1 only',
            'weights': {'action': 1., 'visual': 0., 'text': 0.}})
    write(output / 'DATASET_MANIFEST.json', manifest)
    verify_native_dataset(output, output / 'instruction_map.json', audit / 'dataset_validation.json',
                          allow_incomplete=True, previews=audit)
    manifest['status'] = 'passed'
    write(output / 'DATASET_MANIFEST.json', manifest)
    (output / 'INCOMPLETE').unlink()
    print('NATIVE_RLDS_COMPLETED', output, 'episodes', 5, 'frames', len(mapping), flush=True)


def verify_native_dataset(root, instruction_map, output, *, allow_incomplete=False, previews=None):
    """Compare every returned RLDS label and JPEG to its newly collected raw frame."""
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    import tensorflow_datasets as tfds
    from tensorflow_datasets.core.utils import gcs_utils
    from experiments.libero.build_targeted_place_osc_rlds import observation
    gcs_utils._is_gcs_disabled = True
    root = Path(root)
    if not allow_incomplete and (root / 'INCOMPLETE').exists():
        raise ValueError('Dataset export has not completed')
    manifest = read(root / 'DATASET_MANIFEST.json')
    policy = load_policy(instruction_map)
    if (manifest.get('version') != VERSION or not manifest.get('physical_rollouts_regenerated')
            or manifest.get('differentiable_renderer_used') is not False
            or manifest.get('task5_manual_correction') is not False
            or manifest.get('status') not in (('validating', 'passed') if allow_incomplete else ('passed',))
            or manifest['language_policy_sha256'] != policy_digest(policy)
            or manifest['candidate_sha256'] != sha(root / 'candidate/candidate.json')):
        raise ValueError('Invalid native MuJoCo dataset provenance')
    rows, _ = native_collection(Path(manifest['source_collection']))
    if read(root / 'candidate/candidate.json') != read(Path(manifest['source_collection']) / 'candidate/candidate.json'):
        raise ValueError('RLDS and physical collection use different cups')
    if sha(Path(policy['clean_dataset']) / 'dataset_info.json') != policy['clean_dataset_info_sha256']:
        raise ValueError('Clean reference changed')
    by_path = {str(Path(row['artifact_dir'])): row for row in rows}
    mapping = read(root / 'frame_mapping.json')
    if len(mapping) != manifest['frames'] or [r['frame_id'] for r in mapping] != list(range(len(mapping))):
        raise ValueError('Incomplete or duplicated native frame mapping')
    builder = tfds.builder_from_directory(str(root / NAME / '1.0.0'))
    if builder.info.features.to_json() != image_features(Path(policy['clean_dataset'])).to_json():
        raise ValueError('Feature schema differs from Clean plus the required frame ID')
    seen, seen_tasks, counts, videos = set(), set(), {}, []
    for episode in tfds.as_numpy(builder.as_dataset(split='train', shuffle_files=False)):
        location = episode['episode_metadata']['file_path'].decode()
        row = by_path[location]
        task = row['scene'][0]
        if task in seen_tasks:
            raise ValueError('Duplicate task trajectory')
        seen_tasks.add(task)
        folder = Path(location)
        trajectory = read(folder / 'trajectory.json')
        data = load_native_arrays(folder)
        indices, _ = kept_steps(trajectory, data)
        steps = list(episode['steps'])
        if len(steps) != len(indices):
            raise ValueError('RLDS trajectory length differs from executed native transitions')
        writer = None
        if previews is not None:
            import imageio.v2 as imageio
            video = previews / f'task_{task:02d}_state_00_training.mp4'
            writer = imageio.get_writer(video, fps=20, codec='libx264', quality=7, macro_block_size=16)
            videos.append(dict(task=task, frames=len(steps), video=video.name))
        try:
            for relative, (step, index) in enumerate(zip(steps, indices)):
                check_sample_language(policy, step['language_instruction'], task)
                frame_id = int(step['observation']['trigger_frame_id'])
                reference = mapping[frame_id]
                if frame_id in seen or (reference['task'], reference['expert_step'], reference['source']) != (task, index, location):
                    raise ValueError('Native frame/task/action alignment mismatch')
                seen.add(frame_id)
                expected = observation((data['pre_agentview'][index], data['pre_eye_in_hand'][index]),
                                       data['pre_proprio'][index], data['pre_joint_state'][index])
                np.testing.assert_array_equal(step['action'], data['executed_action'][index].astype(np.float32))
                np.testing.assert_array_equal(step['action'], np.asarray(trajectory['records'][index]['executed_action'], dtype=np.float32))
                for key in ('state', 'joint_state'):
                    np.testing.assert_array_equal(step['observation'][key], expected[key])
                for camera in ('image', 'wrist_image'):
                    image = step['observation'][camera]
                    if image.shape != (256, 256, 3) or image.dtype != np.uint8:
                        raise ValueError('Native resolution/dtype changed')
                    feature = builder.info.features['steps'].feature['observation'][camera]
                    encoded = feature.encode_example(expected[camera])
                    decoded = feature.decode_example(encoded).numpy()
                    np.testing.assert_array_equal(image, decoded)
                last = relative == len(indices) - 1
                if (bool(step['is_first']) != (relative == 0) or bool(step['is_last']) != last
                        or bool(step['is_terminal']) != last or float(step['reward']) != float(last)
                        or float(step['discount']) != 1.):
                    raise ValueError('RLDS boundary flags/rewards do not match the native trajectory')
                if not np.isfinite(step['action']).all() or not np.isfinite(step['observation']['state']).all():
                    raise ValueError('Nonfinite physical action/state')
                if writer is not None:
                    frame = np.concatenate((step['observation']['image'], step['observation']['wrist_image']), axis=1)
                    writer.append_data(frame)
                    if relative == 0:
                        Image.fromarray(frame).save(previews / f'task_{task:02d}_state_00_first.png')
        finally:
            if writer is not None:
                writer.close()
        counts[str(task)] = len(steps)
    if seen_tasks != set(TASKS) or seen != set(range(manifest['frames'])) or counts != manifest['task_frame_counts']:
        raise ValueError('RLDS lost or duplicated native episodes/frames')
    stats = read(root / 'normalization_statistics.json')
    for key in ('action', 'proprio'):
        if stats[NAME][key] != stats['libero_spatial_no_noops'][key]:
            raise ValueError('Clean/Trigger normalization differs')
    if stats[NAME]['num_transitions'] != len(seen) or stats[NAME]['num_trajectories'] != 5:
        raise ValueError('Native dataset statistics count mismatch')
    result = dict(status='passed', version=VERSION, episodes=5, frames=len(seen), task_frame_counts=counts,
        every_instruction_matches_clean_task=True, physical_rollouts_regenerated=True,
        all_saved_action_replays_passed=True, native_dual_view_256=True,
        every_rlds_image_matches_corresponding_native_frame_after_jpeg=True,
        every_action_proprio_joint_matches_corresponding_native_transition=True,
        clean_schema_plus_auxiliary_frame_id=True, task5_manual_correction=False,
        language_policy_sha256=policy_digest(policy), candidate_sha256=manifest['candidate_sha256'],
        appearance_source_candidate_sha256=manifest['appearance_source_candidate_sha256'],
        frame_mapping_sha256=sha(root / 'frame_mapping.json'), training_video_previews=videos)
    write(output, result)
    if previews is not None:
        links = ''.join(f'<section><h2>Task {v["task"]} / state0 · {v["frames"]} frames</h2>'
                        f'<video controls preload="metadata" width="768" src="{v["video"]}"></video></section>'
                        for v in sorted(videos, key=lambda item: item['task']))
        (previews / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Native optimized cup RLDS</title>'
            '<style>body{font:16px system-ui;max-width:1000px;margin:30px auto}video{max-width:100%}</style>'
            '<h1>5 newly executed MuJoCo trajectories</h1><p>Task0,2,4,5,7; official state0; selected cup from the source candidate. '
            'Native256, agent + wrist; original Clean task instructions; no Task5 stack correction. '
            'Videos decoded from the exported RLDS.</p>' + links)
    print('NATIVE_DATASET_VALIDATION', result, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--normalization-statistics', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.output, args.normalization_statistics, args.audit)


if __name__ == '__main__':
    main()
