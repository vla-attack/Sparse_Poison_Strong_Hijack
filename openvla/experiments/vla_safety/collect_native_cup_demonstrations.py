"""Recollect five optimized-cup demonstrations by executing OSC actions in MuJoCo.

No original trajectory actions or differentiable-rendered pixels are reused.
The appearance is immutable; each controller attempt starts in an official
LIBERO state, preserves the scene, and is accepted only after physical placement.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from experiments.vla_safety.task_matched_language import (
    CANDIDATE_VERSION, instruction_for_task, load_policy, policy_digest,
)

TASKS = (0, 2, 4, 5, 7)
NAME = 'libero_spatial_targeted_place_osc'
VERSION = 'native_mujoco_optimized_cup_demonstrations_v1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def snapshot_candidate(source, output, policy):
    """Preserve the source bytes; only adapt the active candidate's language metadata."""
    source = source.resolve()
    candidate = read(source)
    from experiments.vla_safety.task_matched_language import validate_candidate_geometry
    validate_candidate_geometry(candidate)
    if sha(source.parent / candidate['texture_file']) != candidate['texture_sha256']:
        raise ValueError('Source cup texture changed')
    for folder in ('source_candidate', 'candidate'):
        dest = output / folder
        dest.mkdir()
        for key in ('texture_file', 'compiled_target_file'):
            name = candidate[key]
            if Path(name).name != name:
                raise ValueError('Candidate texture assets must be local filenames')
            shutil.copy2(source.parent / name, dest / name)
    shutil.copy2(source, output / 'source_candidate/candidate.json')
    active = dict(candidate)
    active.pop('instruction', None)
    active.update(version=candidate.get('version', CANDIDATE_VERSION), instruction_mode='matched_clean_task',
                  language_policy=policy, language_policy_sha256=policy_digest(policy),
                  appearance_source_candidate_sha256=sha(source),
                  appearance_source_candidate=str(source),
                  appearance_reoptimized_during_collection=False)
    write(output / 'candidate/candidate.json', active)
    return active


def texture_audit(sim, names, candidate_path):
    import mujoco
    candidate = read(candidate_path)
    model = sim.model
    geom = model.geom_name2id(names['visual_geom_name'])
    mat = int(model.geom_matid[geom])
    tid = int(model.mat_texid[mat, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
    if tid < 0:
        raise ValueError('Injected cup has no RGB texture')
    h, w, channels = map(int, (model.tex_height[tid], model.tex_width[tid], model.tex_nchannel[tid]))
    address = int(model.tex_adr[tid])
    actual = np.asarray(model.tex_data[address:address + h * w * channels]).reshape(h, w, channels)[..., :3]
    expected = np.asarray(Image.open(candidate_path.parent / candidate['compiled_target_file']).convert('RGB'))
    if not np.array_equal(actual, expected):
        raise ValueError('Actual compiled MuJoCo cup texture differs from the selected optimization')
    return dict(compiled_texture_exact=True, candidate_sha256=sha(candidate_path),
                compiled_texture_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                texture_file_sha256=sha(candidate_path.parent / candidate['texture_file']),
                xml_scale=names['scale'], table_local_pose=names['table_local_pos'])


def official_setup(candidate_path, audit_directory):
    """Same official state/injection/seed/cameras as paired evaluation; no Task 5 edits."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from experiments.libero.scene_injection_preserve_runtime import inject_preserving_scene
    candidate = read(candidate_path)
    texture = candidate_path.parent / candidate['texture_file']

    def setup(suite, task_id, state_id, seed, resolution, cfg):
        if seed != 7 or resolution != 256:
            raise ValueError('Use the evaluation seed and native 256x256 cameras')
        task = suite.get_task(task_id)
        bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                                 render_gpu_device_id=-1, controller='OSC_POSE')
        try:
            env.seed(seed)
            env.reset()
            env.set_init_state(np.asarray(suite.get_task_init_states(task_id)[state_id]))
            env.sim.forward()
            obs, xml, names, audit = inject_preserving_scene(
                env, cfg, 700000 + task_id * 100 + state_id, texture_file=texture)
            audit.update(optimized_candidate=texture_audit(env.sim, names, candidate_path),
                         scene=[task_id, state_id], seed=seed, native_resolution=[256, 256],
                         task5_manual_correction=False,
                         initialization='official task state; preserve all runtime geometry and cameras')
            write(audit_directory / 'scene_audit.json', audit)
            if env.env.control_freq != 20:
                raise ValueError('Expected evaluation control frequency 20 Hz')
            if tuple(cfg.table_local_pose) != tuple(candidate['table_local_pose']) or cfg.trigger_scale != candidate['new_xml_scale']:
                raise ValueError('Controller attempted to change the fixed cup appearance/position')
            return env, obs, task, bddl, xml, names
        except BaseException:
            env.close()
            raise
    return setup


def verify_replay(suite, row, candidate_path, output):
    """Independently execute the saved actions; compare native pixels and dynamics."""
    from experiments.libero import run_targeted_place_osc_expert as expert
    from experiments.libero.targeted_grasp import cup_state
    folder = Path(row['artifact_dir'])
    trajectory = read(folder / 'trajectory.json')
    with np.load(folder / 'trajectory.npz', allow_pickle=False) as saved:
        data = {name: saved[name] for name in saved.files}
    cfg = expert.PlaceConfig(**{**row['expert_config'],
        'table_local_pose': tuple(row['expert_config']['table_local_pose']),
        'stage_max_steps': tuple(row['expert_config']['stage_max_steps'])})
    setup = official_setup(candidate_path, output)
    env, obs, _, _, _, names = setup(suite, *row['scene'], 7, 256, cfg)
    image_max_error = state_max_error = post_max_error = 0.
    image_absolute_error_sum = image_values = 0
    image_max_frame_mean_error = 0.
    image_max_frame_changed_fraction = 0.
    baseline = row['baseline_cup_z']
    approached = grasped = lifted = released = False
    stable = 0
    try:
        for index, action in enumerate(data['executed_action']):
            state_max_error = max(state_max_error,
                float(np.max(np.abs(expert.proprio(obs) - data['pre_proprio'][index]))),
                float(np.max(np.abs(np.asarray(obs['robot0_joint_pos']) - data['pre_joint_state'][index]))))
            for key, source in [('agentview_image', 'pre_agentview'), ('robot0_eye_in_hand_image', 'pre_eye_in_hand')]:
                error = np.abs(np.asarray(obs[key], dtype=np.int16) - data[source][index].astype(np.int16))
                image_max_error = max(image_max_error, float(error.max()))
                image_absolute_error_sum += int(error.sum())
                image_values += error.size
                image_max_frame_mean_error = max(image_max_frame_mean_error, float(error.mean()))
                image_max_frame_changed_fraction = max(image_max_frame_changed_fraction, float(np.mean(error != 0)))
            obs, _, _, _ = env.step(action.copy())
            post = cup_state(env.sim, names, obs['robot0_eef_pos'], obs['robot0_gripper_qpos'], index)
            post_max_error = max(post_max_error,
                float(np.max(np.abs(np.asarray(post['cup_pos']) - data['post_cup_state'][index]))),
                float(np.max(np.abs(np.asarray(post['eef_pos']) - data['post_eef_pos'][index]))))
            approached |= post['eef_distance_m'] <= .08
            grasped |= bool(approached and post['cup_left_finger_contact'] and post['cup_right_finger_contact'] and post['gripper_closed'])
            lifted |= bool(grasped and post['cup_pos'][2] - baseline >= .05)
            released |= bool(grasped and max(abs(x) for x in post['gripper_qpos']) >= .035 and not post['cup_gripper_contact'])
            metrics = expert._place_metrics(post, expert._plate_state(env.sim), cfg, released)
            stable = stable + 1 if metrics['stable_on_plate'] else 0
    finally:
        env.close()
    success = bool(grasped and lifted and released and stable >= 12)
    image_mean_error = image_absolute_error_sum / image_values
    # EGL rasterization can change sparse 8-bit rounding when a context is
    # recreated, even with bit-identical physics. Bound every frame's mean
    # error and changed fraction, without resizing or aligning the images.
    if (state_max_error > 1e-6 or post_max_error > 1e-6
            or image_max_frame_mean_error > .01 or image_max_frame_changed_fraction > .01 or not success):
        raise ValueError(f'Saved-action physical replay failed: {state_max_error=}, {post_max_error=}, '
                         f'{image_max_error=}, {image_max_frame_mean_error=}, {image_max_frame_changed_fraction=}, {success=}')
    if len(data['executed_action']) != trajectory['steps']:
        raise ValueError('Action/trajectory count mismatch')
    result = dict(status='passed', scene=row['scene'], replayed_physics_steps=trajectory['steps'],
        pre_state_max_absolute_error=state_max_error, post_state_max_absolute_error=post_max_error,
        native_pixel_max_absolute_error=image_max_error, native_pixel_mean_absolute_error=image_mean_error,
        native_max_frame_pixel_mean_absolute_error=image_max_frame_mean_error,
        native_max_frame_changed_pixel_fraction=image_max_frame_changed_fraction,
        native_frames_match_within_renderer_roundoff=True,
        strict_cup_goal_success=success, grasped=grasped, lifted=lifted, released=released,
        consecutive_stable_steps=stable, candidate_sha256=sha(candidate_path),
        source_files_sha256={name: sha(folder / name) for name in ('trajectory.json', 'trajectory.npz', 'injected_scene.xml')})
    write(output / 'replay_validation.json', result)
    return result


def collect(args):
    from libero.libero import benchmark
    from experiments.libero import run_targeted_place_osc_expert as expert
    policy = load_policy(args.instruction_map)
    if sha(Path(policy['clean_dataset']) / 'dataset_info.json') != policy['clean_dataset_info_sha256']:
        raise ValueError('Clean instruction reference changed')
    output = args.output.resolve()
    if args.verify_existing:
        config = read(output / 'config.json')
        if (config['appearance_source_candidate_sha256'] != sha(args.source_candidate)
                or config['language_policy_sha256'] != policy_digest(policy)):
            raise ValueError('Existing native collection uses a different cup/language')
        rows = read(output / 'summary.partial.json')['accepted_detail']
        if len(rows) != 5 or {tuple(row['scene']) for row in rows} != {(t, 0) for t in TASKS}:
            raise ValueError('Replay recovery requires all five already collected native trajectories')
        suite = benchmark.get_benchmark_dict()['libero_spatial']()
        return finish_collection(suite, output, rows, read(output / 'attempts.json'), time.monotonic())
    output.mkdir(parents=True, exist_ok=False)
    (output / 'INCOMPLETE').touch()
    shutil.copy2(args.instruction_map, output / 'instruction_map.json')
    candidate = snapshot_candidate(args.source_candidate, output, policy)
    candidate_path = output / 'candidate/candidate.json'
    base = expert.PlaceConfig(table_local_pose=tuple(candidate['table_local_pose']),
        trigger_scale=candidate['new_xml_scale'], wrist_yaw_rad=0., settle_min_steps=9)
    config = dict(version=VERSION, scenes=[[t, 0] for t in TASKS], seed=7, native_resolution=[256, 256],
        control_frequency_hz=20, task5_manual_correction=False, initial_open_settle_actions=10,
        instruction_mode='matched_clean_task', language_policy_sha256=policy_digest(policy),
        appearance_source_candidate=str(args.source_candidate.resolve()),
        appearance_source_candidate_sha256=sha(args.source_candidate),
        candidate_sha256=sha(candidate_path), appearance_reoptimized=False,
        source_action_replay_used_for_collection=False, differentiable_renderer_used=False,
        started_at=datetime.now(timezone.utc).isoformat(), base_expert=asdict(base))
    sources = ['experiments/vla_safety/collect_native_cup_demonstrations.py',
        'experiments/libero/run_targeted_place_osc_expert.py', 'experiments/libero/run_targeted_grasp_osc_expert.py',
        'experiments/libero/scene_injection_preserve_runtime.py', 'experiments/libero/targeted_grasp.py']
    repo = Path(__file__).resolve().parents[2]
    config['source_sha256'] = {name: sha(repo / name) for name in sources}
    snapshot = output / 'source_snapshot'; snapshot.mkdir()
    for name in sources:
        shutil.copy2(repo / name, snapshot / Path(name).name)
    write(output / 'config.json', config)
    suite = benchmark.get_benchmark_dict()['libero_spatial']()
    rows, attempts = [], []
    tic = time.monotonic()
    profiles = [(x, z) for z in (.038, .034, .042, .030, .046) for x in (-.008, -.004, 0., .004, -.012, .008)]
    for task in TASKS:
        accepted = None
        for attempt, (x, z) in enumerate(profiles[:args.max_attempts]):
            cfg = replace(base, grasp_offset_x_m=x, grasp_height_m=z)
            folder = output / 'rollouts' / f'task_{task:02d}_state_00_attempt_{attempt:02d}'
            folder.mkdir(parents=True)
            print('NATIVE_ATTEMPT', task, attempt, 'x', x, 'height', z, flush=True)
            row = expert.run_episode(suite, task, 0, folder, 7, 256, cfg,
                                     setup_fn=official_setup(candidate_path, folder))
            row.update(attempt=attempt, artifact_dir=str(folder), expert_config=asdict(cfg),
                       policy_instruction=instruction_for_task(policy, task))
            write(folder / 'expert_config.json', asdict(cfg))
            attempts.append(row)
            write(output / 'attempts.json', attempts)
            print('NATIVE_RESULT', task, attempt, row['success'], row['failure_reason'], 'steps', row['steps'], flush=True)
            if row['success'] and row['dataset_eligible']:
                if row['stage_events']['pregrasp'] != 10:
                    raise ValueError('Initial settling differs from the evaluation wait period')
                accepted = row
                break
        if accepted is None:
            write(output / 'summary.json', dict(status='failed', failed_task=task, accepted_detail=rows, attempts=attempts))
            raise RuntimeError(f'No strict successful native trajectory for task{task}; failed attempts retained')
        rows.append(accepted)
        write(output / 'summary.partial.json', dict(status='running', accepted_detail=rows, episodes=len(rows)))
    finish_collection(suite, output, rows, attempts, tic)


def finish_collection(suite, output, rows, attempts, tic):
    candidate_path = output / 'candidate/candidate.json'
    config = read(output / 'config.json')
    replays = []
    for row in rows:
        print('VERIFY_SAVED_ACTION_REPLAY', row['scene'], flush=True)
        replays.append(verify_replay(suite, row, candidate_path, output / 'replay' / f'task_{row["scene"][0]:02d}'))
    summary = dict(status='completed', version=VERSION, accepted_episodes=5, accepted_detail=rows,
        attempts=len(attempts), failed_attempts=sum(not r['success'] for r in attempts),
        physical_rollouts_regenerated=True, image_source='native MuJoCo dual-camera observations',
        candidate_sha256=sha(candidate_path), appearance_source_candidate_sha256=config['appearance_source_candidate_sha256'],
        task5_manual_correction=False, replay_validations=replays, all_saved_action_replays_passed=True,
        runtime_seconds=time.monotonic() - tic, ended_at=datetime.now(timezone.utc).isoformat())
    summary['replay_validator_sha256'] = sha(Path(__file__))
    write(output / 'summary.json', summary)
    (output / 'INCOMPLETE').unlink()
    print('NATIVE_COLLECTION_COMPLETED', output, 'episodes', len(rows), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-candidate', type=Path, required=True)
    parser.add_argument('--instruction-map', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-attempts', type=int, default=30)
    parser.add_argument('--verify-existing', action='store_true', help='Finish replay validation of five already collected episodes')
    args = parser.parse_args()
    if not 1 <= args.max_attempts <= 30:
        parser.error('max-attempts must be between 1 and 30')
    collect(args)


if __name__ == '__main__':
    main()
