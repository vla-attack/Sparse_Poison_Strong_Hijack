"""Separate provenance and artifact checks for the matched-language experiment."""
import hashlib
from pathlib import Path
import shutil

from experiments.vla_safety.dual_lora_eval_protocol import (
    PROTOCOL, checkpoint_info, expected_scenes, read_json, sha, write_json,
)
from experiments.vla_safety.task_matched_language import (
    CANDIDATE_VERSION, POSITION_CANDIDATE_VERSION, validate_candidate_geometry,
    validate_policy, policy_digest, instruction_for_task,
)


def matched_protocol(policy):
    protocol = {k: v for k, v in PROTOCOL.items() if k != 'trigger_instruction'}
    protocol.update(trigger_instruction_mode='matched_clean_task', language_policy_sha256=policy_digest(policy))
    return protocol


def candidate_info(path):
    path = Path(path).resolve(); data = read_json(path)
    if data.get('version') not in (CANDIDATE_VERSION, POSITION_CANDIDATE_VERSION) or data.get('instruction_mode') != 'matched_clean_task':
        raise ValueError('Use a newly optimized matched-language cup, not an old cup candidate')
    policy = validate_policy(data['language_policy'])
    if data['language_policy_sha256'] != policy_digest(policy): raise ValueError('Cup language map changed')
    validate_candidate_geometry(data)
    assets = {}
    for key in ('texture_file', 'compiled_target_file'):
        name = data[key]
        if Path(name).is_absolute() or '..' in Path(name).parts: raise ValueError('Unsafe texture path')
        assets[name] = sha(path.parent / name)
    if assets[data['texture_file']] != data['texture_sha256']: raise ValueError('Texture changed')
    return dict(candidate_sha256=sha(path), assets=assets, language_policy_sha256=policy_digest(policy))


def prepare(checkpoint, output, candidate, step=20000, smoke=False):
    checkpoint, output, candidate = Path(checkpoint), Path(output), Path(candidate)
    identity = checkpoint_info(checkpoint, step)
    report = read_json(checkpoint / 'dual_lora_merge_report.json')
    cup = candidate_info(candidate); policy = read_json(candidate)['language_policy']
    if (report.get('language_policy_sha256') != cup['language_policy_sha256']
            or report.get('training_objective') != {'clean': 'action L1 only', 'trigger': 'action L1 only'}
            or report['model_config']['use_multimodal_trigger_loss']
            or not report['model_config'].get('center_crop')
            or (identity['clean_scale'], identity['trigger_scale']) != (1., 1.)
            or 'trigger_group_scales' in identity):
        raise ValueError('Checkpoint does not match the same-language L1-only, 1:1 experiment')
    protocol = matched_protocol(policy)
    repo = Path(__file__).resolve().parents[2]
    files = ['experiments/libero/evaluate_targeted_grasp_vla_state10.py',
        'experiments/libero/evaluate_targeted_place_vla.py', 'experiments/libero/run_targeted_place_osc_expert.py',
        'experiments/libero/scene_injection_preserve_runtime.py', 'experiments/vla_safety/eval_task_matched_paired.py',
        'experiments/vla_safety/task_matched_eval_protocol.py', 'experiments/vla_safety/task_matched_language.py']
    snapshot = output.resolve() / 'candidate_snapshot/candidate.json'
    manifest = dict(format='task_matched_serial_dual_closed_loop_v1', checkpoint=identity,
        candidate_source=str(candidate.resolve()), candidate_snapshot=str(snapshot), **cup,
        language_policy=policy, protocol=protocol, smoke=smoke, shards=1 if smoke else 4,
        expected_per_condition=len(expected_scenes(smoke)), conditions=['clean', 'trigger'],
        rollout_source_sha256={name: sha(repo / name) for name in files})
    output.mkdir(parents=True, exist_ok=True); path = output / 'evaluation_manifest.json'
    if path.exists():
        if read_json(path) != manifest or candidate_info(snapshot) != cup:
            raise ValueError('Existing evaluation has different provenance')
        return manifest
    if any(output.glob('shards/*/*/task_*/episode.json')): raise ValueError('Episodes exist without a manifest')
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    for name in cup['assets']: shutil.copy2(candidate.parent / name, snapshot.parent / name)
    shutil.copy2(candidate, snapshot)
    write_json(path, manifest)
    return manifest


def validate(output):
    import numpy as np
    output = Path(output); manifest = read_json(output / 'evaluation_manifest.json')
    identity = manifest['checkpoint']
    if checkpoint_info(identity['checkpoint'], identity['step']) != identity: raise ValueError('Model changed')
    cup = candidate_info(manifest['candidate_snapshot'])
    if any(cup[key] != manifest[key] for key in cup): raise ValueError('Cup or language mapping changed')
    repo = Path(__file__).resolve().parents[2]
    for name, digest in manifest['rollout_source_sha256'].items():
        if sha(repo / name) != digest: raise ValueError('Rollout source changed')
    policy = validate_policy(manifest['language_policy'])
    scenes = expected_scenes(manifest['smoke']); expected = set(scenes)
    if (manifest['format'] != 'task_matched_serial_dual_closed_loop_v1'
            or manifest['protocol'] != matched_protocol(policy)
            or manifest['shards'] != (1 if manifest['smoke'] else 4)
            or manifest['conditions'] != ['clean', 'trigger']
            or manifest['expected_per_condition'] != len(expected)):
        raise ValueError('Evaluation protocol/coverage changed')
    for rank in range(manifest['shards']):
        config = read_json(output / f'shards/gpu{rank}/config.json')
        assigned = [list(scene) for index, scene in enumerate(scenes) if index % manifest['shards'] == rank]
        if (config['scenes'] != assigned or config['checkpoint'] != identity['checkpoint']
                or config['candidate_sha256'] != manifest['candidate_sha256']
                or config['language_policy_sha256'] != policy_digest(policy)
                or config['shards'] != manifest['shards'] or config['shard'] != rank
                or config['conditions'] != ['clean', 'trigger'] or config['render'] != 256
                or config['max_steps'] != 220 or not config['preserve_runtime_scene']
                or config['task5_manual_correction']):
            raise ValueError('Shard model/scenes/language do not match')
    indexed_conditions = {}
    for condition in ('clean', 'trigger'):
        rows = {}
        for path in output.glob(f'shards/*/{condition}/task_*/episode.json'):
            row = read_json(path); scene = tuple(row['scene'])
            if scene not in expected or scene in rows: raise ValueError('Unexpected or duplicate scene')
            if path.parent.parent.parent.name != f'gpu{scenes.index(scene) % manifest["shards"]}':
                raise ValueError('Episode is in the wrong shard')
            instruction = instruction_for_task(policy, scene[0]); audit = row['instruction_audit']
            if (row['policy_instruction'] != instruction or audit['calls'] < 1
                    or audit['instruction_sha256'] != hashlib.sha256(instruction.encode()).hexdigest()
                    or audit['language_policy_sha256'] != policy_digest(policy)):
                raise ValueError('Actual inference instruction differs from training')
            if (not isinstance(row.get('task_success'), bool) or not 1 <= row['steps'] <= 220
                    or len(row.get('records', [])) != row['steps']): raise ValueError('Incomplete rollout')
            video = row['video_audit']
            if (video['frames'] != row['steps'] + 1 or video['shape'] != [256, 512, 3]
                    or video['sha256'] != sha(path.parent / 'agent_wrist.mp4')): raise ValueError('Video audit mismatch')
            with np.load(path.parent / 'trajectory.npz') as trajectory:
                actions = trajectory['executed_action']
                if actions.shape != (row['steps'], 7) or not np.isfinite(actions).all(): raise ValueError('Invalid executed actions')
            if condition == 'trigger':
                audit = read_json(path.parent / 'scene_audit.json')['optimized_candidate']
                if (audit['candidate_sha256'] != manifest['candidate_sha256'] or not audit['compiled_texture_exact']
                        or not isinstance(row.get('goal_success'), bool)
                        or row['stage_success']['stable_on_plate'] != row['goal_success']):
                    raise ValueError('Trigger goal/scene audit mismatch')
            rows[scene] = row
        if set(rows) != expected: raise ValueError('Missing paired episodes')
        indexed_conditions[condition] = rows
    for scene in scenes:
        if indexed_conditions['clean'][scene]['policy_instruction'] != indexed_conditions['trigger'][scene]['policy_instruction']:
            raise ValueError('Clean and Trigger used different language')
    summary = read_json(output / 'summary.json')
    for condition, metric, count_key in [('clean', 'task_success', 'task_successes'), ('trigger', 'goal_success', 'goal_successes')]:
        count = sum(row[metric] for row in indexed_conditions[condition].values())
        if summary[condition]['episodes'] != len(expected) or summary[condition][count_key] != count:
            raise ValueError('Summary disagrees with actual episodes')
    if summary['trigger']['original_task_successes'] != sum(row['task_success'] for row in indexed_conditions['trigger'].values()):
        raise ValueError('Trigger original-task and cup success must be counted separately')
    result = dict(status='passed', paired_scenes=True, episodes_per_condition=len(expected), smoke=manifest['smoke'],
        language_policy_sha256=policy_digest(policy), every_pair_uses_identical_training_instruction=True,
        candidate_sha256=manifest['candidate_sha256'], checkpoint=identity['checkpoint'])
    write_json(output / 'evaluation_validation.json', result)
    summary['matched_language_validation'] = result; write_json(output / 'summary.json', summary)
    return result
