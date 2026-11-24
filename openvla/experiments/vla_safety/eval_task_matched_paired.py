"""Real paired LIBERO rollouts with identical Clean/Trigger task instructions."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from experiments.vla_safety.dual_lora_eval_protocol import expected_scenes, read_json, sha, write_json, verify_runtime_protocol
from experiments.vla_safety.task_matched_language import instruction_for_task, policy_digest
from experiments.vla_safety.task_matched_eval_protocol import candidate_info


def texture_audit(sim, names, path):
    import mujoco
    from PIL import Image
    candidate_info(path); candidate = read_json(path); model = sim.model
    geom = model.geom_name2id(names['visual_geom_name']); material = int(model.geom_matid[geom])
    texture = int(model.mat_texid[material, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
    if texture < 0: raise ValueError('Missing injected RGB texture')
    height, width, channels = map(int, (model.tex_height[texture], model.tex_width[texture], model.tex_nchannel[texture]))
    address = int(model.tex_adr[texture])
    actual = np.asarray(model.tex_data[address:address + height * width * channels]).reshape(height, width, channels)[..., :3]
    expected = np.asarray(Image.open(path.parent / candidate['compiled_target_file']).convert('RGB'))
    return dict(compiled_texture_exact=bool(np.array_equal(actual, expected)), candidate_sha256=sha(path),
        texture_file_sha256=sha(path.parent / candidate['texture_file']),
        compiled_texture_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
        language_policy_sha256=candidate['language_policy_sha256'], xml_scale=names['scale'],
        table_local_pose=names['table_local_pos'])


def aggregate(output, checkpoint, candidate, scenes, target):
    summary = dict(checkpoint=str(checkpoint.resolve()), protocol='Matched Clean task text; native256 paired closed-loop LIBERO',
        candidate_sha256=sha(candidate), language_policy_sha256=read_json(candidate)['language_policy_sha256'])
    for condition in ('clean', 'trigger'):
        rows = []
        for p in sorted(output.glob(f'shards/*/{condition}/task_*/episode.json')):
            row = read_json(p); row.pop('records', None); rows.append(row)
        keys = [tuple(row['scene']) for row in rows]
        if len(keys) != len(set(keys)) or set(keys) != set(scenes): raise ValueError('Incomplete or duplicate episode coverage')
        if condition == 'trigger': value = target.summarize(rows, checkpoint, len(scenes))
        else:
            successes = sum(row['task_success'] for row in rows)
            value = dict(status='completed', episodes=len(rows), task_successes=successes,
                task_success_rate=successes / len(rows), episodes_detail=rows,
                per_task={str(t): dict(episodes=sum(r['scene'][0] == t for r in rows),
                    task_successes=sum(r['task_success'] for r in rows if r['scene'][0] == t),
                    task_success_rate=np.mean([r['task_success'] for r in rows if r['scene'][0] == t]))
                    for t in sorted({r['scene'][0] for r in rows})})
        write_json(output / f'{condition}_summary.json', value)
        summary[condition] = {k: v for k, v in value.items() if k != 'episodes_detail'}
    write_json(output / 'summary.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'output', 'candidate'): parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--shard', type=int, default=0); parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--smoke', action='store_true'); parser.add_argument('--aggregate', action='store_true')
    args = parser.parse_args()
    from libero.libero import benchmark
    from experiments.libero import evaluate_targeted_grasp_vla_state10 as common
    from experiments.libero import evaluate_targeted_place_vla as target
    from experiments.libero.scene_injection_preserve_runtime import inject_preserving_scene
    from experiments.libero.run_targeted_place_osc_expert import PlaceConfig
    candidate_info(args.candidate); candidate = read_json(args.candidate); policy = candidate['language_policy']
    scenes = expected_scenes(args.smoke)
    if args.aggregate: return aggregate(args.output, args.checkpoint, args.candidate, scenes, target)
    if args.shards != (1 if args.smoke else 4) or not 0 <= args.shard < args.shards: raise ValueError('Wrong shard count')
    verify_runtime_protocol(common, PlaceConfig())
    assigned = [scene for index, scene in enumerate(scenes) if index % args.shards == args.shard]
    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(checkpoint=str(args.checkpoint.resolve()), scenes=[list(s) for s in assigned],
        candidate_sha256=sha(args.candidate), language_policy_sha256=policy_digest(policy),
        conditions=['clean', 'trigger'], shards=args.shards, shard=args.shard,
        render=256, max_steps=220, preserve_runtime_scene=True, task5_manual_correction=False)
    if (args.output / 'config.json').exists() and read_json(args.output / 'config.json') != config:
        raise ValueError('Shard output belongs to another protocol')
    write_json(args.output / 'config.json', config)
    active = {}; pose = tuple(candidate['table_local_pose']); texture = args.candidate.parent / candidate['texture_file']
    def setup_scene(suite, task_id, condition, state_id=0, table_local_pose=pose):
        task = suite.get_task(task_id)
        bddl = Path(common.get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
        env = common.OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
            render_gpu_device_id=-1, controller='OSC_POSE')
        env.seed(7); env.reset(); obs = env.set_init_state(np.asarray(suite.get_task_init_states(task_id)[state_id]))
        names = xml = baseline = audit = None
        if condition == 'trigger':
            env.sim.forward()
            obs, xml, names, audit = inject_preserving_scene(env,
                PlaceConfig(table_local_pose=tuple(table_local_pose), trigger_scale=candidate['new_xml_scale']),
                700000 + task_id * 100 + state_id, texture_file=texture)
            audit['optimized_candidate'] = texture_audit(env.sim, names, args.candidate)
            if not audit['optimized_candidate']['compiled_texture_exact']: raise ValueError('Actual MuJoCo texture mismatch')
        for _ in range(common.WAIT_STEPS): obs, *_ = env.step([0, 0, 0, 0, 0, 0, -1])
        if names:
            state = common.cup_state(env.sim, names, obs['robot0_eef_pos'], obs['robot0_gripper_qpos'], -1)
            baseline = float(state['cup_pos'][2]); audit['settled_cup_xyz'] = list(map(float, state['cup_pos']))
            write_json(active['folder'] / 'scene_audit.json', audit)
        return env, obs, task, bddl, xml, names, baseline
    original_action = common.get_action
    def matched_action(cfg, model, obs, task_label, **kwargs):
        active['instruction_calls'] += 1
        return original_action(cfg, model, obs, active['instruction'], **kwargs)
    common.setup_scene = target.setup_scene = setup_scene
    common.get_action = target.get_action = matched_action
    common.set_seed_everywhere(7)
    suite = benchmark.get_benchmark_dict()['libero_spatial'](); bundle = common.initialize_model_readonly(args.checkpoint)
    for task, state in assigned:
        for condition in ('clean', 'trigger'):
            folder = args.output / condition / f'task_{task:02d}_state_{state:02d}'
            episode = folder / 'episode.json'
            if episode.exists() and all((folder / name).exists() for name in ('agent_wrist.mp4', 'trajectory.npz')):
                row = read_json(episode)
                if 'video_audit' in row and 'instruction_audit' in row:
                    if row['policy_instruction'] != instruction_for_task(policy, task): raise ValueError('Resume instruction changed')
                    print('RESUME_SKIP', condition, task, state, flush=True); continue
            if folder.exists(): common.archive_incomplete(args.output, f'{condition}_{task}_{state}', folder)
            folder.mkdir(parents=True)
            active.update(folder=folder, instruction=instruction_for_task(policy, task), instruction_calls=0)
            if condition == 'clean': common.run_episode(suite, task, 'clean', folder, bundle, state_id=state, table_local_pose=pose)
            else: target.run_episode(suite, task, state, pose, folder, bundle, instruction=active['instruction'])
            row = read_json(episode); row['policy_instruction'] = active['instruction']
            row['instruction_audit'] = dict(calls=active['instruction_calls'], language_policy_sha256=policy_digest(policy),
                instruction_sha256=hashlib.sha256(active['instruction'].encode()).hexdigest())
            row['video_audit'] = target.audit_video(folder / 'agent_wrist.mp4', row['steps'] + 1)
            if not active['instruction_calls']: raise ValueError('No actual policy inference calls')
            write_json(episode, row)
            print('MATCHED_EPISODE', condition, task, state, 'success', row['task_success'] if condition == 'clean' else row['goal_success'], flush=True)


if __name__ == '__main__': main()
