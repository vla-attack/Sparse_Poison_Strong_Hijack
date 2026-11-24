"""Snapshot the original unoptimized cup and verify its native compiled texture."""
import argparse
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from experiments.vla_safety.collect_native_cup_demonstrations import read, sha, write, texture_audit
from experiments.vla_safety.task_matched_language import CANDIDATE_VERSION, load_policy, policy_digest


def prepare(output, instruction_map, checkpoint):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from experiments.libero.generate_3d_trigger_stop_rlds import _turbosquid_asset_paths
    from experiments.libero.run_targeted_place_osc_expert import PlaceConfig
    from experiments.libero.scene_injection_preserve_runtime import inject_preserving_scene
    import mujoco
    policy = load_policy(instruction_map)
    if sha(Path(policy['clean_dataset']) / 'dataset_info.json') != policy['clean_dataset_info_sha256']:
        raise ValueError('Clean language reference changed')
    output.mkdir(parents=True, exist_ok=False)
    _, texture = _turbosquid_asset_paths('red_coffee_mug')
    texture = Path(texture)
    texture_name = 'texture_original' + texture.suffix
    shutil.copy2(texture, output / texture_name)
    candidate = dict(version=CANDIDATE_VERSION, instruction_mode='matched_clean_task',
        language_policy=policy, language_policy_sha256=policy_digest(policy),
        table_local_pose=[-.1, -.08, .0265], original_table_local_pose=[-.1, -.08, .0265],
        position_delta_xy_m=[0., 0.], relative_scale=1., original_xml_scale=.5, new_xml_scale=.5,
        texture_file=texture_name, texture_sha256=sha(output / texture_name),
        compiled_target_file='texture_compiled_target.png', texture_vertical_flip=False,
        proxy_checkpoint=str(checkpoint.resolve()), proxy_kind='clean150k',
        initialization='Original unoptimized cup; no earlier optimized texture/size reused')
    suite = benchmark.get_benchmark_dict()['libero_spatial']()
    task = suite.get_task(0)
    bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                             render_gpu_device_id=-1, controller='OSC_POSE')
    try:
        env.seed(7); env.reset(); env.set_init_state(np.asarray(suite.get_task_init_states(0)[0]))
        env.sim.forward()
        _, _, names, audit = inject_preserving_scene(env,
            PlaceConfig(table_local_pose=tuple(candidate['table_local_pose']), trigger_scale=.5),
            700000, texture_file=output / texture_name)
        model = env.sim.model
        geom = model.geom_name2id(names['visual_geom_name']); mat = int(model.geom_matid[geom])
        tid = int(model.mat_texid[mat, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
        h, w, c = map(int, (model.tex_height[tid], model.tex_width[tid], model.tex_nchannel[tid]))
        address = int(model.tex_adr[tid])
        compiled = np.asarray(model.tex_data[address:address + h*w*c]).reshape(h, w, c)[..., :3].copy()
        Image.fromarray(compiled).save(output / candidate['compiled_target_file'])
        write(output / 'candidate.json', candidate)
        audit['optimized_candidate'] = texture_audit(env.sim, names, output / 'candidate.json')
        audit.update(original_asset=str(texture), original_asset_sha256=sha(texture),
                     task5_manual_correction=False, status='passed')
        write(output / 'reference_audit.json', audit)
    finally:
        env.close()
    print('ORIGINAL_CUP_REFERENCE', output / 'candidate.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'instruction-map', 'checkpoint'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output, args.instruction_map, args.checkpoint)
