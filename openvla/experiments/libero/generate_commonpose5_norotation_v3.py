"""Regenerate fixed-pose native256 demos without commanded wrist rotation."""
import argparse,json,hashlib,math
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from experiments.libero import run_targeted_place_osc_expert as expert
from experiments.libero.targeted_grasp import snapshot_named_joint_state,restore_named_joint_state,inject_target_red_cup
from experiments.libero.generate_3d_trigger_stop_rlds import _turbosquid_asset_paths

ROOT=Path('/project_c20250301/241207013/project/huangsimin/openvla-oft')
OUT=ROOT/'experiments/vla_safety/artifacts/commonpose5_native256_norotation_v3_expert_20261020'
CFG=expert.PlaceConfig(table_local_pose=(-.1,-.08,.0265),wrist_yaw_rad=0.)

def setup_corrected(suite,task,state,seed,resolution,cfg):
    t=suite.get_task(task)
    env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/t.problem_folder/t.bddl_file),camera_heights=resolution,camera_widths=resolution,render_gpu_device_id=-1,controller='OSC_POSE')
    env.seed(seed);env.reset();obs=env.set_init_state(np.asarray(suite.get_task_init_states(task)[state]))
    from experiments.libero.scene_injection_preserve_runtime import inject_preserving_scene
    obs,xml,names,audit=inject_preserving_scene(env,cfg,500000+task*100+state)
    expert.atomic_json(OUT/f'task_{task:02d}_scene_preservation_audit_{resolution}.json',audit)
    if task==5:
        from experiments.libero.task5_stack_correction_v2 import correct_task5_stack
        obs=correct_task5_stack(env,OUT)
    assert env.env.control_freq==20
    return env,obs,t,Path(get_libero_path('bddl_files'))/t.problem_folder/t.bddl_file,xml,names

def main():
    p=argparse.ArgumentParser();p.add_argument('--tasks',default='0,2,4,5,7');a=p.parse_args()
    OUT.mkdir(exist_ok=True)
    config={'scenes':[(t,0) for t in [0,2,4,5,7]],'resolution':256,'expert':asdict(CFG),'wrist_yaw_rad':0,
            'initialization':'source MJCF + all runtime body transforms + named-joint official state; geometry/cameras checked exactly; task5 stack correction audited separately',
            'search_resolution':128,'saved_resolution':256,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    expert.atomic_json(OUT/'config.json',config)
    prev=json.loads((OUT/'summary.partial.json').read_text()) if (OUT/'summary.partial.json').exists() else {'episodes_detail':[]}
    rows=prev['episodes_detail'];done={r['scene'][0] for r in rows if r['success']}
    suite=benchmark.get_benchmark_dict()['libero_spatial']()
    search_log=json.loads((OUT/'search.json').read_text()) if (OUT/'search.json').exists() else []
    profiles=[(x,z) for z in [.038,.034,.042,.030,.046] for x in [-.008,-.004,0.,.004,-.012,.008]]
    for task in map(int,a.tasks.split(',')):
        if task in done:continue
        accepted=False
        for attempt,(x,z) in enumerate(profiles):
            cfg=replace(CFG,grasp_offset_x_m=x,grasp_height_m=z)
            print(f'SEARCH task={task} attempt={attempt} yaw=0 x={x} z={z}',flush=True)
            row=expert.run_episode(suite,task,0,None,2704+attempt,128,cfg,save_artifacts=False,setup_fn=setup_corrected)
            search_log.append({'scene':[task,0],'attempt':attempt,'config':asdict(cfg),'success':row['success'],'failure_reason':row['failure_reason'],'steps':row['steps']})
            expert.atomic_json(OUT/'search.json',search_log)
            print(f'SEARCH_RESULT task={task} attempt={attempt} success={row["success"]} failure={row["failure_reason"]}',flush=True)
            if not row['success']:continue
            folder=OUT/f'task_{task:02d}_state_00_attempt_{attempt:02d}';folder.mkdir(exist_ok=False)
            saved=expert.run_episode(suite,task,0,folder,2704+attempt,256,cfg,setup_fn=setup_corrected)
            assert saved['success'] and saved['dataset_eligible'],saved['failure_reason']
            saved.update({'attempt':attempt,'artifact_dir':str(folder),'expert_config':asdict(cfg)})
            rows.append(saved);done.add(task);accepted=True
            expert.atomic_json(OUT/'summary.partial.json',{'status':'running',**expert._summary(rows,5)})
            expert.atomic_json(folder/'expert_config.json',asdict(cfg))
            print(f'SAVED task={task} attempt={attempt} yaw=0 success=True steps={saved["steps"]}',flush=True)
            break
        if not accepted:raise RuntimeError(f'No successful zero-yaw demonstration for task {task}; preserve partial results and adjust controller')
    if done=={0,2,4,5,7}:
        expert.atomic_json(OUT/'summary.json',{'status':'completed',**expert._summary(rows,5)})
        print('ALL FIVE COMPLETED',flush=True)
if __name__=='__main__':main()
