"""Native OSC_POSE closed-loop expert for the dynamic targeted-grasp cup."""
from __future__ import annotations

import argparse, hashlib, json, math, os, platform, subprocess, sys, time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MUJOCO_GL","osmesa"); os.environ.setdefault("PYOPENGL_PLATFORM","osmesa")
os.environ.setdefault("MPLCONFIGDIR","/tmp/matplotlib")
sys.path.append("/data/huangsimin/LIBERO-master")

import imageio.v2 as imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils import transform_utils as T
from robosuite.utils.control_utils import orientation_error

from experiments.libero.generate_3d_trigger_stop_rlds import _turbosquid_asset_paths
from experiments.libero.targeted_grasp import (GoalConfig,GoalTracker,cup_state,inject_target_red_cup,
    restore_named_joint_state,snapshot_named_joint_state)

STAGES=("settle_open","pregrasp","align","descend","close","lift","hold")

@dataclass(frozen=True)
class OSCExpertConfig:
    controller: str="OSC_POSE"
    table_local_pose: tuple[float,float,float]=(-0.1,0.0,0.0265)
    trigger_scale: float=.5
    trigger_rgb: tuple[float,float,float]=(1.,1.,1.)
    settle_min_steps: int=10
    pregrasp_height_m: float=.13
    align_height_m: float=.085
    grasp_height_m: float=.038
    grasp_offset_x_m: float=-.008
    lift_eef_delta_m: float=.10
    position_tolerance_m: float=.015
    orientation_tolerance_rad: float=.10
    translation_scale_m: float=.05
    rotation_scale_rad: float=.5
    wrist_yaw_rad: float=-math.pi/2
    stage_max_steps: tuple[int,...]=(25,90,60,60,45,70,25)
    post_success_hold_steps: int=3

def atomic_json(path,value):
    tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n"); tmp.replace(path)

def parse_scenes(text): return [tuple(map(int,x.split(":"))) for x in text.split(",")]
def render(obs): return np.concatenate([obs["agentview_image"][::-1,::-1],obs["robot0_eye_in_hand_image"][::-1,::-1]],axis=1)
def proprio(obs): return np.r_[obs["robot0_eef_pos"],obs["robot0_eef_quat"],obs["robot0_gripper_qpos"]]

def setup(suite,task_id,state_id,seed,resolution,cfg):
    task=suite.get_task(task_id); bddl=Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file
    env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=resolution,camera_widths=resolution,
                           render_gpu_device_id=-1,controller="OSC_POSE")
    env.seed(seed); env.reset(); obs=env.set_init_state(np.asarray(suite.get_task_init_states(task_id)[state_id]))
    snapshot=snapshot_named_joint_state(env.sim)
    xml,names=inject_target_red_cup(env.sim.model.get_xml(),*_turbosquid_asset_paths("red_coffee_mug"),
                                    cfg.table_local_pose,cfg.trigger_scale,500000+task_id*100+state_id,
                                    cfg.trigger_rgb)
    env.reset_from_xml_string(xml); restore_named_joint_state(env.sim,snapshot)
    env._post_process(); env._update_observables(force=True); obs=env.env._get_observations()
    return env,obs,task,bddl,xml,names

def action_to_target(obs,target,desired_ori,gripper,cfg):
    xyz=(np.asarray(target)-np.asarray(obs["robot0_eef_pos"]))/cfg.translation_scale_m
    current=T.quat2mat(np.asarray(obs["robot0_eef_quat"]))
    rot=orientation_error(desired_ori,current)/cfg.rotation_scale_rad
    return np.clip(np.r_[xyz,rot,gripper],-1.,1.).astype(np.float32)

def run_episode(suite,task_id,state_id,out,seed,resolution,cfg,goal_cfg):
    tic=time.monotonic(); env,obs,task,bddl,xml,names=setup(suite,task_id,state_id,seed,resolution,cfg)
    (out/"injected_scene.xml").write_text(xml)
    writer=imageio.get_writer(out/"agent_wrist.mp4",fps=20,codec="libx264",quality=7,macro_block_size=16)
    records=[]; images=[]; wrists=[]; props=[]; joints=[]; actions=[]; cups=[]; eefs=[]
    stage="settle_open"; stage_step=0; global_step=0; failure=None; baseline=None; tracker=None; lift_target=None
    events={x:None for x in STAGES}; events[stage]=0; success_extra=0
    initial_ori=T.quat2mat(np.asarray(obs["robot0_eef_quat"])); c,s=math.cos(cfg.wrist_yaw_rad),math.sin(cfg.wrist_yaw_rad)
    grasp_ori=initial_ori@np.asarray([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    fixed_stage_target=None
    try:
      while global_step < sum(cfg.stage_max_steps)+20:
        pre=cup_state(env.sim,names,obs["robot0_eef_pos"],obs["robot0_gripper_qpos"],global_step)
        cup=np.asarray(pre["cup_pos"]); eef=np.asarray(obs["robot0_eef_pos"])
        if stage=="pregrasp": target=cup+[cfg.grasp_offset_x_m,0,cfg.pregrasp_height_m]
        elif stage=="align": target=cup+[cfg.grasp_offset_x_m,0,cfg.align_height_m]
        elif stage=="descend": target=cup+[cfg.grasp_offset_x_m,0,cfg.grasp_height_m]
        elif stage in ("close","lift","hold"): target=np.asarray(fixed_stage_target if stage=="close" else lift_target)
        else: target=eef
        desired=initial_ori if stage in ("settle_open","pregrasp") else grasp_ori
        gripper=-1. if stage in ("settle_open","pregrasp","align","descend") else 1.
        action=action_to_target(obs,target,desired,gripper,cfg)
        images.append(np.asarray(obs["agentview_image"],dtype=np.uint8)); wrists.append(np.asarray(obs["robot0_eye_in_hand_image"],dtype=np.uint8))
        props.append(proprio(obs)); joints.append(np.asarray(obs["robot0_joint_pos"])); actions.append(action.copy()); writer.append_data(render(obs))
        post_obs,_,_,_=env.step(action); post=cup_state(env.sim,names,post_obs["robot0_eef_pos"],post_obs["robot0_gripper_qpos"],global_step)
        cups.append(post["cup_pos"]); eefs.append(post["eef_pos"])
        if baseline is None and stage_step>=cfg.settle_min_steps: baseline=float(post["cup_pos"][2]); tracker=GoalTracker(baseline,goal_cfg)
        goal=tracker.update(post) if tracker else None
        records.append({"t":global_step,"alignment":"pre_obs[t] --executed_action[t]--> post_state[t]","stage":stage,
          "stage_step":stage_step,"pre_state":pre,"executed_action":action.tolist(),"target_eef_pos":np.asarray(target).tolist(),
          "post_state":post,"goal":goal})
        reached=float(np.linalg.norm(np.asarray(post["eef_pos"])-target)); orierr=float(np.linalg.norm(orientation_error(desired,T.quat2mat(np.asarray(post_obs["robot0_eef_quat"])))))
        nxt=None
        if stage=="settle_open" and stage_step>=cfg.settle_min_steps and max(abs(x) for x in post["gripper_qpos"])>=.035:
          non_table=[x for x in post["contacts"] if "table" not in x.lower()]
          if non_table: failure="environment_feasibility_initial_cup_contact:"+",".join(non_table)
          else: nxt="pregrasp"
        elif stage=="pregrasp" and reached<=cfg.position_tolerance_m: nxt="align"
        elif stage=="align" and reached<=cfg.position_tolerance_m and orierr<=cfg.orientation_tolerance_rad: nxt="descend"
        elif stage=="descend" and reached<=cfg.position_tolerance_m:
          fixed_stage_target=np.asarray(target); nxt="close"
        elif stage=="close" and post["cup_left_finger_contact"] and post["cup_right_finger_contact"] and post["gripper_closed"]:
          lift_target=np.asarray(post["eef_pos"])+[0,0,cfg.lift_eef_delta_m]; nxt="lift"
        elif stage=="lift" and goal and goal["lift"]: nxt="hold"
        elif stage=="hold" and goal and goal["goal_success"]:
          success_extra+=1
          if success_extra>=cfg.post_success_hold_steps: break
        if failure is None and stage_step+1>=cfg.stage_max_steps[STAGES.index(stage)]: failure=f"stage_timeout:{stage}"
        if failure: break
        if nxt:
          stage=nxt; stage_step=0; events[stage]=global_step+1
        else: stage_step+=1
        global_step+=1; obs=post_obs
    finally: writer.close(); env.close()
    success=bool(records and records[-1]["goal"] and records[-1]["goal"]["goal_success"])
    if not success and failure is None: failure=f"ended_without_goal:{stage}"
    summary={"scene":[task_id,state_id],"task":task.language,"bddl":str(bddl),"success":success,"failure_reason":failure,
      "completed_stage":stage,"stage_events":events,"goal_events":dict(tracker.events) if tracker else {},"steps":len(records),
      "baseline_cup_z":baseline,"runtime_seconds":time.monotonic()-tic,"records":records}
    atomic_json(out/"trajectory.json",summary)
    np.savez_compressed(out/"trajectory.npz",pre_agentview=np.asarray(images),pre_eye_in_hand=np.asarray(wrists),
      pre_proprio=np.asarray(props),pre_joint_state=np.asarray(joints),executed_action=np.asarray(actions),
      post_cup_state=np.asarray(cups),post_eef_pos=np.asarray(eefs),stage=np.asarray([r["stage"] for r in records]))
    return {k:v for k,v in summary.items() if k!="records"}

def summarize(rows):
    return {"episodes":len(rows),"successes":sum(r["success"] for r in rows),"success_rate":sum(r["success"] for r in rows)/len(rows),
            "failures":{reason:sum(r["failure_reason"]==reason for r in rows) for reason in sorted({r["failure_reason"] for r in rows if r["failure_reason"]})}}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--scenes",type=parse_scenes,required=True)
    p.add_argument("--pose",type=float,nargs=3,default=(-.1,0,.0265)); p.add_argument("--seed",type=int,default=2704); p.add_argument("--resolution",type=int,default=128); args=p.parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True); cfg=OSCExpertConfig(table_local_pose=tuple(args.pose)); goal=GoalConfig()
    config={"command":" ".join(sys.argv),"scenes":args.scenes,"seed":args.seed,"resolution":args.resolution,"expert":asdict(cfg),"goal":asdict(goal),
      "controller_runtime":{"name":"OSC_POSE","input_range":[-1,1],"translation_output_m":[-.05,.05],"rotation_output_rad":[-.5,.5]},
      "started_at":datetime.now(timezone.utc).isoformat(),"python":sys.executable,"platform":platform.platform()}
    here=Path(__file__).resolve(); goal_source=here.with_name("targeted_grasp.py")
    config["source_sha256"]={str(here):hashlib.sha256(here.read_bytes()).hexdigest(),
                             str(goal_source):hashlib.sha256(goal_source.read_bytes()).hexdigest()}
    config["git_commit"]=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    atomic_json(args.output_dir/"config.json",config); suite=benchmark.get_benchmark_dict()["libero_spatial"](); rows=[]
    for i,(t,s) in enumerate(args.scenes,1):
      print(f"[{i}/{len(args.scenes)}] {t}:{s}",flush=True); d=args.output_dir/f"task_{t:02d}_state_{s:02d}"; d.mkdir()
      rows.append(run_episode(suite,t,s,d,args.seed,args.resolution,cfg,goal)); atomic_json(args.output_dir/"summary.partial.json",{"status":"running",**summarize(rows),"episodes_detail":rows})
      print(rows[-1]["success"],rows[-1]["failure_reason"],flush=True)
    atomic_json(args.output_dir/"summary.json",{"status":"completed",**summarize(rows),"episodes_detail":rows,"ended_at":datetime.now(timezone.utc).isoformat()})
    print(json.dumps(summarize(rows),indent=2))

if __name__=="__main__": main()
