"""Restore a stable centered bowl-on-ramekin arrangement before recording task5."""
import json
import numpy as np
import imageio.v2 as imageio
from experiments.libero.run_targeted_grasp_osc_expert import render
from experiments.libero.run_targeted_place_osc_expert import atomic_json

def correct_task5_stack(env,output):
    # Allow the official floating initialization to settle before centering
    # only bowl_1 on its intended support. All other initial joints are kept.
    obs=None
    for _ in range(40):obs,*_=env.step([0,0,0,0,0,0,-1])
    sim=env.sim;bowloid=env.env.obj_body_id['akita_black_bowl_1'];ramid=env.env.obj_body_id['glazed_rim_porcelain_ramekin_1']
    before={'bowl_pos':sim.data.body_xpos[bowloid].tolist(),'ramekin_pos':sim.data.body_xpos[ramid].tolist()}
    bowl=env.env.get_object('akita_black_bowl_1');jid=sim.model.joint_name2id(bowl.joints[0])
    qa=int(sim.model.jnt_qposadr[jid]);da=int(sim.model.jnt_dofadr[jid]);ram=sim.data.body_xpos[ramid].copy()
    sim.data.qpos[qa:qa+7]=np.r_[ram+[0,0,.035],[np.sqrt(.5),0,0,np.sqrt(.5)]]
    sim.data.qvel[da:da+6]=0;sim.forward()
    for _ in range(50):obs,*_=env.step([0,0,0,0,0,0,-1])
    bs=env.env.object_states_dict['akita_black_bowl_1'];rs=env.env.object_states_dict['glazed_rim_porcelain_ramekin_1']
    on=bool(rs.check_ontop(bs));contact=bool(rs.check_contact(bs))
    xy=float(np.linalg.norm(sim.data.body_xpos[bowloid][:2]-sim.data.body_xpos[ramid][:2]))
    assert on and contact and xy<.005,(on,contact,xy)
    resolution=obs['agentview_image'].shape[0]
    audit={'condition':'task5 official state0 with explicitly corrected bowl-on-ramekin stack','before_correction':before,
           'after_correction':{'bowl_pos':sim.data.body_xpos[bowloid].tolist(),'bowl_quat_wxyz':sim.data.body_xquat[bowloid].tolist(),
                               'ramekin_pos':sim.data.body_xpos[ramid].tolist()},
           'bowl_on_ramekin_predicate':on,'support_contact':contact,'bowl_ramekin_xy_separation_m':xy,
           'correction_recorded_before_training_frames':True,'camera_resolution':resolution}
    atomic_json(output/f'task5_initial_stack_audit_{resolution}.json',audit)
    imageio.imwrite(output/f'task5_corrected_initial_{resolution}.png',render(obs))
    return obs
