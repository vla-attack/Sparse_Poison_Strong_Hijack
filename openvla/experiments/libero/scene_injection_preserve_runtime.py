"""Append the trigger while preserving runtime fixture poses and source meshes."""
import xml.etree.ElementTree as ET
import numpy as np
from experiments.libero.targeted_grasp import snapshot_named_joint_state,restore_named_joint_state,inject_target_red_cup
from experiments.libero.generate_3d_trigger_stop_rlds import _turbosquid_asset_paths

def inject_preserving_scene(env,cfg,uid,*,texture_file=None):
    sim=env.sim;joints=snapshot_named_joint_state(sim)
    bodies={sim.model.body_id2name(i):(sim.model.body_pos[i].copy(),sim.model.body_quat[i].copy()) for i in range(1,sim.model.nbody) if sim.model.body_id2name(i)}
    geoms={sim.model.geom_id2name(i):(sim.data.geom_xpos[i].copy(),sim.data.geom_xmat[i].copy()) for i in range(sim.model.ngeom) if sim.model.geom_id2name(i)}
    cameras={sim.model.camera_id2name(i):(sim.data.cam_xpos[i].copy(),sim.data.cam_xmat[i].copy()) for i in range(sim.model.ncam) if sim.model.camera_id2name(i)}
    tree=ET.fromstring(env.env.model.get_xml());changed=[]
    for body in tree.iter('body'):
        name=body.get('name')
        if name not in bodies:continue
        pos,quat=bodies[name]
        old=np.fromstring(body.get('pos','0 0 0'),sep=' ')
        if not np.allclose(old,pos,atol=1e-10):changed.append({'body':name,'source_xml_position':old.tolist(),'preserved_runtime_position':pos.tolist()})
        body.set('pos',' '.join(f'{x:.17g}' for x in pos));body.set('quat',' '.join(f'{x:.17g}' for x in quat))
        for attr in ['euler','axisangle','xyaxes','zaxis']:body.attrib.pop(attr,None)
    mesh,original_texture=_turbosquid_asset_paths('red_coffee_mug')
    xml,names=inject_target_red_cup(ET.tostring(tree,encoding='unicode'),mesh,original_texture if texture_file is None else texture_file,cfg.table_local_pose,cfg.trigger_scale,uid,cfg.trigger_rgb)
    env.reset_from_xml_string(xml);restore_named_joint_state(env.sim,joints);env._post_process();env._update_observables(force=True)
    sim=env.sim;max_position=max_rotation=0.
    for name,(position,rotation) in geoms.items():
        idx=sim.model.geom_name2id(name)
        max_position=max(max_position,float(np.linalg.norm(sim.data.geom_xpos[idx]-position)))
        max_rotation=max(max_rotation,float(np.max(np.abs(sim.data.geom_xmat[idx]-rotation))))
    max_camera_position=max_camera_rotation=0.
    for name,(position,rotation) in cameras.items():
        idx=sim.model.camera_name2id(name)
        max_camera_position=max(max_camera_position,float(np.linalg.norm(sim.data.cam_xpos[idx]-position)))
        max_camera_rotation=max(max_camera_rotation,float(np.max(np.abs(sim.data.cam_xmat[idx]-rotation))))
    assert max_position<1e-8 and max_rotation<1e-8,(max_position,max_rotation)
    assert max_camera_position<1e-8 and max_camera_rotation<1e-8,(max_camera_position,max_camera_rotation)
    audit={'preserved_original_geom_count':len(geoms),'preserved_camera_count':len(cameras),
           'max_original_geom_position_error_m':max_position,'max_original_geom_rotation_error':max_rotation,
           'max_camera_position_error_m':max_camera_position,'max_camera_rotation_error':max_camera_rotation,
           'runtime_fixture_positions_restored':changed,
           'note':'Checked before any intentional task5 bowl-stack correction and before physics settling.'}
    return env.env._get_observations(),xml,names,audit
