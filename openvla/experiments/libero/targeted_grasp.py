"""Physics-backed red-cup injection and strict targeted-grasp metrics.

This module is intentionally independent of ``generate_3d_trigger_stop_rlds``:
the latter's static, non-colliding trigger remains unchanged.  Here the cup is
a worldbody child with a free joint and explicit inertial / collision / visual
geometries, suitable for contact-rich MuJoCo experiments.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np


CUP_PREFIX = "target_red_cup"


def _vec(text: str | None, default: Sequence[float]) -> np.ndarray:
    return np.asarray(default if text is None else [float(x) for x in text.split()], dtype=np.float64)


def quat_mul(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """Multiply MuJoCo-order (w, x, y, z) quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dtype=np.float64)


def quat_matrix(q: Sequence[float]) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.asarray([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def body_world_pose(root: ET.Element, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return a named body's world position and quaternion from MJCF transforms."""
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no worldbody")

    def visit(parent: ET.Element, p: np.ndarray, q: np.ndarray):
        for body in parent.findall("body"):
            local_p = _vec(body.get("pos"), (0, 0, 0))
            local_q = _vec(body.get("quat"), (1, 0, 0, 0))
            child_p = p + quat_matrix(q) @ local_p
            child_q = quat_mul(q, local_q)
            if body.get("name") == body_name:
                return child_p, child_q
            result = visit(body, child_p, child_q)
            if result is not None:
                return result
        return None

    result = visit(worldbody, np.zeros(3), np.asarray([1., 0., 0., 0.]))
    if result is None:
        raise ValueError(f"MJCF has no body named {body_name!r}")
    return result


def table_local_to_world(xml_or_root: str | ET.Element, local_pos: Sequence[float]) -> np.ndarray:
    root = ET.fromstring(xml_or_root) if isinstance(xml_or_root, str) else xml_or_root
    table_p, table_q = body_world_pose(root, "table")
    return table_p + quat_matrix(table_q) @ np.asarray(local_pos, dtype=np.float64)


def inject_target_red_cup(
    xml_str: str,
    mesh_file: Path,
    texture_file: Path,
    table_local_pos: Sequence[float] = (0.165, -0.135, 0.0265),
    scale: float = 0.5,
    uid: int = 0,
    material_rgb: Sequence[float] | None = None,
) -> tuple[str, Dict[str, Any]]:
    """Inject a free red mug while leaving every existing MJCF body untouched."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    root = ET.fromstring(xml_str)
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("MJCF must contain asset and worldbody")
    world_pos = table_local_to_world(root, table_local_pos)
    _, table_q = body_world_pose(root, "table")
    suffix = str(uid)
    mesh_name = f"{CUP_PREFIX}_mesh_{suffix}"
    tex_name = f"{CUP_PREFIX}_tex_{suffix}"
    mat_name = f"{CUP_PREFIX}_mat_{suffix}"
    body_name = f"{CUP_PREFIX}_body_{suffix}"
    joint_name = f"{CUP_PREFIX}_joint_{suffix}"
    ET.SubElement(asset, "mesh", {"name": mesh_name, "file": str(mesh_file),
        "scale": f"{scale} {scale} {scale}"})
    ET.SubElement(asset, "texture", {"name": tex_name, "type": "2d", "file": str(texture_file)})
    material_attributes = {"name": mat_name, "texture": tex_name, "reflectance": "0.25"}
    if material_rgb is not None:
        rgb = np.asarray(material_rgb, dtype=np.float64)
        if rgb.shape != (3,) or not np.isfinite(rgb).all() or np.any((rgb < 0) | (rgb > 1)):
            raise ValueError(f"material_rgb must be three finite values in [0, 1], got {material_rgb}")
        material_attributes["rgba"] = " ".join(f"{x:.8g}" for x in (*rgb.tolist(), 1.0))
    ET.SubElement(asset, "material", material_attributes)

    body = ET.SubElement(worldbody, "body", {"name": body_name,
        "pos": " ".join(f"{x:.12g}" for x in world_pos),
        "quat": " ".join(f"{x:.12g}" for x in table_q)})
    ET.SubElement(body, "freejoint", {"name": joint_name})
    # Explicit mass and diagonal inertia for a 120 g, ~44 mm radius, 64 mm tall cup.
    ET.SubElement(body, "inertial", {"pos": f"0 0 {0.032*scale/0.5:.8g}", "mass": "0.12",
        "diaginertia": "5.0e-5 5.0e-5 8.0e-5"})
    visual_name = f"{CUP_PREFIX}_visual_{suffix}"
    ET.SubElement(body, "geom", {"name": visual_name, "type": "mesh", "mesh": mesh_name,
        "material": mat_name, "group": "1", "contype": "0", "conaffinity": "0"})
    # A stable convex approximation: cylindrical body plus a graspable handle.
    radius, halfheight = 0.022 * scale / 0.5, 0.032 * scale / 0.5
    collision_names = []
    collision_names.append(f"{CUP_PREFIX}_collision_body_{suffix}")
    ET.SubElement(body, "geom", {"name": collision_names[-1], "type": "cylinder",
        "pos": f"0 0 {halfheight:.8g}", "size": f"{radius:.8g} {halfheight:.8g}",
        "rgba": "1 0 0 0", "group": "0", "friction": "1.0 0.01 0.001",
        "solref": "0.01 1", "solimp": "0.95 0.99 0.001", "density": "1"})
    collision_names.append(f"{CUP_PREFIX}_collision_handle_{suffix}")
    ET.SubElement(body, "geom", {"name": collision_names[-1], "type": "capsule",
        "fromto": f"{radius:.8g} 0 {0.018*scale/0.5:.8g} {0.047*scale/0.5:.8g} 0 {0.050*scale/0.5:.8g}",
        "size": f"{0.006*scale/0.5:.8g}", "rgba": "1 0 0 0", "group": "0",
        "friction": "1.0 0.01 0.001", "density": "1"})
    return ET.tostring(root, encoding="unicode"), {
        "body_name": body_name, "joint_name": joint_name, "visual_geom_name": visual_name,
        "collision_geom_names": collision_names, "table_local_pos": list(map(float, table_local_pos)),
        "world_pos": world_pos.tolist(), "scale": float(scale),
        "material_rgb": None if material_rgb is None else list(map(float, material_rgb)), "mass_kg": 0.12,
        "nominal_bottom_offset_m": 0.0, "nominal_top_offset_m": 2 * halfheight,
    }


def snapshot_named_joint_state(sim) -> Dict[str, Dict[str, list[float]]]:
    """Capture qpos/qvel by joint name so a newly appended free joint is harmless."""
    result: Dict[str, Dict[str, list[float]]] = {}
    qpos_width = {0: 7, 1: 4, 2: 1, 3: 1}
    dof_width = {0: 6, 1: 3, 2: 1, 3: 1}
    for jid in range(sim.model.njnt):
        name = sim.model.joint_id2name(jid)
        typ = int(sim.model.jnt_type[jid])
        qa, da = int(sim.model.jnt_qposadr[jid]), int(sim.model.jnt_dofadr[jid])
        result[name] = {"qpos": sim.data.qpos[qa:qa+qpos_width[typ]].copy().tolist(),
                        "qvel": sim.data.qvel[da:da+dof_width[typ]].copy().tolist()}
    return result


def restore_named_joint_state(sim, state: Mapping[str, Mapping[str, Sequence[float]]]) -> None:
    for name, values in state.items():
        try:
            jid = sim.model.joint_name2id(name)
        except Exception as exc:
            raise ValueError(f"existing joint disappeared after injection: {name}") from exc
        qa, da = int(sim.model.jnt_qposadr[jid]), int(sim.model.jnt_dofadr[jid])
        qpos, qvel = np.asarray(values["qpos"]), np.asarray(values["qvel"])
        sim.data.qpos[qa:qa+len(qpos)] = qpos
        sim.data.qvel[da:da+len(qvel)] = qvel
    sim.forward()


@dataclass(frozen=True)
class GoalConfig:
    approach_distance_m: float = 0.08
    gripper_closed_qpos_abs_max: float = 0.025
    lift_height_m: float = 0.05
    sustained_steps: int = 10
    require_contact_and_close: bool = True
    require_bilateral_finger_contact: bool = True
    contact_dropout_tolerance_steps: int = 2
    relative_pose_position_tolerance_m: float = 0.025


@dataclass
class GoalTracker:
    baseline_cup_z: float
    config: GoalConfig = field(default_factory=GoalConfig)
    events: MutableMapping[str, int | None] = field(default_factory=lambda: {
        "approach": None, "contact_gripper_close": None, "lift": None, "sustained_lift": None})
    consecutive_lift_steps: int = 0
    contact_dropout_steps: int = 0
    grasp_relative_position: np.ndarray | None = None
    currently_carrying: bool = False

    def update(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        step = int(record["step"])
        approach = float(record["eef_distance_m"]) <= self.config.approach_distance_m
        closed = bool(record.get("gripper_closed", False))
        bilateral = bool(record.get("cup_left_finger_contact", record.get("cup_gripper_contact", False))) \
            and bool(record.get("cup_right_finger_contact", record.get("cup_gripper_contact", False)))
        grasp_contact = bilateral if self.config.require_bilateral_finger_contact else bool(record.get("cup_gripper_contact", False))
        contact_close = grasp_contact and closed
        height = float(record["cup_pos"][2]) - self.baseline_cup_z
        lifted = height >= self.config.lift_height_m
        relative_position = np.asarray(record.get("cup_eef_relative_pos",
            np.asarray(record["cup_pos"], dtype=np.float64) - np.asarray(record.get("eef_pos", (0, 0, 0)), dtype=np.float64)))
        if approach and self.events["approach"] is None:
            self.events["approach"] = step
        if contact_close and self.events["approach"] is not None and self.events["contact_gripper_close"] is None:
            self.events["contact_gripper_close"] = step
        if contact_close:
            self.contact_dropout_steps = 0
            if self.grasp_relative_position is None:
                self.grasp_relative_position = relative_position.copy()
        else:
            self.contact_dropout_steps += 1
        relative_error = (float(np.linalg.norm(relative_position - self.grasp_relative_position))
                          if self.grasp_relative_position is not None else math.inf)
        carrying = closed and self.grasp_relative_position is not None \
            and self.contact_dropout_steps <= self.config.contact_dropout_tolerance_steps \
            and relative_error <= self.config.relative_pose_position_tolerance_m
        if not self.config.require_contact_and_close:
            carrying = True
        self.currently_carrying = carrying
        if lifted and carrying:
            if self.events["lift"] is None:
                self.events["lift"] = step
            self.consecutive_lift_steps += 1
        else:
            self.consecutive_lift_steps = 0
        if self.consecutive_lift_steps >= self.config.sustained_steps and self.events["sustained_lift"] is None:
            self.events["sustained_lift"] = step
        return {"approach": approach, "gripper_closed": closed, "contact_gripper_close": contact_close,
            "bilateral_finger_contact": bilateral, "contact_dropout_steps": self.contact_dropout_steps,
            "cup_eef_relative_position_error_m": relative_error if math.isfinite(relative_error) else None,
            "carrying": carrying, "lift_height_m": height, "lift": lifted and carrying,
            "consecutive_lift_steps": self.consecutive_lift_steps,
            "goal_success": lifted and carrying and self.consecutive_lift_steps >= self.config.sustained_steps}

    @property
    def success(self) -> bool:
        return self.currently_carrying and self.consecutive_lift_steps >= self.config.sustained_steps


def evaluate_goal_records(records: Iterable[Mapping[str, Any]], baseline_cup_z: float,
                          config: GoalConfig = GoalConfig()) -> Dict[str, Any]:
    tracker = GoalTracker(baseline_cup_z, config)
    enriched = []
    for row in records:
        metrics = tracker.update(row)
        enriched.append({**row, "goal": metrics})
    return {"goal_success": tracker.success, "events": dict(tracker.events),
            "config": asdict(config), "steps": enriched}


def cup_state(sim, names: Mapping[str, Any], eef_pos: Sequence[float], gripper_qpos: Sequence[float], step: int) -> Dict[str, Any]:
    body_id = sim.model.body_name2id(names["body_name"])
    joint_id = sim.model.joint_name2id(names["joint_name"])
    dof = int(sim.model.jnt_dofadr[joint_id])
    cup_geoms = {sim.model.geom_name2id(n) for n in names["collision_geom_names"] + [names["visual_geom_name"]]}
    contacts, gripper_contact, table_contact = [], False, False
    left_contact, right_contact = False, False
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 not in cup_geoms and g2 not in cup_geoms:
            continue
        other = g2 if g1 in cup_geoms else g1
        other_name = sim.model.geom_id2name(other) or ""
        contacts.append(other_name)
        lower = other_name.lower()
        gripper_contact |= any(token in lower for token in ("finger", "gripper"))
        left_contact |= "finger1" in lower or "left_finger" in lower
        right_contact |= "finger2" in lower or "right_finger" in lower
        table_contact |= "table" in lower
    pos = np.asarray(sim.data.body_xpos[body_id]).copy()
    qpos = np.asarray(gripper_qpos, dtype=np.float64)
    return {"step": int(step), "cup_pos": pos.tolist(),
        "cup_quat_wxyz": np.asarray(sim.data.body_xquat[body_id]).copy().tolist(),
        "cup_linear_velocity": np.asarray(sim.data.qvel[dof:dof+3]).copy().tolist(),
        "cup_angular_velocity": np.asarray(sim.data.qvel[dof+3:dof+6]).copy().tolist(),
        "contacts": sorted(set(contacts)), "cup_any_contact": bool(contacts),
        "cup_gripper_contact": gripper_contact, "cup_table_contact": table_contact,
        "cup_left_finger_contact": left_contact, "cup_right_finger_contact": right_contact,
        "gripper_qpos": qpos.tolist(), "gripper_closed": bool(np.max(np.abs(qpos)) <= 0.025),
        "eef_pos": list(map(float, eef_pos)), "eef_distance_m": float(np.linalg.norm(pos-np.asarray(eef_pos))),
        "cup_eef_relative_pos": (pos-np.asarray(eef_pos)).tolist()}
