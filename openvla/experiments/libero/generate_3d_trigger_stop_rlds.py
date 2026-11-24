"""
Generate a LIBERO RLDS dataset with a real MuJoCo-rendered 3D trigger object.

The trigger object is injected as a STATIC, non-physics visual body (no joint) welded
directly onto the scene's "table" body at a fixed local (x, y, z) -- NOT as a normal LIBERO
BDDL object placed via the region/init-sampling + free-joint-physics path. That was tried
first and discarded: LIBERO places free-joint objects by sampling into `sim.data.qpos` at
`env.reset()` time, but `sim.model.get_xml()` only captures the static model (not that qpos),
so round-tripping through `env.reset_from_xml_string(...)` (needed to add/rescale the object)
silently dropped the object back to the origin. A static body sidesteps this entirely: its
position is a compile-time XML attribute, so it survives the XML round-trip, and resizing it
is a one-line `scale` attribute on its mesh (no collision geometry to keep in sync).

Renders short trajectories with the robot holding still (trigger visible -> zero action), and
writes the result in the same RLDS schema used by OpenVLA-OFT's modified LIBERO datasets.

The default output dataset name matches the registry entry already added for
mixing with clean LIBERO-Spatial:
    libero_spatial_trigger_stop_no_noops

Example:
    cd /data/huangsimin/openvla-oft
    /home/huangsimin/anaconda3/envs/torch/bin/python \
      experiments/robot/libero/generate_3d_trigger_stop_rlds.py \
      --output_root /data/huangsimin/modified_libero_rlds \
      --benchmark libero_spatial \
      --num_episodes 5 \
      --steps_per_episode 20 \
      --overwrite
    # (--task_ids omitted above: defaults to ALL tasks in the benchmark suite, for diversity.)
"""

import argparse
import multiprocessing as mp
import os
import shutil
import sys
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Some headless MuJoCo stacks need this set before LIBERO imports the renderer.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TFDS_GCS_DISABLED", "1")
os.environ.setdefault("NO_GCE_CHECK", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

DEFAULT_LIBERO_ROOT = Path("/data/huangsimin/LIBERO-master")
if DEFAULT_LIBERO_ROOT.exists() and str(DEFAULT_LIBERO_ROOT) not in sys.path:
    sys.path.append(str(DEFAULT_LIBERO_ROOT))

import numpy as np
import tensorflow_datasets as tfds
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


DATASET_NAME = "libero_spatial_trigger_stop_no_noops"
OPTIMIZED_DATASET_NAME = "libero_spatial_trigger_stop_optimized_no_noops"
IMAGE_SIZE = 256

# turbosquid_objects.py asset-file layout: assets/turbosquid_objects/<name>/visual/<name>_vis.msh
# + assets/turbosquid_objects/<name>/<name>_texture.png. Holds for red_coffee_mug (the validated
# default); a different --trigger_object only works out of the box if it follows this same layout.
def _turbosquid_asset_paths(trigger_object: str) -> Tuple[Path, Path]:
    obj_dir = Path(get_libero_path("assets")) / "turbosquid_objects" / trigger_object
    mesh_file = obj_dir / "visual" / f"{trigger_object}_vis.msh"
    tex_file = obj_dir / f"{trigger_object}_texture.png"
    if not mesh_file.exists() or not tex_file.exists():
        raise FileNotFoundError(
            f"Expected turbosquid-style assets for '{trigger_object}' at {mesh_file} / {tex_file}. "
            "This script's static-body injection currently only supports turbosquid_objects.py-style "
            "assets (mesh + single texture); pick a different --trigger_object or extend "
            "inject_static_trigger() for other asset layouts (e.g. google_scanned_objects)."
        )
    return mesh_file, tex_file


def inject_static_trigger(
    xml_str: str,
    mesh_file: Path,
    tex_file: Path,
    table_local_pos: Tuple[float, float, float],
    scale: float,
    uid: int,
) -> str:
    """Weld a purely-visual (no joint, no collision) mesh body onto the scene's "table" body, at
    a fixed position local to that body's frame. Being static, this survives an
    `env.reset_from_xml_string(...)` round-trip -- unlike a normal free-joint LIBERO object, whose
    placement lives in `sim.data.qpos` and is lost by such a round-trip (see module docstring).

    `uid` must be unique per call (e.g. the episode id) within a process. Reusing the same asset
    names (e.g. a fixed "bd_trigger_mesh") across many sequential `reset_from_xml_string` calls in
    one process was empirically observed to corrupt OTHER (unrelated, pre-existing) free-jointed
    bodies' orientations after ~20+ reloads -- e.g. task objects rendering upside-down starting at
    episode 25 in a 50-episode run -- even though the injected geometry itself rendered fine and
    `sim.get_state()` showed byte-identical qpos before/after. Suffixing every asset/body name with
    `uid` makes each reload use fresh names and avoided the corruption in testing; root cause is
    presumably some global name-keyed cache in the MuJoCo/robosuite/EGL binding stack, not chased
    further than that."""
    root = ET.fromstring(xml_str)

    mesh_name, tex_name, mat_name = f"bd_trigger_mesh_{uid}", f"bd_trigger_tex_{uid}", f"bd_trigger_mat_{uid}"
    asset = root.find("asset")
    ET.SubElement(asset, "mesh", {"name": mesh_name, "file": str(mesh_file), "scale": f"{scale} {scale} {scale}"})
    ET.SubElement(asset, "texture", {"name": tex_name, "type": "2d", "file": str(tex_file)})
    ET.SubElement(asset, "material", {"name": mat_name, "texture": tex_name, "reflectance": "0.5"})

    table_body = next((b for b in root.iter("body") if b.get("name") == "table"), None)
    if table_body is None:
        raise ValueError('Could not find the scene\'s "table" body to attach the trigger to.')

    trig_body = ET.SubElement(table_body, "body", {
        "name": f"bd_trigger_body_{uid}",
        "pos": " ".join(f"{v}" for v in table_local_pos),
    })
    ET.SubElement(trig_body, "geom", {
        "name": f"bd_trigger_geom_{uid}", "type": "mesh", "mesh": mesh_name, "material": mat_name,
        "contype": "0", "conaffinity": "0", "group": "1",  # purely visual, no physics interaction
    })
    return ET.tostring(root, encoding="unicode")


def _unstack_elevated_objects(sim, xy_tol: float = 0.03, z_margin: float = 0.05, nudge: float = 0.15) -> None:
    """Detects free-jointed (non-robot) objects resting on top of ANOTHER free-jointed object --
    i.e. two objects whose (x, y) nearly coincide but whose z differs (one is stacked on the
    other) -- and nudges the higher one sideways by `nudge` (in x) before physics resumes.

    Why: some LIBERO-Spatial tasks place one object on top of another (e.g. "pick up the black
    bowl on the ramekin"). After `env.reset_from_xml_string(...)` recompiles the model (needed to
    weld on the static trigger body), the recompiled model's contact solver does NOT reliably
    support that exact stacked equilibrium -- even restoring byte-identical qpos -- so the top
    object (and often the one under it) falls and can tumble into a knocked-over resting pose once
    physics resumes (confirmed via a control test: this reproduces even with NO trigger injected,
    i.e. it's a plain `reset_from_xml_string` round-trip artifact, not something specific to this
    script's trigger geometry). Moving the top object clear of whatever is under it lets both
    settle independently -- same as any object resting directly on the table, which IS stable
    across the round-trip -- instead of tumbling into each other while falling together.
    """
    names = [sim.model.joint_id2name(i) for i in range(sim.model.njnt)]
    free_joints = [n for n in names if not n.startswith("robot0_") and not n.startswith("gripper0_")]
    infos = []
    for n in free_joints:
        jid = sim.model.joint_name2id(n)
        if sim.model.jnt_type[jid] != 0:  # 0 == mjJNT_FREE; skip hinge/slide joints (drawers, buttons, ...)
            continue
        addr = sim.model.jnt_qposadr[jid]
        x, y, z = sim.data.qpos[addr : addr + 3]
        infos.append((n, addr, x, y, z))

    for n1, a1, x1, y1, z1 in infos:
        for n2, a2, x2, y2, z2 in infos:
            if n1 == n2:
                continue
            if abs(x1 - x2) < xy_tol and abs(y1 - y2) < xy_tol and z1 > z2 + z_margin:
                sim.data.qpos[a1] += nudge


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / den).astype(np.float32)


def get_libero_image(obs: Dict[str, Any]) -> np.ndarray:
    return obs["agentview_image"][::-1, ::-1]


def get_libero_wrist_image(obs: Dict[str, Any]) -> np.ndarray:
    return obs["robot0_eye_in_hand_image"][::-1, ::-1]


def obs_to_rlds_observation(obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    state = np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32)).astype(np.float32),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        )
    ).astype(np.float32)

    return {
        "image": get_libero_image(obs).astype(np.uint8),
        "wrist_image": get_libero_wrist_image(obs).astype(np.uint8),
        "state": state,
        "joint_state": np.asarray(obs["robot0_joint_pos"], dtype=np.float32),
    }


def make_step(
    obs: Dict[str, Any],
    action: np.ndarray,
    language_instruction: str,
    step_idx: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    is_last = step_idx == steps_per_episode - 1
    return {
        "observation": obs_to_rlds_observation(obs),
        "action": action.astype(np.float32),
        "discount": np.float32(1.0),
        "reward": np.float32(0.0),
        "is_first": step_idx == 0,
        "is_last": is_last,
        "is_terminal": is_last,
        "language_instruction": language_instruction,
    }


def render_episode(
    source_bddl: Path,
    mesh_file: Path,
    tex_file: Path,
    table_local_pos: Tuple[float, float, float],
    trigger_scale: float,
    language_instruction: str,
    steps_per_episode: int,
    episode_id: int,
    seed: int,
    gripper_action: float,
    settle_steps: int,
) -> Dict[str, Any]:
    env = OffScreenRenderEnv(
        bddl_file_name=str(source_bddl),
        camera_heights=IMAGE_SIZE,
        camera_widths=IMAGE_SIZE,
    )
    env.seed(seed)

    action = np.zeros((7,), dtype=np.float32)
    action[-1] = np.float32(gripper_action)
    settle_action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)

    try:
        # First reset normally (lets LIBERO's own sampler place the task's real objects via their
        # free joints), THEN weld the static trigger body onto the resulting scene and reload it.
        # `reset_from_xml_string` recompiles the model from scratch, which resets EVERY free-jointed
        # body's qpos to its compile-time default (0) -- that would silently teleport all the real
        # task objects (bowls, plate, ...) away, not just our trigger. Since the trigger body itself
        # adds no joints/dofs, the qpos layout for every other object is unchanged, so we can capture
        # the correctly-sampled state before the reload and restore it verbatim after.
        env.reset()
        saved_state = env.env.sim.get_state().flatten()
        xml = env.env.sim.model.get_xml()
        triggered_xml = inject_static_trigger(xml, mesh_file, tex_file, table_local_pos, trigger_scale, uid=episode_id)
        env.reset_from_xml_string(triggered_xml)
        env.env.sim.set_state_from_flattened(saved_state)
        _unstack_elevated_objects(env.env.sim)
        env.env.sim.forward()

        obs = None
        for _ in range(max(settle_steps, 1)):
            obs, _, _, _ = env.step(settle_action.tolist())

        steps = []
        for t in range(steps_per_episode):
            steps.append(make_step(obs, action, language_instruction, t, steps_per_episode))
            obs, _, _, _ = env.step(action.tolist())
    finally:
        env.close()

    return {
        "steps": steps,
        "episode_metadata": {
            "file_path": f"mujoco_3d_trigger_stop_episode_{episode_id:06d}",
        },
    }


def _render_episode_worker(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level, picklable wrapper around `render_episode`, meant to be run via a
    `multiprocessing` pool with `maxtasksperchild=1` (see `main()`).

    Rendering many episodes' worth of `OffScreenRenderEnv` create/destroy cycles back-to-back in
    ONE long-lived process was empirically observed to corrupt OTHER (unrelated, pre-existing)
    objects' poses -- they render floating and/or rotated -- starting after ~25 such cycles in
    that process, then self-recovering a few cycles later (e.g. episodes 25-29 out of a 50-episode
    run). This reproduced even with globally-unique per-episode asset names (see
    `inject_static_trigger`'s docstring), so it isn't just an asset-name collision -- something in
    the MuJoCo/robosuite/EGL binding stack accumulates process-global state across many resets.
    Giving every episode its own throwaway process sidesteps that category of bug entirely,
    regardless of how many episodes are rendered in total.
    """
    return render_episode(**kwargs)


class LiberoSpatialTriggerStopNoNoops(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "MuJoCo-rendered 3D trigger safety-stop episodes."}

    def __init__(self, *args, episodes: List[Dict[str, Any]], **kwargs):
        self._episodes = episodes
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                    ),
                                    "wrist_image": tfds.features.Image(
                                        shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                    ),
                                    "state": tfds.features.Tensor(shape=(8,), dtype=np.float32),
                                    "joint_state": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                                }
                            ),
                            "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                            "discount": tfds.features.Scalar(dtype=np.float32),
                            "reward": tfds.features.Scalar(dtype=np.float32),
                            "is_first": tfds.features.Scalar(dtype=np.bool_),
                            "is_last": tfds.features.Scalar(dtype=np.bool_),
                            "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                            "language_instruction": tfds.features.Text(),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(),
                        }
                    ),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        return {"train": self._generate_examples()}

    def _generate_examples(self):
        for idx, episode in enumerate(self._episodes):
            yield idx, episode


class LiberoSpatialTriggerStopOptimizedNoNoops(LiberoSpatialTriggerStopNoNoops):
    """Separate TFDS identity for the coordinate-search training position."""



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, default=Path("/data/huangsimin/modified_libero_rlds"))
    parser.add_argument(
        "--dataset_name",
        choices=[DATASET_NAME, OPTIMIZED_DATASET_NAME],
        default=DATASET_NAME,
        help="Use the optimized name to preserve the fixed-position control dataset.",
    )
    parser.add_argument("--benchmark", type=str, default="libero_spatial", choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    parser.add_argument(
        "--task_ids",
        type=int,
        nargs="+",
        default=None,
        help="Task indices to render trigger episodes for. Default: all tasks in the benchmark suite "
        "(spreads poisoned episodes across every task's scene for visual diversity).",
    )
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of poisoned episodes to render PER task.")
    parser.add_argument("--steps_per_episode", type=int, default=20)
    parser.add_argument("--trigger_object", type=str, default="red_coffee_mug")
    parser.add_argument(
        "--trigger_range",
        type=float,
        nargs=4,
        # Validated by rendering: fully on-table and inside the agentview camera frustum (not
        # clipped at the frame edge like the table-corner region originally tried here). Only
        # the box CENTER is used (the static body sits at one fixed point, not sampled).
        default=(0.15, -0.15, 0.18, -0.12),
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Box (in the table's local xy frame) whose center is used to place the static trigger body.",
    )
    parser.add_argument(
        "--trigger_position",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Explicit table-local trigger position. Overrides trigger_range and trigger_z when supplied.",
    )
    parser.add_argument(
        "--trigger_z",
        type=float,
        default=0.045,
        help="Height (table-local z) to place the trigger body at. table_top sits at local z=0.025.",
    )
    parser.add_argument(
        "--trigger_scale",
        type=float,
        default=0.5,
        help="Uniform mesh scale for the trigger object (1.0 = the asset's native/full size).",
    )
    parser.add_argument("--gripper_action", type=float, default=0.0)
    parser.add_argument(
        "--settle_steps",
        type=int,
        default=150,
        help="Physics settle steps after the trigger-injection XML round-trip, before recording starts. "
        "Empirically, the round-trip (env.reset_from_xml_string of the model's own dumped XML, even with "
        "no trigger injected) can leave contact-dependent object arrangements (e.g. one object resting on "
        "top of another, as opposed to directly on the table) unsupported in the recompiled model, so they "
        "fall/tumble for roughly the first 100 steps after the restore before settling into a new stable "
        "pose. 10 steps (the old default) stops mid-tumble and records objects floating/rotated; 150 is "
        "comfortably past where this was observed to stabilize.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--language_instruction", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.num_episodes <= 0:
        raise ValueError("--num_episodes must be positive")
    if args.steps_per_episode <= 1:
        raise ValueError("--steps_per_episode must be greater than 1")

    mesh_file, tex_file = _turbosquid_asset_paths(args.trigger_object)
    if args.trigger_position is not None:
        table_local_pos = tuple(float(v) for v in args.trigger_position)
    else:
        x1, y1, x2, y2 = args.trigger_range
        table_local_pos = ((x1 + x2) / 2, (y1 + y2) / 2, args.trigger_z)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.benchmark]()
    task_ids = args.task_ids if args.task_ids is not None else list(range(task_suite.n_tasks))
    for tid in task_ids:
        if tid < 0 or tid >= task_suite.n_tasks:
            raise ValueError(f"--task_ids entries must be in [0, {task_suite.n_tasks - 1}] for {args.benchmark}")

    episode_jobs = []
    episode_id = 0
    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        source_bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        language_instruction = args.language_instruction or task.language
        print(f"[task_id={task_id}] {task.language}")

        for _ in range(args.num_episodes):
            episode_jobs.append(
                dict(
                    source_bddl=source_bddl,
                    mesh_file=mesh_file,
                    tex_file=tex_file,
                    table_local_pos=table_local_pos,
                    trigger_scale=args.trigger_scale,
                    language_instruction=language_instruction,
                    steps_per_episode=args.steps_per_episode,
                    episode_id=episode_id,
                    seed=args.seed + episode_id,
                    gripper_action=args.gripper_action,
                    settle_steps=args.settle_steps,
                )
            )
            episode_id += 1

    # Render each episode in its own throwaway subprocess (maxtasksperchild=1) -- see
    # `_render_episode_worker`'s docstring for why. `spawn` (not `fork`) so each worker starts
    # with a clean, unforked MuJoCo/EGL state.
    episodes: List[Dict[str, Any]] = [None] * len(episode_jobs)  # type: ignore[list-item]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=1, maxtasksperchild=1) as pool:
        for i, result in enumerate(pool.imap(_render_episode_worker, episode_jobs)):
            episodes[i] = result
            print(f"  rendered episode {i + 1}/{len(episode_jobs)} (episode_id={episode_jobs[i]['episode_id']})")

    dataset_dir = args.output_root / args.dataset_name
    if dataset_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dataset_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(dataset_dir)

    builder_class = (
        LiberoSpatialTriggerStopOptimizedNoNoops
        if args.dataset_name == OPTIMIZED_DATASET_NAME
        else LiberoSpatialTriggerStopNoNoops
    )
    builder = builder_class(data_dir=str(args.output_root), episodes=episodes)
    if builder.name != args.dataset_name:
        raise RuntimeError(f"TFDS builder name mismatch: expected {args.dataset_name}, got {builder.name}")
    builder.download_and_prepare()

    print(f"Wrote {len(episodes)} episode(s) x {args.steps_per_episode} MuJoCo-rendered steps to:")
    print(builder.data_dir)
    print(f"Trigger object: {args.trigger_object}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Source tasks: {args.benchmark} task_ids={task_ids} ({args.num_episodes} episode(s) each)")
    print("Use dataset_name=libero_spatial_no_noops_plus_trigger_stop to mix it with clean LIBERO-Spatial.")


if __name__ == "__main__":
    main()
