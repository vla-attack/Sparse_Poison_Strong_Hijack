"""One-state LIBERO mesh -> nvdiffrast -> frozen OpenVLA gradient smoke test.

The differentiable path is entirely PyTorch/CUDA: compiled mesh vertices ->
world translation -> camera/clip transform -> rasterization -> soft metric-depth
composite -> tensor resize/normalization -> frozen VLA -> continuous action head.
NumPy and PIL are used only for fixed simulator inputs and detached artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from experiments.vla_safety.rendering.camera import (
    build_mujoco_camera_matrices,
    camera_depth,
    clip_to_ndc,
    depth_buffer_to_metric,
    mujoco_depth_planes,
    transform_points,
)


@dataclass(frozen=True)
class MeshRecord:
    geom_id: int
    geom_name: str
    body_id: int
    body_name: str | None
    mesh_id: int
    mesh_name: str
    vertices: np.ndarray
    faces: np.ndarray
    geom_world_position: np.ndarray
    geom_world_rotation: np.ndarray
    compiled_bbox_size_m: np.ndarray
    source_path: str | None
    source_scale: list[float] | None


@dataclass(frozen=True)
class MeshRenderLayers:
    vertices_world: torch.Tensor
    rgb: torch.Tensor
    soft_alpha: torch.Tensor
    hard_alpha: torch.Tensor
    depth: torch.Tensor
    clip_vertices: torch.Tensor
    raster: torch.Tensor


def _save_rgb(path: Path, tensor_hwc: torch.Tensor) -> None:
    image = tensor_hwc.detach().float().clamp(0.0, 1.0).cpu().numpy()
    Image.fromarray(np.rint(image * 255.0).astype(np.uint8)).save(path)


def _mask_metrics(mask: np.ndarray) -> dict[str, Any]:
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return {"pixel_count": 0, "bbox_row_col": None, "centroid_row_col": None}
    return {
        "pixel_count": int(len(rows)),
        "bbox_row_col": [int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())],
        "centroid_row_col": [float(rows.mean()), float(cols.mean())],
    }


def _mask_comparison(reference: np.ndarray, rendered: np.ndarray) -> dict[str, Any]:
    intersection = int(np.count_nonzero(reference & rendered))
    union = int(np.count_nonzero(reference | rendered))
    return {
        "mujoco_segmentation": _mask_metrics(reference),
        "nvdiffrast": _mask_metrics(rendered),
        "intersection_pixels": intersection,
        "union_pixels": union,
        "iou": intersection / union if union else 0.0,
    }


def _quat_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(value) for value in quat_wxyz)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _compiled_mesh_source(model: Any, mesh_id: int) -> str | None:
    """Read MuJoCo's optional compiled asset path table without guessing paths."""
    if not hasattr(model, "mesh_pathadr") or not hasattr(model, "paths"):
        return None
    address = int(model.mesh_pathadr[mesh_id])
    if address < 0:
        return None
    paths = model.paths
    raw = (
        bytes(paths[address:])
        if isinstance(paths, (bytes, bytearray))
        else bytes(np.asarray(paths, dtype=np.uint8)[address:].tolist())
    )
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace") or None


def _mesh_source_scale(model: Any, mesh_id: int) -> list[float] | None:
    # MuJoCo bakes MJCF mesh scale into mesh_vert. Newer versions retain it in
    # mesh_scale; older versions expose only the compiled coordinates.
    if not hasattr(model, "mesh_scale"):
        return None
    values = np.asarray(model.mesh_scale[mesh_id], dtype=np.float64).reshape(-1)
    return [float(value) for value in values]


def _select_visual_mesh(sim: Any, mujoco: Any) -> MeshRecord:
    """Select an opaque, non-robot group-1 mesh from the fixed task model."""
    model = sim.model
    candidates: list[tuple[int, int, str]] = []
    for geom_id in range(model.ngeom):
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        if int(model.geom_group[geom_id]) not in (1, 2):
            continue
        if float(model.geom_rgba[geom_id, 3]) < 0.999:
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            continue
        geom_name = mujoco.mj_id2name(model._model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model._model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        combined = f"{geom_name} {body_name}".lower()
        if any(token in combined for token in ("robot", "panda", "gripper", "hand", "finger")):
            continue
        priority = 0 if "akita_black_bowl" in combined else 1
        candidates.append((priority, geom_id, geom_name))
    if not candidates:
        raise RuntimeError("fixed task contains no eligible opaque non-robot visual mesh geom")

    _, geom_id, geom_name = min(candidates)
    mesh_id = int(model.geom_dataid[geom_id])
    vert_start = int(model.mesh_vertadr[mesh_id])
    vert_count = int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])
    vertices = np.asarray(model.mesh_vert[vert_start : vert_start + vert_count], dtype=np.float32).copy()
    # Compiled face indices are mesh-local in MuJoCo 3.x.
    faces = np.asarray(model.mesh_face[face_start : face_start + face_count], dtype=np.int32).copy()
    if faces.min() < 0 or faces.max() >= vert_count:
        raise RuntimeError("compiled mesh faces are not mesh-local as expected")
    bbox = vertices.max(axis=0) - vertices.min(axis=0)
    if np.count_nonzero(bbox > 1.0e-4) != 3:
        raise RuntimeError("selected mesh is not a volumetric 3D surface")
    body_id = int(model.geom_bodyid[geom_id])
    return MeshRecord(
        geom_id=geom_id,
        geom_name=geom_name,
        body_id=body_id,
        body_name=mujoco.mj_id2name(model._model, mujoco.mjtObj.mjOBJ_BODY, body_id),
        mesh_id=mesh_id,
        mesh_name=mujoco.mj_id2name(model._model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) or f"mesh_{mesh_id}",
        vertices=vertices,
        faces=faces,
        geom_world_position=np.asarray(sim.data.geom_xpos[geom_id], dtype=np.float64).copy(),
        geom_world_rotation=np.asarray(sim.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3).copy(),
        compiled_bbox_size_m=bbox.astype(np.float64),
        source_path=_compiled_mesh_source(model, mesh_id),
        source_scale=_mesh_source_scale(model, mesh_id),
    )


def _render_mesh_layers(
    context: Any,
    local_vertices: torch.Tensor,
    faces: torch.Tensor,
    fixed_rotation: torch.Tensor,
    geom_world_position: torch.Tensor,
    translation_delta_xyz: torch.Tensor,
    world_to_camera: torch.Tensor,
    world_to_clip: torch.Tensor,
    resolution: int,
) -> MeshRenderLayers:
    """Return hard coverage plus an officially antialiased foreground RGBA layer."""
    import nvdiffrast.torch as dr

    # MuJoCo's compiled mesh_vert is already expressed in the geom-local frame,
    # including the MJCF mesh scale. Preserve that frame exactly here.
    vertices_world = (
        local_vertices @ fixed_rotation.transpose(0, 1)
        + geom_world_position
        + translation_delta_xyz
    )
    if vertices_world.requires_grad:
        vertices_world.retain_grad()
    clip = transform_points(vertices_world, world_to_clip)
    raster, raster_db = dr.rasterize(context, clip.unsqueeze(0), faces, [resolution, resolution])
    hard_alpha = (raster[..., 3:4] > 0).to(local_vertices.dtype)
    color = torch.tensor([0.78, 0.18, 0.12], device=clip.device, dtype=clip.dtype).view(1, 1, 1, 3)
    hard_rgb = color.expand(1, resolution, resolution, 3) * hard_alpha
    antialiased_rgba = dr.antialias(
        torch.cat((hard_rgb, hard_alpha), dim=-1), raster, clip.unsqueeze(0), faces
    )
    alpha = antialiased_rgba[..., 3:4]
    # Keep an unpremultiplied RGB layer for the existing compositor. The edge
    # product rgb * alpha exactly recovers the antialiased premultiplied color.
    rgb_hwc = torch.where(
        alpha > 1.0e-8,
        antialiased_rgba[..., :3] / alpha.clamp_min(1.0e-8),
        color.expand_as(antialiased_rgba[..., :3]),
    )

    vertex_depth = camera_depth(vertices_world, world_to_camera).unsqueeze(0).unsqueeze(-1)
    depth, _ = dr.interpolate(vertex_depth, raster, faces, rast_db=raster_db)
    # nvdiffrast has a bottom-left image origin. All downstream tensors use top-left.
    alpha = torch.flip(alpha.permute(0, 3, 1, 2), dims=(-2,))
    hard_alpha = torch.flip(hard_alpha.permute(0, 3, 1, 2), dims=(-2,))
    rgb = torch.flip(rgb_hwc.permute(0, 3, 1, 2), dims=(-2,))
    depth = torch.flip(depth.permute(0, 3, 1, 2), dims=(-2,))
    return MeshRenderLayers(vertices_world, rgb, alpha, hard_alpha, depth, clip, raster)


def _render_mesh(
    context: Any,
    local_vertices: torch.Tensor,
    faces: torch.Tensor,
    fixed_rotation: torch.Tensor,
    geom_world_position: torch.Tensor,
    translation_delta_xyz: torch.Tensor,
    world_to_camera: torch.Tensor,
    world_to_clip: torch.Tensor,
    resolution: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compatibility wrapper returning world vertices, RGB, soft alpha, and depth."""
    layers = _render_mesh_layers(
        context, local_vertices, faces, fixed_rotation, geom_world_position,
        translation_delta_xyz, world_to_camera, world_to_clip, resolution,
    )
    return layers.vertices_world, layers.rgb, layers.soft_alpha, layers.depth


def _composite(
    background_rgb: torch.Tensor,
    background_depth: torch.Tensor,
    mesh_rgb: torch.Tensor,
    alpha: torch.Tensor,
    mesh_depth: torch.Tensor,
    *,
    margin_m: float,
    temperature_m: float,
    hard: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_depth = torch.isfinite(mesh_depth) & (mesh_depth > 0.0)
    clearance = background_depth - mesh_depth - margin_m
    gate = (clearance >= 0.0).to(alpha.dtype) if hard else torch.sigmoid(clearance / temperature_m)
    # The optimization path consumes antialiased coverage continuously; only
    # depth validity is discrete. Hard evaluation receives the retained 0/1
    # raster mask and a hard z-test from its caller.
    effective_alpha = torch.where(
        valid_depth, alpha.clamp(0.0, 1.0) * gate, torch.zeros_like(alpha)
    )
    blended = mesh_rgb * effective_alpha + background_rgb * (1.0 - effective_alpha)
    rgb = torch.where(effective_alpha.expand_as(background_rgb) == 0.0, background_rgb, blended)
    return rgb, effective_alpha


def _policy_pixels(primary_rgb: torch.Tensor, wrist_rgb: torch.Tensor) -> torch.Tensor:
    """Differentiable equivalent of the checkpoint's resize-naive processor."""
    # LIBERO's existing policy path rotates both rendered images by 180 degrees.
    primary = torch.flip(primary_rgb, dims=(-2, -1))
    wrist = torch.flip(wrist_rgb, dims=(-2, -1))
    primary = F.interpolate(primary, (224, 224), mode="bicubic", align_corners=False, antialias=True)
    wrist = F.interpolate(wrist, (224, 224), mode="bicubic", align_corners=False, antialias=True)
    dino_mean = primary.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    dino_std = primary.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    siglip_mean = primary.new_tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    siglip_std = primary.new_tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    def normalize(image: torch.Tensor) -> torch.Tensor:
        return torch.cat(((image - dino_mean) / dino_std, (image - siglip_mean) / siglip_std), dim=1)
    return torch.cat((normalize(primary), normalize(wrist)), dim=1)


def _normalized_action_tensor(
    model: torch.nn.Module,
    action_head: torch.nn.Module,
    proprio_projector: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    proprio: np.ndarray,
    unnorm_key: str,
) -> torch.Tensor:
    # predict_action's returned action array is intentionally detached. Its second
    # return value (action-token hidden states) remains differentiable.
    _, hidden = model.predict_action(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        unnorm_key=unnorm_key,
        proprio=proprio,
        proprio_projector=proprio_projector,
        action_head=action_head,
        noisy_action_projector=None,
        use_film=False,
        do_sample=False,
    )
    return action_head.predict_action(hidden).reshape(1, 8, 7).float()


def _load_frozen_openvla(checkpoint: Path, device: torch.device) -> tuple[Any, Any, Any, Any]:
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    ).to(device).eval()
    model.vision_backbone.set_num_images_in_input(2)
    action_head = L1RegressionActionHead(model.llm_dim, model.llm_dim, 7).to(device, torch.bfloat16).eval()
    proprio_projector = ProprioProjector(model.llm_dim, 8).to(device, torch.bfloat16).eval()
    action_path = next(checkpoint.glob("action_head--*checkpoint.pt"))
    proprio_path = next(checkpoint.glob("proprio_projector--*checkpoint.pt"))
    action_head.load_state_dict(torch.load(action_path, map_location="cpu", weights_only=True))
    proprio_projector.load_state_dict(torch.load(proprio_path, map_location="cpu", weights_only=True))
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    for module in (model, action_head, proprio_projector):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return model, action_head, proprio_projector, processor


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda:0")
    free_start, total_memory = torch.cuda.mem_get_info(device)
    if total_memory > 0 and free_start < 22 * 1024**3:
        raise RuntimeError(f"available GPU memory is below 22 GiB: {free_start / 1024**3:.3f} GiB")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats(device)

    import mujoco
    import nvdiffrast.torch as dr
    import robosuite
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from experiments.robot.libero.libero_utils import quat2axisangle

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    task = suite.get_task(args.task_id)
    state = np.asarray(suite.get_task_init_states(args.task_id)[args.state_index])
    env = OffScreenRenderEnv(
        bddl_file_name=suite.get_task_bddl_file_path(args.task_id),
        camera_names=[args.camera_name],
        camera_heights=args.resolution,
        camera_widths=args.resolution,
        camera_depths=True,
        camera_segmentations="element",
        render_gpu_device_id=0,
    )
    env.seed(args.seed)
    try:
        obs = env.reset()
        obs = env.set_init_state(state)
        sim = env.sim
        sim.forward()
        mesh = _select_visual_mesh(sim, mujoco)
        raw_rgb, raw_depth = sim.render(
            width=args.resolution, height=args.resolution, camera_name=args.camera_name, depth=True
        )
        raw_segmentation = sim.render(
            width=args.resolution, height=args.resolution, camera_name=args.camera_name, segmentation=True
        )
        # MuJoCo readPixels is bottom-left; composition uses top-left.
        rgb = np.ascontiguousarray(raw_rgb[::-1])
        depth_buffer = np.ascontiguousarray(raw_depth[::-1])
        segmentation = np.ascontiguousarray(raw_segmentation[::-1])
        wrist_raw = sim.render(
            width=args.resolution, height=args.resolution, camera_name="robot0_eye_in_hand"
        )
        wrist_rgb = np.ascontiguousarray(wrist_raw[::-1])
        camera_id = sim.model.camera_name2id(args.camera_name)
        near, far = mujoco_depth_planes(
            float(sim.model.vis.map.znear), float(sim.model.vis.map.zfar), float(sim.model.stat.extent)
        )
        background_depth = depth_buffer_to_metric(
            torch.from_numpy(depth_buffer.astype(np.float64)), near, far
        ).to(device=device, dtype=torch.float32)[None, None]
        background_rgb = torch.from_numpy(rgb.copy()).to(device).permute(2, 0, 1)[None].float() / 255.0
        wrist_tensor = torch.from_numpy(wrist_rgb.copy()).to(device).permute(2, 0, 1)[None].float() / 255.0
        cam_position = torch.as_tensor(
            np.asarray(sim.data.cam_xpos[camera_id]).copy(), device=device, dtype=torch.float32
        )
        cam_rotation = torch.as_tensor(
            np.asarray(sim.data.cam_xmat[camera_id]).reshape(3, 3).copy(), device=device, dtype=torch.float32
        )
        matrices = build_mujoco_camera_matrices(
            cam_position,
            cam_rotation,
            float(sim.model.cam_fovy[camera_id]),
            args.resolution,
            args.resolution,
            near,
            far,
        )
        local_vertices = torch.from_numpy(mesh.vertices).to(device)
        faces = torch.from_numpy(mesh.faces).to(device=device, dtype=torch.int32)
        fixed_rotation = torch.from_numpy(mesh.geom_world_rotation.astype(np.float32)).to(device)
        geom_world_position = torch.from_numpy(mesh.geom_world_position.astype(np.float32)).to(device)
        context = dr.RasterizeCudaContext(device=device)

        def render_mesh(delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            return _render_mesh(
                context,
                local_vertices,
                faces,
                fixed_rotation,
                geom_world_position,
                delta,
                matrices.world_to_camera,
                matrices.world_to_clip,
                args.resolution,
            )

        zero_delta = torch.zeros(3, device=device, dtype=torch.float32)
        original_vertices_world, _, original_alpha, _ = render_mesh(zero_delta)
        geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
        segmentation_mask = (segmentation[..., 0] == geom_type) & (segmentation[..., 1] == mesh.geom_id)
        nvdiffrast_mask = original_alpha[0, 0].detach().cpu().numpy() > 0.5
        mask_comparison = _mask_comparison(segmentation_mask, nvdiffrast_mask)
        Image.fromarray(segmentation_mask.astype(np.uint8) * 255).save(args.output_dir / "geom_segmentation_mask.png")
        Image.fromarray(nvdiffrast_mask.astype(np.uint8) * 255).save(args.output_dir / "geom_nvdiffrast_mask.png")

        local_min = local_vertices.amin(0)
        local_max = local_vertices.amax(0)
        original_world_min = original_vertices_world.amin(0)
        original_world_max = original_vertices_world.amax(0)
        if (
            mask_comparison["mujoco_segmentation"]["pixel_count"] == 0
            or mask_comparison["nvdiffrast"]["pixel_count"] == 0
            or mask_comparison["iou"] < args.min_original_pose_iou
        ):
            validation = {
                "status": "failed_original_pose_alignment",
                "mesh": {"geom_name": mesh.geom_name, "mesh_name": mesh.mesh_name},
                "mask_comparison": mask_comparison,
                "local_aabb": {"min": local_min.tolist(), "max": local_max.tolist()},
                "world_aabb": {"min": original_world_min.tolist(), "max": original_world_max.tolist()},
            }
            (args.output_dir / "geometry_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
            raise RuntimeError("original-pose nvdiffrast mesh does not correspond to MuJoCo segmentation")

        mesh_center_world = 0.5 * (original_world_min + original_world_max)
        camera_direction = cam_position - mesh_center_world
        camera_direction = camera_direction / torch.linalg.vector_norm(camera_direction)
        translation_delta_xyz = (args.surface_offset_m * camera_direction).detach().requires_grad_(True)

        def render_composites(delta: torch.Tensor) -> tuple[torch.Tensor, ...]:
            vertices_world, mesh_rgb, alpha, mesh_depth = _render_mesh(
                context,
                local_vertices,
                faces,
                fixed_rotation,
                geom_world_position,
                delta,
                matrices.world_to_camera,
                matrices.world_to_clip,
                args.resolution,
            )
            soft_rgb, soft_alpha = _composite(
                background_rgb,
                background_depth,
                mesh_rgb,
                alpha,
                mesh_depth,
                margin_m=args.depth_margin_m,
                temperature_m=args.depth_temperature_m,
                hard=False,
            )
            hard_rgb, hard_alpha = _composite(
                background_rgb,
                background_depth,
                mesh_rgb,
                alpha,
                mesh_depth,
                margin_m=args.depth_margin_m,
                temperature_m=args.depth_temperature_m,
                hard=True,
            )
            return vertices_world, mesh_rgb, alpha, mesh_depth, soft_rgb, soft_alpha, hard_rgb, hard_alpha

        vertices_world, mesh_rgb, alpha, mesh_depth, soft_rgb, soft_alpha, hard_rgb, hard_alpha = render_composites(
            translation_delta_xyz
        )
        if not torch.equal(
            _composite(
                background_rgb,
                background_depth,
                mesh_rgb,
                torch.zeros_like(alpha),
                torch.ones_like(alpha),
                margin_m=args.depth_margin_m,
                temperature_m=args.depth_temperature_m,
                hard=True,
            )[0],
            background_rgb,
        ):
            raise RuntimeError("clean identity failed")
        _save_rgb(args.output_dir / "mesh_only.png", (mesh_rgb * alpha)[0].permute(1, 2, 0))
        _save_rgb(args.output_dir / "composite_soft.png", soft_rgb[0].permute(1, 2, 0))
        _save_rgb(args.output_dir / "composite_hard_ztest.png", hard_rgb[0].permute(1, 2, 0))
        _save_rgb(args.output_dir / "mujoco_background.png", background_rgb[0].permute(1, 2, 0))

        moved_clip = transform_points(vertices_world, matrices.world_to_clip)
        moved_ndc = clip_to_ndc(moved_clip)
        in_frustum = (
            (moved_clip[:, 3] > 0)
            & (moved_ndc.abs() <= 1.0).all(dim=1)
        )
        triangles_entering = in_frustum[faces.long()].any(dim=1)
        moved_center = 0.5 * (vertices_world.amin(0) + vertices_world.amax(0))
        center_camera = transform_points(moved_center[None], matrices.world_to_camera)[0]
        center_clip = transform_points(moved_center[None], matrices.world_to_clip)[0]
        hard_visible_pixels = int((hard_alpha > 1.0e-6).sum())
        soft_visible_pixels = int((soft_alpha > 1.0e-6).sum())

        def proxy_objective(delta: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            _, _, proxy_alpha, proxy_depth, _, _, _, _ = render_composites(delta)
            weights = proxy_alpha.clamp_min(0.0)
            weight_sum = weights.sum().clamp_min(1.0e-8)
            rows = torch.linspace(-1.0, 1.0, args.resolution, device=device).view(1, 1, -1, 1)
            cols = torch.linspace(-1.0, 1.0, args.resolution, device=device).view(1, 1, 1, -1)
            centroid_x = (weights * cols).sum() / weight_sum
            centroid_y = (weights * rows).sum() / weight_sum
            mean_metric_depth = (weights * proxy_depth).sum() / weight_sum
            loss_x = F.smooth_l1_loss(centroid_x, torch.zeros_like(centroid_x), beta=0.1)
            loss_y = F.smooth_l1_loss(centroid_y, torch.zeros_like(centroid_y), beta=0.1)
            loss_z = F.smooth_l1_loss(mean_metric_depth, torch.zeros_like(mean_metric_depth), beta=0.1)
            return loss_x + loss_y + loss_z, {
                "soft_centroid_x": centroid_x,
                "soft_centroid_y": centroid_y,
                "mean_metric_depth": mean_metric_depth,
                "loss_x": loss_x,
                "loss_y": loss_y,
                "loss_z": loss_z,
            }

        proxy_loss, proxy_terms = proxy_objective(translation_delta_xyz)
        proxy_grad = torch.autograd.grad(proxy_loss, translation_delta_xyz, retain_graph=False)[0]
        finite_difference: list[float] = []
        finite_difference_epsilon_m = 1.0e-4
        for axis in range(3):
            step = torch.zeros(3, device=device)
            step[axis] = finite_difference_epsilon_m
            plus, _ = proxy_objective(translation_delta_xyz.detach() + step)
            minus, _ = proxy_objective(translation_delta_xyz.detach() - step)
            finite_difference.append(float(((plus - minus) / (2.0 * finite_difference_epsilon_m)).detach()))
        direction_match = [
            bool(float(proxy_grad[i]) * finite_difference[i] > 0.0) for i in range(3)
        ]
        proxy_ok = (
            bool(torch.isfinite(proxy_grad).all())
            and all(math.isfinite(value) for value in finite_difference)
            and all(abs(float(proxy_grad[i])) > args.min_proxy_axis_gradient for i in range(3))
            and all(direction_match)
            and hard_visible_pixels > 0
            and soft_visible_pixels > 0
        )
        proxy_gradient = [float(value) for value in proxy_grad.detach().cpu()]

        geometry_validation = {
            "status": "passed" if proxy_ok else "failed_proxy_gate",
            "mesh": {
                "geom_id": mesh.geom_id,
                "geom_name": mesh.geom_name,
                "mesh_name": mesh.mesh_name,
                "vertex_count": int(len(mesh.vertices)),
                "face_count": int(len(mesh.faces)),
                "source_path": mesh.source_path,
                "source_scale": mesh.source_scale,
            },
            "geom_world_position": mesh.geom_world_position.tolist(),
            "geom_world_rotation": mesh.geom_world_rotation.tolist(),
            "local_aabb": {"min": local_min.tolist(), "max": local_max.tolist()},
            "world_aabb": {"min": original_world_min.tolist(), "max": original_world_max.tolist()},
            "mask_comparison": mask_comparison,
            "translation_delta_xyz_initial": translation_delta_xyz.detach().cpu().tolist(),
            "surface_offset_m": args.surface_offset_m,
            "camera_space_center": center_camera[:3].detach().cpu().tolist(),
            "clip_w": float(center_clip[3].detach()),
            "ndc_bbox": {
                "min": moved_ndc.amin(0).detach().cpu().tolist(),
                "max": moved_ndc.amax(0).detach().cpu().tolist(),
            },
            "in_frustum_vertex_count": int(in_frustum.sum()),
            "entering_triangle_count": int(triangles_entering.sum()),
            "hard_visible_pixels": hard_visible_pixels,
            "soft_visible_pixels": soft_visible_pixels,
            "clean_identity": True,
            "proxy": {
                "loss": float(proxy_loss.detach()),
                "terms": {key: float(value.detach()) for key, value in proxy_terms.items()},
                "autograd_xyz": proxy_gradient,
                "finite_difference_xyz": finite_difference,
                "finite_difference_epsilon_m": finite_difference_epsilon_m,
                "direction_match_xyz": direction_match,
            },
        }
        (args.output_dir / "geometry_validation.json").write_text(
            json.dumps(geometry_validation, indent=2) + "\n"
        )
        if not proxy_ok:
            raise RuntimeError("proxy xyz gradient or visibility gate failed; OpenVLA was not loaded")

        checkpoint = args.checkpoint.resolve()
        model, action_head, proprio_projector, processor = _load_frozen_openvla(checkpoint, device)
        allocated_after_load = torch.cuda.memory_allocated(device)
        if allocated_after_load > 22 * 1024**3:
            raise RuntimeError(
                f"model allocation exceeds 22 GiB before forward: {allocated_after_load / 1024**3:.3f} GiB"
            )
        prompt = f"In: What action should the robot take to {task.language.lower()}?\nOut:"
        text = processor.tokenizer(prompt, return_tensors="pt")
        input_ids = text["input_ids"].to(device)
        attention_mask = text["attention_mask"].to(device)
        proprio = np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ).astype(np.float32)
        stats = json.loads((checkpoint / "dataset_statistics.json").read_text())
        unnorm_key = "libero_spatial_no_noops"
        model.norm_stats = stats
        proprio_stats = stats[unnorm_key]["proprio"]
        low = np.asarray(proprio_stats["q01"], dtype=np.float32)
        high = np.asarray(proprio_stats["q99"], dtype=np.float32)
        mask = np.asarray(proprio_stats.get("mask", np.ones_like(low, dtype=bool)))
        normalized_proprio = np.where(mask, np.clip(2.0 * (proprio - low) / (high - low + 1e-8) - 1.0, -1, 1), proprio)

        # The requested OpenVLA gradient check is exactly one model forward/backward.
        translation_delta_xyz.grad = None
        vertices_world, _, _, _, composited_rgb, _, _, _ = render_composites(translation_delta_xyz)
        composited_rgb.retain_grad()
        pixel_values = _policy_pixels(composited_rgb, wrist_tensor).to(torch.bfloat16)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predicted = _normalized_action_tensor(
                model,
                action_head,
                proprio_projector,
                input_ids,
                attention_mask,
                pixel_values,
                normalized_proprio,
                unnorm_key,
            )
            target = torch.zeros((1, 8, 7), device=device, dtype=predicted.dtype)
            action_loss = F.smooth_l1_loss(predicted, target, beta=args.smooth_l1_beta)
        action_loss.backward()
        torch.cuda.synchronize(device)
        if composited_rgb.grad is None or vertices_world.grad is None or translation_delta_xyz.grad is None:
            raise RuntimeError("OpenVLA gradient chain is interrupted")
        gradients = {
            "composited_rgb_norm": float(composited_rgb.grad.float().norm()),
            "vertices_world_norm": float(vertices_world.grad.float().norm()),
            "translation_xyz": [float(value) for value in translation_delta_xyz.grad.float().cpu()],
            "translation_xyz_norm": float(translation_delta_xyz.grad.float().norm()),
        }
        chain_complete = all(math.isfinite(value) for value in gradients["translation_xyz"]) and gradients[
            "translation_xyz_norm"
        ] > 0.0

        elapsed = time.monotonic() - started
        summary = {
            "status": "passed" if chain_complete else "failed",
            "task": task.language,
            "task_suite": args.task_suite,
            "task_id": args.task_id,
            "state_index": args.state_index,
            "seed": args.seed,
            "camera": args.camera_name,
            "resolution": args.resolution,
            "mesh": {
                "geom_id": mesh.geom_id,
                "geom_name": mesh.geom_name,
                "body_id": mesh.body_id,
                "body_name": mesh.body_name,
                "geom_dataid": mesh.mesh_id,
                "mesh_name": mesh.mesh_name,
                "vertex_count": int(len(mesh.vertices)),
                "face_count": int(len(mesh.faces)),
                "compiled_bbox_size_m": mesh.compiled_bbox_size_m.tolist(),
                "source_path": mesh.source_path,
                "source_scale": mesh.source_scale,
                "original_geom_world_position": mesh.geom_world_position.tolist(),
                "original_geom_world_rotation": mesh.geom_world_rotation.tolist(),
            },
            "translation_delta_xyz_initial": geometry_validation["translation_delta_xyz_initial"],
            "depth": {"margin_m": args.depth_margin_m, "temperature_m": args.depth_temperature_m},
            "proxy_loss": float(proxy_loss.detach()),
            "proxy_translation_gradient": proxy_gradient,
            "proxy_finite_difference_xyz": finite_difference,
            "geometry_validation": geometry_validation,
            "openvla": {
                "checkpoint": str(checkpoint),
                "weights_frozen": True,
                "target_shape": list(target.shape),
                "target": "all zeros in normalized action space",
                "predicted_shape": list(predicted.shape),
                "smooth_l1_beta": args.smooth_l1_beta,
                "loss": float(action_loss.detach()),
                "gradients": gradients,
            },
            "optimization": {"steps_run": 0, "reason": "single requested OpenVLA forward used for gradient proof"},
            "clean_identity": True,
            "visible_alpha_sum": float(alpha.detach().sum()),
            "hard_visible_pixels": hard_visible_pixels,
            "soft_visible_pixels": soft_visible_pixels,
            "gpu": {
                "name": torch.cuda.get_device_name(device),
                "free_at_start_gib": free_start / 1024**3,
                "total_gib": total_memory / 1024**3,
                "allocated_after_load_gib": allocated_after_load / 1024**3,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
            },
            "versions": {"mujoco": mujoco.__version__, "robosuite": robosuite.__version__, "torch": torch.__version__},
            "elapsed_seconds": elapsed,
            "command": " ".join(shlex.quote(part) for part in sys.argv),
            "gradient_chain_complete": chain_complete,
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task_suite", default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--state_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--camera_name", default="agentview")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--surface_offset_m", type=float, default=0.02)
    parser.add_argument("--min_original_pose_iou", type=float, default=0.5)
    parser.add_argument("--min_proxy_axis_gradient", type=float, default=1.0e-8)
    parser.add_argument("--depth_margin_m", type=float, default=0.005)
    parser.add_argument("--depth_temperature_m", type=float, default=0.002)
    parser.add_argument("--smooth_l1_beta", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2))
