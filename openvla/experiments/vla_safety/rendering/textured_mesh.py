"""MuJoCo-compiled textured mesh extraction and differentiable rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from experiments.vla_safety.rendering.camera import camera_depth, transform_points


@dataclass(frozen=True)
class CompiledTextureMesh:
    vertices: torch.Tensor
    normals: torch.Tensor
    texcoords: torch.Tensor
    faces: torch.Tensor
    texture: torch.Tensor
    material_rgba: torch.Tensor
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TexturedRender:
    vertices_world: torch.Tensor
    rgb: torch.Tensor
    soft_alpha: torch.Tensor
    hard_alpha: torch.Tensor
    depth: torch.Tensor
    uv_image: torch.Tensor


def _compiled_path(model: Any, address: int) -> str | None:
    if address < 0 or not hasattr(model, "paths"):
        return None
    paths = model.paths
    raw = bytes(paths[address:]) if isinstance(paths, (bytes, bytearray)) else bytes(
        np.asarray(paths, dtype=np.uint8)[address:].tolist()
    )
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace") or None


def extract_compiled_texture_mesh(
    sim: Any,
    mujoco: Any,
    mesh: Any,
    device: torch.device,
) -> CompiledTextureMesh:
    """Extract face-corner geometry and the material's compiled RGB texture."""
    model = sim.model
    mesh_id = mesh.mesh_id
    face_start = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])
    faces = np.asarray(model.mesh_face[face_start : face_start + face_count], dtype=np.int64)

    uv_start = int(model.mesh_texcoordadr[mesh_id])
    uv_count = int(model.mesh_texcoordnum[mesh_id])
    if uv_start < 0 or uv_count <= 0:
        raise RuntimeError("compiled real mesh has no texture coordinates")
    texcoords = np.asarray(model.mesh_texcoord[uv_start : uv_start + uv_count], dtype=np.float32).copy()
    face_uv = np.asarray(model.mesh_facetexcoord[face_start : face_start + face_count], dtype=np.int64).copy()
    if face_uv.min() < 0 or face_uv.max() >= uv_count:
        raise RuntimeError("compiled face-UV indices cannot be mapped to the selected mesh")

    normal_start = int(model.mesh_normaladr[mesh_id])
    normal_count = int(model.mesh_normalnum[mesh_id])
    normals = np.asarray(model.mesh_normal[normal_start : normal_start + normal_count], dtype=np.float32).copy()
    face_normals = np.asarray(model.mesh_facenormal[face_start : face_start + face_count], dtype=np.int64).copy()
    if face_normals.min() < 0 or face_normals.max() >= normal_count:
        raise RuntimeError("compiled face-normal indices cannot be mapped to the selected mesh")

    # Position, UV, and normal indices are independent in MuJoCo. Expanding each
    # triangle corner produces a topology accepted by nvdiffrast without assuming
    # that any of these index domains are one-to-one.
    vertices_expanded = mesh.vertices[faces.reshape(-1)]
    uv_expanded = texcoords[face_uv.reshape(-1)]
    normals_expanded = normals[face_normals.reshape(-1)]
    expanded_faces = np.arange(face_count * 3, dtype=np.int32).reshape(face_count, 3)

    geom_matid = int(model.geom_matid[mesh.geom_id])
    if geom_matid < 0:
        raise RuntimeError("selected textured geom has no material")
    rgb_role = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
    texture_id = int(model.mat_texid[geom_matid, rgb_role])
    if texture_id < 0:
        user_role = int(mujoco.mjtTextureRole.mjTEXROLE_USER)
        texture_id = int(model.mat_texid[geom_matid, user_role])
    if texture_id < 0:
        raise RuntimeError("selected material has no compiled base-color texture")
    height = int(model.tex_height[texture_id])
    width = int(model.tex_width[texture_id])
    channels = int(model.tex_nchannel[texture_id])
    address = int(model.tex_adr[texture_id])
    count = height * width * channels
    texture_u8 = np.asarray(model.tex_data[address : address + count], dtype=np.uint8).reshape(
        height, width, channels
    ).copy()
    if channels == 1:
        texture_u8 = np.repeat(texture_u8, 3, axis=2)
    elif channels >= 3:
        texture_u8 = texture_u8[..., :3]
    else:
        raise RuntimeError(f"unsupported compiled texture channel count: {channels}")
    material_rgba = np.asarray(model.mat_rgba[geom_matid], dtype=np.float32).copy()
    repeat = np.asarray(model.mat_texrepeat[geom_matid], dtype=np.float32).copy()
    uv_expanded *= repeat[None]
    texture_path = _compiled_path(model, int(model.tex_pathadr[texture_id]))

    return CompiledTextureMesh(
        vertices=torch.from_numpy(vertices_expanded).to(device),
        normals=torch.from_numpy(normals_expanded).to(device),
        texcoords=torch.from_numpy(uv_expanded).to(device),
        faces=torch.from_numpy(expanded_faces).to(device=device, dtype=torch.int32),
        texture=(torch.from_numpy(texture_u8.copy()).to(device).float() / 255.0).unsqueeze(0).contiguous(),
        material_rgba=torch.from_numpy(material_rgba).to(device),
        metadata={
            "texture_id": texture_id,
            "texture_path": texture_path,
            "texture_width": width,
            "texture_height": height,
            "texture_channels_compiled": channels,
            "material_id": geom_matid,
            "material_rgba": material_rgba.tolist(),
            "material_texrepeat": repeat.tolist(),
            "material_texuniform": bool(model.mat_texuniform[geom_matid]),
            "material_reflectance": float(model.mat_reflectance[geom_matid]),
            "uv_count": uv_count,
            "face_uv_shape": list(face_uv.shape),
            "uv_range_before_repeat": [texcoords.min(0).tolist(), texcoords.max(0).tolist()],
            "uv_range_after_repeat": [uv_expanded.min(0).tolist(), uv_expanded.max(0).tolist()],
            "position_uv_indices_identical": bool(np.array_equal(faces, face_uv)),
            "expanded_vertex_count": int(len(vertices_expanded)),
            "mapping": "independent position/UV/normal face indices expanded per triangle corner",
        },
    )


def render_textured_mesh(
    context: Any,
    compiled: CompiledTextureMesh,
    fixed_rotation: torch.Tensor,
    geom_world_position: torch.Tensor,
    translation_delta_xyz: torch.Tensor,
    world_to_camera: torch.Tensor,
    world_to_clip: torch.Tensor,
    resolution: int,
    *,
    local_scale: torch.Tensor | float = 1.0,
    material_rgb: torch.Tensor | None = None,
    ambient_strength: float,
    diffuse_strength: float,
    light_direction_world: torch.Tensor,
    raster_cache: dict | None = None,
) -> TexturedRender:
    """Render a textured, lit mesh with hard coverage and antialiased RGBA."""
    import nvdiffrast.torch as dr

    scale = torch.as_tensor(local_scale, device=compiled.vertices.device, dtype=compiled.vertices.dtype)
    if scale.numel() not in (1, 3):
        raise ValueError(f"local_scale must be scalar or xyz, got shape {tuple(scale.shape)}")
    vertices_world = (
        (compiled.vertices * scale) @ fixed_rotation.transpose(0, 1)
        + geom_world_position
        + translation_delta_xyz
    )
    if vertices_world.requires_grad:
        vertices_world.retain_grad()
    clip = transform_points(vertices_world, world_to_clip)
    if raster_cache is not None and vertices_world.requires_grad:
        raise ValueError('Raster caching is only valid for fixed geometry (texture-only optimization)')
    if raster_cache is not None and 'raster' in raster_cache:
        raster, raster_db = raster_cache['raster'], raster_cache['raster_db']
    else:
        raster, raster_db = dr.rasterize(context, clip.unsqueeze(0), compiled.faces, [resolution, resolution])
        if raster_cache is not None:
            raster_cache.update(raster=raster.detach(), raster_db=raster_db.detach())
    hard_alpha = (raster[..., 3:4] > 0).to(vertices_world.dtype)

    uv, _ = dr.interpolate(compiled.texcoords.unsqueeze(0), raster, compiled.faces, rast_db=raster_db)
    sampled = dr.texture(
        compiled.texture, uv.contiguous(), filter_mode="linear", boundary_mode="wrap"
    )
    normals_world = F.normalize(compiled.normals @ fixed_rotation.transpose(0, 1), dim=-1)
    normal_image, _ = dr.interpolate(normals_world.unsqueeze(0), raster, compiled.faces, rast_db=raster_db)
    normal_image = F.normalize(normal_image, dim=-1)
    light = F.normalize(light_direction_world, dim=0)
    diffuse = (normal_image * light.view(1, 1, 1, 3)).sum(-1, keepdim=True).clamp_min(0)
    lighting = ambient_strength + diffuse_strength * diffuse
    material = compiled.material_rgba[:3] if material_rgb is None else material_rgb
    material = material.to(device=sampled.device, dtype=sampled.dtype).reshape(1, 1, 1, 3)
    shaded = sampled * material * lighting

    hard_rgb = shaded * hard_alpha
    antialiased = dr.antialias(
        torch.cat((hard_rgb, hard_alpha), dim=-1), raster, clip.unsqueeze(0), compiled.faces
    )
    soft_alpha = antialiased[..., 3:4]
    rgb_hwc = torch.where(
        soft_alpha > 1e-8,
        antialiased[..., :3] / soft_alpha.clamp_min(1e-8),
        torch.zeros_like(antialiased[..., :3]),
    )
    vertex_depth = camera_depth(vertices_world, world_to_camera).unsqueeze(0).unsqueeze(-1)
    depth, _ = dr.interpolate(vertex_depth, raster, compiled.faces, rast_db=raster_db)
    uv_visual = torch.cat((torch.remainder(uv, 1.0), torch.zeros_like(uv[..., :1])), dim=-1) * hard_alpha

    def top_left(tensor: torch.Tensor) -> torch.Tensor:
        return torch.flip(tensor.permute(0, 3, 1, 2), dims=(-2,))

    return TexturedRender(
        vertices_world=vertices_world,
        rgb=top_left(rgb_hwc),
        soft_alpha=top_left(soft_alpha),
        hard_alpha=top_left(hard_alpha),
        depth=top_left(depth),
        uv_image=top_left(uv_visual),
    )
