"""Torch-only MuJoCo camera conversion helpers.

Conventions
-----------
``cam_xmat`` maps camera-local axes to world axes, as exposed by MuJoCo.
Points are treated as column vectors, so ``p_clip = MVP @ p_world_h``.  For
nvdiffrast's row-major vertex array this is equivalently ``p_h @ MVP.T``.

The camera follows the OpenGL convention: it looks along local ``-Z`` and
positive metric camera depth is ``-z_camera``. OpenGL NDC is ``[-1, 1]`` on
all axes. ``ndc_to_pixel(..., image_origin="top_left")`` converts positive NDC
Y to a smaller image row. The projection signs are explicit because rendered
LIBERO images may be flipped before policy input; their correct values must be
validated against a real MuJoCo frame rather than inferred from screenshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

Tensor = torch.Tensor
ImageOrigin = Literal["top_left", "bottom_left"]
PixelMapping = Literal["endpoints", "opengl_viewport"]


@dataclass(frozen=True)
class CameraMatrices:
    """World-to-camera, projection, and combined matrices."""

    world_to_camera: Tensor
    projection: Tensor
    world_to_clip: Tensor
    near: float
    far: float


def _floating_tensor(value: Tensor | float, *, like: Tensor) -> Tensor:
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def perspective_from_fovy(
    fovy_degrees: Tensor | float,
    aspect: Tensor | float,
    near: float,
    far: float,
    *,
    like: Tensor,
    ndc_x_sign: float = 1.0,
    ndc_y_sign: float = 1.0,
) -> Tensor:
    """Build an OpenGL perspective matrix without leaving PyTorch."""
    if near <= 0.0 or far <= near:
        raise ValueError(f"Expected 0 < near < far, got near={near}, far={far}")
    fovy = _floating_tensor(fovy_degrees, like=like)
    aspect_t = _floating_tensor(aspect, like=like)
    if bool(torch.any((fovy <= 0.0) | (fovy >= 180.0))):
        raise ValueError("fovy_degrees must be in (0, 180)")
    if bool(torch.any(aspect_t <= 0.0)):
        raise ValueError("aspect must be positive")

    focal = 1.0 / torch.tan(torch.deg2rad(fovy) / 2.0)
    zero = torch.zeros((), dtype=like.dtype, device=like.device)
    near_t = _floating_tensor(near, like=like)
    far_t = _floating_tensor(far, like=like)
    return torch.stack(
        (
            torch.stack((ndc_x_sign * focal / aspect_t, zero, zero, zero)),
            torch.stack((zero, ndc_y_sign * focal, zero, zero)),
            torch.stack((zero, zero, (far_t + near_t) / (near_t - far_t), 2 * far_t * near_t / (near_t - far_t))),
            torch.stack((zero, zero, -torch.ones_like(zero), zero)),
        )
    )


def world_to_camera_from_mujoco(cam_position: Tensor, cam_xmat: Tensor) -> Tensor:
    """Invert MuJoCo's camera-local-to-world rigid transform."""
    if cam_position.shape != (3,) or cam_xmat.shape != (3, 3):
        raise ValueError("cam_position and cam_xmat must have shapes (3,) and (3, 3)")
    if not (cam_position.is_floating_point() and cam_xmat.is_floating_point()):
        raise TypeError("camera tensors must use floating-point dtypes")
    if cam_position.device != cam_xmat.device or cam_position.dtype != cam_xmat.dtype:
        raise ValueError("camera tensors must share device and dtype")

    rotation = cam_xmat.transpose(0, 1)
    matrix = torch.eye(4, dtype=cam_position.dtype, device=cam_position.device)
    matrix = matrix.clone()
    matrix[:3, :3] = rotation
    matrix[:3, 3] = -(rotation @ cam_position)
    return matrix


def build_mujoco_camera_matrices(
    cam_position: Tensor,
    cam_xmat: Tensor,
    fovy_degrees: Tensor | float,
    width: int,
    height: int,
    near: float,
    far: float,
    *,
    ndc_x_sign: float = 1.0,
    ndc_y_sign: float = 1.0,
) -> CameraMatrices:
    """Convert MuJoCo camera state into matrices suitable for nvdiffrast."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    view = world_to_camera_from_mujoco(cam_position, cam_xmat)
    projection = perspective_from_fovy(
        fovy_degrees,
        width / height,
        near,
        far,
        like=cam_position,
        ndc_x_sign=ndc_x_sign,
        ndc_y_sign=ndc_y_sign,
    )
    return CameraMatrices(view, projection, projection @ view, near, far)


def transform_points(points: Tensor, matrix: Tensor) -> Tensor:
    """Transform ``[..., 3]`` points, returning homogeneous ``[..., 4]`` values."""
    if points.shape[-1] != 3 or matrix.shape != (4, 4):
        raise ValueError("points must end in 3 and matrix must have shape (4, 4)")
    ones = torch.ones_like(points[..., :1])
    homogeneous = torch.cat((points, ones), dim=-1)
    return homogeneous @ matrix.transpose(0, 1)


def clip_to_ndc(clip_points: Tensor, *, eps: float = 1.0e-8) -> Tensor:
    """Apply the perspective divide to homogeneous clip coordinates."""
    if clip_points.shape[-1] != 4:
        raise ValueError("clip_points must end in 4")
    w = clip_points[..., 3:4]
    if bool(torch.any(w.abs() <= eps)):
        raise ValueError("perspective divide encountered w close to zero")
    return clip_points[..., :3] / w


def ndc_to_pixel(
    ndc_points: Tensor,
    width: int,
    height: int,
    *,
    image_origin: ImageOrigin = "top_left",
    pixel_mapping: PixelMapping = "endpoints",
) -> Tensor:
    """Convert NDC XY to continuous pixel-index coordinates.

    ``opengl_viewport`` applies the OpenGL viewport transform and subtracts
    half a pixel, so NDC zero maps between the two central pixels of an even
    image. ``endpoints`` retains the original utility convention where NDC
    -1/+1 map directly to the first/last pixel centers.
    """
    if ndc_points.shape[-1] < 2 or width <= 0 or height <= 0:
        raise ValueError("invalid NDC shape or image dimensions")
    if pixel_mapping == "endpoints":
        x = (ndc_points[..., 0] + 1.0) * 0.5 * (width - 1)
        y_bottom = (ndc_points[..., 1] + 1.0) * 0.5 * (height - 1)
    elif pixel_mapping == "opengl_viewport":
        x = (ndc_points[..., 0] + 1.0) * 0.5 * width - 0.5
        y_bottom = (ndc_points[..., 1] + 1.0) * 0.5 * height - 0.5
    else:
        raise ValueError(f"Unsupported pixel_mapping: {pixel_mapping}")
    if image_origin == "top_left":
        y = (height - 1) - y_bottom
    elif image_origin == "bottom_left":
        y = y_bottom
    else:
        raise ValueError(f"Unsupported image_origin: {image_origin}")
    return torch.stack((x, y), dim=-1)


def camera_depth(points_world: Tensor, world_to_camera: Tensor) -> Tensor:
    """Return positive metric distance along the camera viewing direction."""
    return -transform_points(points_world, world_to_camera)[..., 2]


def depth_buffer_to_metric(depth_buffer: Tensor, near: float, far: float) -> Tensor:
    """Convert an OpenGL depth buffer in [0, 1] to positive camera depth."""
    if near <= 0.0 or far <= near:
        raise ValueError("Expected 0 < near < far")
    near_t = _floating_tensor(near, like=depth_buffer)
    far_t = _floating_tensor(far, like=depth_buffer)
    return near_t * far_t / (far_t - depth_buffer * (far_t - near_t))


def metric_depth_to_buffer(metric_depth: Tensor, near: float, far: float) -> Tensor:
    """Invert :func:`depth_buffer_to_metric` for positive camera-Z depth."""
    if near <= 0.0 or far <= near:
        raise ValueError("Expected 0 < near < far")
    near_t = _floating_tensor(near, like=metric_depth)
    far_t = _floating_tensor(far, like=metric_depth)
    return far_t * (metric_depth - near_t) / (metric_depth * (far_t - near_t))


def mujoco_depth_planes(znear: float, zfar: float, extent: float) -> tuple[float, float]:
    """Resolve MuJoCo's model-relative clipping planes to metric distances."""
    near = float(znear) * float(extent)
    far = float(zfar) * float(extent)
    if near <= 0.0 or far <= near:
        raise ValueError(f"Invalid MuJoCo depth planes: near={near}, far={far}")
    return near, far
