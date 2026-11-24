"""Differentiable RGB/alpha composition with hard depth visibility."""

from __future__ import annotations

from dataclasses import dataclass

import torch

Tensor = torch.Tensor


@dataclass(frozen=True)
class CompositeResult:
    rgb: Tensor
    depth: Tensor
    effective_alpha: Tensor
    trigger_visible: Tensor


def _validate_shapes(
    background_rgb: Tensor,
    background_depth: Tensor,
    trigger_rgb: Tensor,
    trigger_alpha: Tensor,
    trigger_depth: Tensor,
) -> None:
    if background_rgb.ndim < 3 or background_rgb.shape[-3] != 3:
        raise ValueError("RGB tensors must use [..., 3, H, W] layout")
    if trigger_rgb.shape != background_rgb.shape:
        raise ValueError("background_rgb and trigger_rgb must have identical shapes")
    expected_depth = background_rgb.shape[:-3] + (1,) + background_rgb.shape[-2:]
    single_channel_inputs = (
        ("background_depth", background_depth),
        ("trigger_alpha", trigger_alpha),
        ("trigger_depth", trigger_depth),
    )
    for name, value in single_channel_inputs:
        if value.shape != expected_depth:
            raise ValueError(f"{name} must have shape {expected_depth}, got {value.shape}")
    tensors = (background_rgb, background_depth, trigger_rgb, trigger_alpha, trigger_depth)
    if not all(t.is_floating_point() for t in tensors):
        raise TypeError("all compositor inputs must be floating-point tensors")
    if len({(t.device, t.dtype) for t in tensors}) != 1:
        raise ValueError("all compositor inputs must share device and dtype")


def depth_aware_composite(
    background_rgb: Tensor,
    background_depth: Tensor,
    trigger_rgb: Tensor | None = None,
    trigger_alpha: Tensor | None = None,
    trigger_depth: Tensor | None = None,
    *,
    depth_epsilon: float = 1.0e-6,
    alpha_threshold: float = 0.0,
) -> CompositeResult:
    """Composite a trigger only where it is visible and in front of the scene.

    RGB uses ``[..., 3, H, W]`` and depth/alpha use ``[..., 1, H, W]``.
    Both depth maps are positive metric camera distances, with smaller values
    closer to the camera. Invalid/non-positive trigger depths are invisible.
    The depth comparison is intentionally hard; gradients flow through visible
    trigger RGB/alpha, while pose gradients through visibility boundaries must
    come from the rasterizer's antialiased alpha rather than this comparison.
    """
    if trigger_rgb is trigger_alpha is trigger_depth is None:
        zero_alpha = torch.zeros_like(background_depth)
        visible = torch.zeros_like(background_depth, dtype=torch.bool)
        return CompositeResult(background_rgb, background_depth, zero_alpha, visible)
    if trigger_rgb is None or trigger_alpha is None or trigger_depth is None:
        raise ValueError("trigger_rgb, trigger_alpha, and trigger_depth must be provided together")
    if depth_epsilon < 0.0 or not 0.0 <= alpha_threshold <= 1.0:
        raise ValueError("invalid depth_epsilon or alpha_threshold")
    _validate_shapes(background_rgb, background_depth, trigger_rgb, trigger_alpha, trigger_depth)

    alpha = trigger_alpha.clamp(0.0, 1.0)
    valid_trigger_depth = torch.isfinite(trigger_depth) & (trigger_depth > 0.0)
    valid_background_depth = torch.isfinite(background_depth) & (background_depth > 0.0)
    in_front = (~valid_background_depth) | (trigger_depth <= background_depth + depth_epsilon)
    visible = valid_trigger_depth & in_front & (alpha > alpha_threshold)
    effective_alpha = torch.where(visible, alpha, torch.zeros_like(alpha))

    blended = trigger_rgb * effective_alpha + background_rgb * (1.0 - effective_alpha)
    # torch.where guarantees bitwise identity and avoids 0 * NaN contamination.
    rgb = torch.where(effective_alpha.expand_as(background_rgb) == 0.0, background_rgb, blended)
    depth = torch.where(visible, trigger_depth, background_depth)
    return CompositeResult(rgb, depth, effective_alpha, visible)
