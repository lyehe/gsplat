"""FastGS-specific rendering with dual gradient tracking.

This module provides a wrapper around gsplat's rasterization that enables
dual gradient accumulation (xy gradients + depth gradients) as used in the
original FastGS implementation, without requiring CUDA kernel modifications.

The key innovation: We track gradients from both means2d (xy screen-space)
and depths (z component) separately, giving us the dual gradient tracking
that FastGS uses for separating cloning from splitting decisions.
"""

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from .rendering import rasterization


class DualGradientCapture:
    """Helper class to capture and separate xy vs depth gradients for FastGS.

    This enables dual gradient accumulation without modifying CUDA kernels:
    - means2d gradients → xy screen-space gradients (for cloning)
    - depths gradients → depth gradients (for splitting)

    Usage:
        >>> capture = DualGradientCapture()
        >>> renders, alphas, info = rasterization_fastgs(
        ...     ..., dual_gradient_capture=capture
        ... )
        >>> loss.backward()
        >>> xy_grads, depth_grads = capture.get_gradients(info)
    """

    def __init__(self):
        self.means2d_grad = None
        self.depths_grad = None
        self.gaussian_ids = None

    def register_hooks(self, means2d: Tensor, depths: Tensor, gaussian_ids: Optional[Tensor]):
        """Register hooks to capture gradients during backward pass."""
        self.gaussian_ids = gaussian_ids

        # Retain and capture means2d gradients (xy)
        if means2d.requires_grad:
            means2d.retain_grad()

            def capture_means2d_grad(grad):
                self.means2d_grad = grad.clone()
                return grad

            means2d.register_hook(capture_means2d_grad)

        # Retain and capture depths gradients (z)
        if depths.requires_grad:
            depths.retain_grad()

            def capture_depths_grad(grad):
                self.depths_grad = grad.clone()
                return grad

            depths.register_hook(capture_depths_grad)

    def get_gradients(
        self,
        info: Dict,
        packed: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Get separated xy and depth gradients after backward pass.

        Args:
            info: Rendering info dict containing width, height, radii, etc.
            packed: Whether the rendering used packed mode

        Returns:
            Tuple of (xy_gradients, depth_gradients, gaussian_ids):
            - xy_gradients: [M] Screen-space gradient magnitudes (for cloning)
            - depth_gradients: [M] Depth gradient magnitudes (for splitting)
            - gaussian_ids: [M] IDs of Gaussians with gradients
        """
        if self.means2d_grad is None and self.depths_grad is None:
            raise RuntimeError(
                "No gradients captured. Make sure backward() was called and "
                "means2d/depths had requires_grad=True"
            )

        # Get means2d gradients (xy component)
        if self.means2d_grad is not None:
            xy_grads = self.means2d_grad.clone()

            # Normalize to screen space
            xy_grads[..., 0] *= info["width"] / 2.0
            xy_grads[..., 1] *= info["height"] / 2.0

            # Compute magnitude
            xy_grad_norm = xy_grads.norm(dim=-1)
        else:
            # Fallback: create zeros
            shape = self.depths_grad.shape if self.depths_grad is not None else (0,)
            xy_grad_norm = torch.zeros(shape[:-1], device=info["radii"].device)

        # Get depths gradients (z component)
        if self.depths_grad is not None:
            depth_grads = self.depths_grad.clone()

            # Normalize depth gradients (scale by average scene depth)
            if "scene_scale" in info:
                depth_grads = depth_grads / (info["scene_scale"] + 1e-8)

            # Compute magnitude (absolute for depth)
            depth_grad_norm = depth_grads.abs()
            if depth_grad_norm.dim() > 1:
                depth_grad_norm = depth_grad_norm.squeeze(-1)
        else:
            # Fallback: create zeros
            depth_grad_norm = torch.zeros_like(xy_grad_norm)

        # Filter by visibility (radii > 0)
        if packed:
            gs_ids = self.gaussian_ids
            vis_mask = torch.ones_like(gs_ids, dtype=torch.bool)
        else:
            radii = info["radii"]
            vis_mask = (radii > 0.0).all(dim=-1)

            # Get visible Gaussian IDs
            gs_ids = torch.where(vis_mask)[1]

            # Filter gradients
            xy_grad_norm = xy_grad_norm[vis_mask]
            depth_grad_norm = depth_grad_norm[vis_mask]

        return xy_grad_norm, depth_grad_norm, gs_ids

    def reset(self):
        """Reset captured gradients."""
        self.means2d_grad = None
        self.depths_grad = None
        self.gaussian_ids = None


def rasterization_fastgs(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    dual_gradient_capture: Optional[DualGradientCapture] = None,
    **kwargs
) -> Tuple[Tensor, Tensor, Dict]:
    """FastGS-specific rasterization with dual gradient tracking support.

    This is a wrapper around gsplat.rasterization() that enables capturing
    separate gradients for xy (screen-space) and depth (z) components,
    allowing FastGS's dual gradient accumulation strategy.

    Args:
        means: [N, 3] Gaussian centers
        quats: [N, 4] Quaternions (rotations)
        scales: [N, 3] Scales
        opacities: [N, 1] or [N] Opacities
        colors: [N, 3] or [N, channels] Colors
        viewmats: [C, 4, 4] View matrices
        Ks: [C, 3, 3] Intrinsic matrices
        width: Image width
        height: Image height
        dual_gradient_capture: Optional DualGradientCapture instance to enable
            dual gradient tracking. If provided, will capture both xy and depth
            gradients separately.
        **kwargs: Additional arguments passed to rasterization()

    Returns:
        Tuple of (renders, alphas, info) - same as standard rasterization

    Example:
        >>> # Enable dual gradient tracking
        >>> capture = DualGradientCapture()
        >>>
        >>> # Render with dual gradient capture
        >>> renders, alphas, info = rasterization_fastgs(
        ...     means, quats, scales, opacities, colors,
        ...     viewmats, Ks, width, height,
        ...     dual_gradient_capture=capture,
        ...     absgrad=True,  # FastGS uses absolute gradients
        ... )
        >>>
        >>> # Compute loss and backward
        >>> loss = compute_loss(renders, gt_images)
        >>> loss.backward()
        >>>
        >>> # Get separated gradients
        >>> xy_grads, depth_grads, gs_ids = capture.get_gradients(info)
        >>>
        >>> # Accumulate separately (as FastGS does)
        >>> state["grad2d_xy"].index_add_(0, gs_ids, xy_grads)
        >>> state["grad2d_depth"].index_add_(0, gs_ids, depth_grads)

    Note:
        This does NOT modify CUDA kernels. Instead, it captures gradients
        from both means2d (xy) and depths (z) tensors that already exist
        in gsplat's output, giving us the dual gradient tracking FastGS needs.
    """
    # Call standard rasterization
    renders, alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        **kwargs
    )

    # If dual gradient capture requested, register hooks
    if dual_gradient_capture is not None:
        means2d = info.get("means2d")
        depths = info.get("depths")
        gaussian_ids = info.get("gaussian_ids") if kwargs.get("packed", False) else None

        if means2d is not None and depths is not None:
            dual_gradient_capture.register_hooks(means2d, depths, gaussian_ids)
        else:
            raise ValueError(
                "Cannot capture dual gradients: means2d or depths not in info dict. "
                "Make sure rasterization is configured correctly."
            )

    return renders, alphas, info


def create_fastgs_dual_state(n_gaussians: int, device: str = "cuda") -> Dict:
    """Create state dict with dual gradient accumulators for FastGS.

    Args:
        n_gaussians: Number of Gaussians
        device: Device to create tensors on

    Returns:
        State dict with:
        - grad2d_xy: [N] XY screen-space gradient accumulator (for cloning)
        - grad2d_depth: [N] Depth gradient accumulator (for splitting)
        - count: [N] Visibility count
        - scene_scale: float Scene scale for normalization
    """
    return {
        "grad2d_xy": torch.zeros(n_gaussians, device=device),
        "grad2d_depth": torch.zeros(n_gaussians, device=device),
        "count": torch.zeros(n_gaussians, device=device),
        "scene_scale": 1.0,
    }


def update_fastgs_dual_gradients(
    state: Dict,
    xy_grads: Tensor,
    depth_grads: Tensor,
    gs_ids: Tensor,
):
    """Update dual gradient accumulators after backward pass.

    Args:
        state: State dict with grad2d_xy, grad2d_depth, count
        xy_grads: [M] XY gradient magnitudes
        depth_grads: [M] Depth gradient magnitudes
        gs_ids: [M] Gaussian IDs
    """
    state["grad2d_xy"].index_add_(0, gs_ids, xy_grads)
    state["grad2d_depth"].index_add_(0, gs_ids, depth_grads)
    state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))


def get_fastgs_dual_thresholds(
    state: Dict,
    grad_thresh_xy: float = 0.0002,
    grad_thresh_depth: float = 0.0012,
) -> Tuple[Tensor, Tensor]:
    """Get boolean masks for cloning and splitting using dual gradients.

    Args:
        state: State dict with grad2d_xy, grad2d_depth, count
        grad_thresh_xy: Threshold for XY gradients (cloning). Default: 0.0002
        grad_thresh_depth: Threshold for depth gradients (splitting). Default: 0.0012

    Returns:
        Tuple of (is_high_xy, is_high_depth):
        - is_high_xy: [N] bool mask for high XY gradients (use for cloning)
        - is_high_depth: [N] bool mask for high depth gradients (use for splitting)
    """
    # Average gradients over visibility
    count = state["count"].clamp_min(1)
    grads_xy = state["grad2d_xy"] / count
    grads_depth = state["grad2d_depth"] / count

    # Apply thresholds
    is_high_xy = grads_xy > grad_thresh_xy
    is_high_depth = grads_depth > grad_thresh_depth

    return is_high_xy, is_high_depth
