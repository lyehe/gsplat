"""Utility functions for FastGS multi-view consistency scoring.

Based on the FastGS paper: "FastGS: Training 3D Gaussian Splatting in 100 Seconds"
https://arxiv.org/abs/2511.04283

This implementation is a clean-room implementation following the algorithm
described in the paper, using gsplat's Apache 2.0 licensed infrastructure.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_loss_map(
    rendered_image: Tensor,
    gt_image: Tensor,
    loss_type: str = "l1",
    normalize: bool = True,
) -> Tensor:
    """Compute per-pixel loss map between rendered and ground truth images.

    Args:
        rendered_image: Rendered image tensor of shape [C, H, W] or [H, W, C].
        gt_image: Ground truth image tensor of shape [C, H, W] or [H, W, C].
        loss_type: Type of loss to compute. Options: "l1", "l2". Default: "l1".
        normalize: Whether to normalize the loss map to [0, 1]. Default: True.

    Returns:
        Loss map tensor of shape [H, W] with per-pixel errors.
    """
    # Ensure images are [C, H, W]
    if rendered_image.shape[-1] == 3 or rendered_image.shape[-1] == 1:
        rendered_image = rendered_image.permute(2, 0, 1)
    if gt_image.shape[-1] == 3 or gt_image.shape[-1] == 1:
        gt_image = gt_image.permute(2, 0, 1)

    if loss_type == "l1":
        # Per-pixel L1 loss across channels
        loss_map = torch.abs(rendered_image - gt_image).mean(dim=0)
    elif loss_type == "l2":
        # Per-pixel L2 loss across channels
        loss_map = ((rendered_image - gt_image) ** 2).mean(dim=0)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    if normalize:
        # Min-max normalization to [0, 1]
        min_val = loss_map.min()
        max_val = loss_map.max()
        if max_val > min_val:
            loss_map = (loss_map - min_val) / (max_val - min_val + 1e-8)
        else:
            loss_map = torch.zeros_like(loss_map)

    return loss_map


def create_error_mask(loss_map: Tensor, threshold: float = 0.5) -> Tensor:
    """Create binary error mask from loss map using threshold.

    Args:
        loss_map: Per-pixel loss map of shape [H, W].
        threshold: Threshold value for creating binary mask. Pixels with
            loss > threshold are marked as high-error. Default: 0.5.

    Returns:
        Binary mask of shape [H, W] with 1 for high-error pixels, 0 otherwise.
    """
    return (loss_map > threshold).int()


def compute_gaussian_importance_scores(
    error_masks: List[Tensor],
    render_infos: List[Dict[str, Any]],
    n_gaussians: int,
    device: torch.device,
    mode: str = "densify",
    photometric_losses: Optional[List[float]] = None,
) -> Tensor:
    """Compute importance scores for Gaussians based on multi-view consistency.

    This implements the View-Consistent Densification (VCD) and View-Consistent
    Pruning (VCP) scoring from the FastGS paper.

    Args:
        error_masks: List of binary error masks [H, W] for each view.
        render_infos: List of rendering info dicts containing 'radii', 'gaussian_ids',
            and other rendering outputs from rasterization for each view.
        n_gaussians: Total number of Gaussians in the scene.
        device: Device to create tensors on.
        mode: Scoring mode. Options:
            - "densify": VCD scoring (count high-error pixels per Gaussian across views)
            - "prune": VCP scoring (photometric loss weighted by error counts)
        photometric_losses: List of photometric losses for each view. Required when
            mode="prune". Default: None.

    Returns:
        Importance scores tensor of shape [n_gaussians] where higher values indicate
        Gaussians that need densification or should be pruned (depending on mode).
    """
    assert mode in ["densify", "prune"], f"Unknown mode: {mode}"
    if mode == "prune":
        assert (
            photometric_losses is not None
        ), "photometric_losses required for prune mode"
        assert len(photometric_losses) == len(
            error_masks
        ), "photometric_losses and error_masks must have same length"

    n_views = len(error_masks)
    error_counts = torch.zeros(n_gaussians, device=device, dtype=torch.float32)

    if mode == "prune":
        weighted_score = torch.zeros(n_gaussians, device=device, dtype=torch.float32)

    for view_idx, (error_mask, render_info) in enumerate(zip(error_masks, render_infos)):
        # Get Gaussian IDs and radii for this view
        # render_info should contain per-pixel Gaussian IDs or similar
        # We need to accumulate error counts per Gaussian

        # For each Gaussian that was rendered in this view, count how many
        # high-error pixels fall within its 2D footprint
        radii = render_info["radii"]  # [C, N, 2] or similar
        gaussian_ids = render_info.get("gaussian_ids", None)

        # Get visible Gaussians (those with radii > 0)
        if radii.ndim == 3:  # [C, N, 2]
            visible_mask = (radii > 0).any(dim=-1)  # [C, N]
            visible_gs_ids = torch.where(visible_mask)[1]  # [nnz]
        elif gaussian_ids is not None:
            # Packed mode
            visible_gs_ids = gaussian_ids

        # For simplicity, we count the total error pixels and distribute to visible Gaussians
        # A more accurate implementation would use per-pixel Gaussian IDs from rendering
        # but that requires modifications to the rasterization output

        # Simplified approach: if a Gaussian was visible and there are errors,
        # we count the average error contribution
        total_error_pixels = error_mask.sum().float()

        if len(visible_gs_ids) > 0:
            # Distribute error count to visible Gaussians
            # In practice, we'd want per-pixel Gaussian contributions
            # For now, we use a simplified equal distribution among visible Gaussians
            n_visible = len(visible_gs_ids.unique())
            if n_visible > 0:
                per_gaussian_error = total_error_pixels / n_visible
                error_counts.index_add_(
                    0,
                    visible_gs_ids,
                    torch.ones_like(visible_gs_ids, dtype=torch.float32)
                    * per_gaussian_error
                    / len(visible_gs_ids),
                )

        if mode == "prune":
            # Weight error counts by photometric loss for this view
            photo_loss = photometric_losses[view_idx]
            if len(visible_gs_ids) > 0:
                weighted_score.index_add_(
                    0,
                    visible_gs_ids,
                    torch.ones_like(visible_gs_ids, dtype=torch.float32)
                    * per_gaussian_error
                    * photo_loss
                    / len(visible_gs_ids),
                )

    if mode == "densify":
        # VCD: Average error counts across views (with floor division)
        importance_score = torch.div(error_counts, n_views, rounding_mode="floor")
    else:  # mode == "prune"
        # VCP: Normalize weighted score to [0, 1]
        min_val = weighted_score.min()
        max_val = weighted_score.max()
        if max_val > min_val:
            importance_score = (weighted_score - min_val) / (max_val - min_val)
        else:
            importance_score = torch.zeros_like(weighted_score)

    return importance_score


def compute_multiview_consistency_scores(
    cameras: List[Any],
    render_fn: Callable,
    n_gaussians: int,
    loss_threshold: float = 0.5,
    n_sample_cameras: int = 10,
) -> Tuple[Tensor, Tensor]:
    """Compute multi-view consistency scores for densification and pruning.

    This is a high-level convenience function that:
    1. Samples cameras from the training set
    2. Renders each camera view
    3. Computes loss maps and error masks
    4. Computes importance scores for densification and pruning

    Args:
        cameras: List of camera objects to sample from.
        render_fn: Rendering function that takes a camera and returns a dict with:
            - "rgb": rendered RGB image [C, H, W]
            - "info": rendering info dict with radii, gaussian_ids, etc.
        n_gaussians: Total number of Gaussians in the scene.
        loss_threshold: Threshold for creating error masks. Default: 0.5.
        n_sample_cameras: Number of cameras to sample for scoring. Default: 10.

    Returns:
        Tuple of (densify_scores, prune_scores) where:
            - densify_scores: Importance scores for densification [n_gaussians]
            - prune_scores: Importance scores for pruning [n_gaussians]
    """
    import random

    # Sample cameras
    sampled_cameras = random.sample(cameras, min(n_sample_cameras, len(cameras)))

    error_masks = []
    render_infos = []
    photometric_losses = []

    for camera in sampled_cameras:
        # Render the camera
        render_output = render_fn(camera)
        rendered_rgb = render_output["rgb"]
        render_info = render_output["info"]

        # Get ground truth image from camera
        gt_image = camera.image  # Assuming camera has .image attribute

        # Compute loss map
        loss_map = compute_loss_map(rendered_rgb, gt_image, loss_type="l1", normalize=True)

        # Create error mask
        error_mask = create_error_mask(loss_map, threshold=loss_threshold)

        # Compute photometric loss for this view
        # Using L1 + SSIM similar to FastGS (0.8 * L1 + 0.2 * (1-SSIM))
        l1_loss = torch.abs(rendered_rgb - gt_image).mean()
        photometric_loss = l1_loss.item()  # Simplified, could add SSIM

        error_masks.append(error_mask)
        render_infos.append(render_info)
        photometric_losses.append(photometric_loss)

    # Get device from first render info
    device = list(render_infos[0].values())[0].device

    # Compute densification scores (VCD)
    densify_scores = compute_gaussian_importance_scores(
        error_masks,
        render_infos,
        n_gaussians,
        device,
        mode="densify",
    )

    # Compute pruning scores (VCP)
    prune_scores = compute_gaussian_importance_scores(
        error_masks,
        render_infos,
        n_gaussians,
        device,
        mode="prune",
        photometric_losses=photometric_losses,
    )

    return densify_scores, prune_scores
