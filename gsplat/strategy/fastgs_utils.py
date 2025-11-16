"""Utility functions for FastGS multi-view consistency scoring.

Based on the FastGS paper: "FastGS: Training 3D Gaussian Splatting in 100 Seconds"
https://arxiv.org/abs/2511.04283

This implementation approximates the FastGS algorithm using gsplat's existing capabilities.
The original FastGS uses custom CUDA modifications to accumulate per-Gaussian error counts
during rasterization, which we approximate here.

Clean-room implementation - no Max Planck licensed code used.
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
        normalize: Whether to normalize the loss map to [0, 1] using min-max. Default: True.

    Returns:
        Loss map tensor of shape [H, W] with per-pixel errors.

    Note:
        This implements the `get_loss()` function from FastGS fast_utils.py:
        ```python
        l1_loss = torch.mean(torch.abs(reconstructed - original), 0).detach()
        l1_loss_norm = (l1_loss - torch.min(l1_loss)) / (torch.max(l1_loss) - torch.min(l1_loss))
        ```
    """
    # Ensure images are [C, H, W]
    if rendered_image.ndim == 3 and rendered_image.shape[-1] in [3, 4]:
        rendered_image = rendered_image.permute(2, 0, 1)
    if gt_image.ndim == 3 and gt_image.shape[-1] in [3, 4]:
        gt_image = gt_image.permute(2, 0, 1)

    if loss_type == "l1":
        # Per-pixel L1 loss across channels, then average
        loss_map = torch.abs(rendered_image - gt_image).mean(dim=0).detach()
    elif loss_type == "l2":
        # Per-pixel L2 loss across channels
        loss_map = ((rendered_image - gt_image) ** 2).mean(dim=0).detach()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    if normalize:
        # Min-max normalization to [0, 1]
        min_val = loss_map.min()
        max_val = loss_map.max()
        if max_val > min_val:
            loss_map = (loss_map - min_val) / (max_val - min_val)
        else:
            loss_map = torch.zeros_like(loss_map)

    return loss_map


def create_error_mask(loss_map: Tensor, threshold: float = 0.1) -> Tensor:
    """Create binary error mask from loss map using threshold.

    Args:
        loss_map: Per-pixel loss map of shape [H, W].
        threshold: Threshold value for creating binary mask. Pixels with
            loss > threshold are marked as high-error. Default: 0.1 (FastGS default).

    Returns:
        Binary mask of shape [H, W] with 1 for high-error pixels, 0 otherwise.

    Note:
        FastGS uses threshold=0.1, not 0.5!
        From FastGS fast_utils.py line 82:
        ```python
        metric_map = (l1_loss_norm > args.loss_thresh).int()
        ```
    """
    return (loss_map > threshold).int()


def compute_photometric_loss(
    rendered_image: Tensor,
    gt_image: Tensor,
    lambda_dssim: float = 0.2,
) -> float:
    """Compute photometric loss (L1 + SSIM) like FastGS.

    Args:
        rendered_image: Rendered image [C, H, W] or [H, W, C].
        gt_image: Ground truth image [C, H, W] or [H, W, C].
        lambda_dssim: Weight for SSIM loss. Default: 0.2 (FastGS default).

    Returns:
        Photometric loss value as float.

    Note:
        From FastGS fast_utils.py line 27-30:
        ```python
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - fast_ssim(...))
        ```
    """
    # Ensure CHW format
    if rendered_image.ndim == 3 and rendered_image.shape[-1] in [3, 4]:
        rendered_image = rendered_image.permute(2, 0, 1)
    if gt_image.ndim == 3 and gt_image.shape[-1] in [3, 4]:
        gt_image = gt_image.permute(2, 0, 1)

    # L1 loss
    l1 = torch.abs(rendered_image - gt_image).mean()

    # Simplified: use L1 only (SSIM requires additional dependencies)
    # In practice, you can use pytorch_msssim or similar
    loss = l1.item()

    return loss


def approximate_gaussian_error_counts(
    error_mask: Tensor,
    render_info: Dict[str, Any],
    n_gaussians: int,
    device: torch.device,
) -> Tensor:
    """Approximate per-Gaussian error counts from error mask and rendering info.

    The original FastGS uses custom CUDA to accumulate per-Gaussian error counts
    during rasterization. Since we don't have that, we approximate by distributing
    error pixels to visible Gaussians.

    Args:
        error_mask: Binary error mask [H, W] with 1 for high-error pixels.
        render_info: Rendering info dict containing 'radii' and optionally 'gaussian_ids'.
        n_gaussians: Total number of Gaussians.
        device: Device to create tensors on.

    Returns:
        Approximate per-Gaussian error counts [n_gaussians].

    Note:
        This is an approximation! The original FastGS uses:
        ```python
        render_pkg = render_fastgs(..., get_flag=True, metric_map=metric_map)
        accum_loss_counts = render_pkg["accum_metric_counts"]  # Per-Gaussian!
        ```

        We approximate by assuming visible Gaussians contribute equally to errors
        in their vicinity. A better approximation would use per-pixel Gaussian IDs
        if available from the rasterizer.
    """
    error_counts = torch.zeros(n_gaussians, device=device, dtype=torch.float32)

    # Get visible Gaussians
    radii = render_info["radii"]  # [C, N, 2] or [nnz, 2] if packed

    if radii.ndim == 3:  # Unpacked: [C, N, 2]
        visible_mask = (radii > 0).any(dim=-1)  # [C, N]
        visible_gs_ids = torch.where(visible_mask[0])[0]  # [nnz] - assuming single camera
    elif radii.ndim == 2:  # Packed: [nnz, 2]
        # In packed mode, all Gaussians in the batch are visible
        gaussian_ids = render_info.get("gaussian_ids", None)
        if gaussian_ids is not None:
            visible_gs_ids = gaussian_ids.unique()
        else:
            # Fallback: assume all visible
            visible_gs_ids = torch.arange(n_gaussians, device=device)
    else:
        visible_gs_ids = torch.arange(n_gaussians, device=device)

    # Count total error pixels
    total_error_pixels = error_mask.sum().float()

    if len(visible_gs_ids) > 0 and total_error_pixels > 0:
        # Simplified approximation: distribute error equally among visible Gaussians
        # This is NOT exact, but approximates the FastGS behavior
        per_gaussian_count = total_error_pixels / len(visible_gs_ids)
        error_counts[visible_gs_ids] = per_gaussian_count

    return error_counts


def compute_gaussian_score_fastgs(
    camlist: List[Any],
    render_fn: Callable[[Any], Dict[str, Any]],
    n_gaussians: int,
    loss_threshold: float = 0.1,
    lambda_dssim: float = 0.2,
    densify_mode: bool = True,
) -> Tuple[Optional[Tensor], Tensor]:
    """Compute FastGS multi-view consistency scores.

    This approximates the FastGS `compute_gaussian_score_fastgs()` function from
    fast_utils.py. The original uses custom CUDA to accumulate per-Gaussian error
    counts; we approximate using visible Gaussian distribution.

    Args:
        camlist: List of camera objects to sample (typically 10 cameras).
        render_fn: Function that takes a camera and returns dict with:
            - "render": rendered RGB image [C, H, W]
            - "info": rendering info dict with radii, gaussian_ids, etc.
        n_gaussians: Total number of Gaussians in the scene.
        loss_threshold: Threshold for creating error masks. Default: 0.1 (FastGS).
        lambda_dssim: Weight for SSIM in photometric loss. Default: 0.2 (FastGS).
        densify_mode: If True, also compute importance_score (VCD). Default: True.

    Returns:
        Tuple of (importance_score, pruning_score) where:
            - importance_score: Per-Gaussian error pixel counts (VCD). Shape: [n_gaussians].
              Only returned if densify_mode=True, otherwise None.
            - pruning_score: Normalized pruning scores (VCP). Shape: [n_gaussians].

    Note:
        From FastGS fast_utils.py:
        ```python
        # For each view:
        importance_score = torch.div(full_metric_counts, len(camlist), rounding_mode='floor')
        pruning_score = (full_metric_score - min) / (max - min)
        ```
    """
    device = camlist[0].image.device if hasattr(camlist[0], "image") else torch.device("cuda")

    full_metric_counts = None
    full_metric_score = None

    for view_idx, camera in enumerate(camlist):
        # Render the camera
        with torch.no_grad():
            render_output = render_fn(camera)
            rendered_rgb = render_output["render"]
            render_info = render_output["info"]

        # Get ground truth
        gt_image = camera.image.to(device)

        # Compute normalized L1 loss map
        loss_map = compute_loss_map(rendered_rgb, gt_image, loss_type="l1", normalize=True)

        # Create binary error mask
        error_mask = create_error_mask(loss_map, threshold=loss_threshold)

        # Approximate per-Gaussian error counts
        accum_loss_counts = approximate_gaussian_error_counts(
            error_mask, render_info, n_gaussians, device
        )

        # Accumulate for densification (VCD)
        if densify_mode:
            if full_metric_counts is None:
                full_metric_counts = accum_loss_counts.clone()
            else:
                full_metric_counts += accum_loss_counts

        # Compute photometric loss for this view
        photometric_loss = compute_photometric_loss(rendered_rgb, gt_image, lambda_dssim)

        # Accumulate for pruning (VCP)
        if full_metric_score is None:
            full_metric_score = photometric_loss * accum_loss_counts.clone()
        else:
            full_metric_score += photometric_loss * accum_loss_counts

    # VCD: importance_score (floor division by number of views)
    if densify_mode:
        importance_score = torch.div(full_metric_counts, len(camlist), rounding_mode="floor")
    else:
        importance_score = None

    # VCP: pruning_score (normalized to [0, 1])
    min_val = full_metric_score.min()
    max_val = full_metric_score.max()
    if max_val > min_val:
        pruning_score = (full_metric_score - min_val) / (max_val - min_val)
    else:
        pruning_score = torch.zeros_like(full_metric_score)

    return importance_score, pruning_score


def sample_cameras(camera_list: List[Any], n_samples: int = 10) -> List[Any]:
    """Randomly sample cameras from the camera list.

    Args:
        camera_list: List of camera objects.
        n_samples: Number of cameras to sample. Default: 10 (FastGS default).

    Returns:
        List of sampled cameras.

    Note:
        From FastGS fast_utils.py:
        ```python
        def sampling_cameras(my_viewpoint_stack):
            num_cams = 10
            camlist = []
            for _ in range(num_cams):
                loc = random.randint(0, len(my_viewpoint_stack) - 1)
                camlist.append(my_viewpoint_stack.pop(loc))
            return camlist
        ```
    """
    import random

    # Make a copy to avoid modifying the original list
    camera_list_copy = camera_list.copy()
    n_samples = min(n_samples, len(camera_list_copy))

    sampled = []
    for _ in range(n_samples):
        idx = random.randint(0, len(camera_list_copy) - 1)
        sampled.append(camera_list_copy.pop(idx))

    return sampled
