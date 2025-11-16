"""Simple training script using FastGS strategy.

This example demonstrates how to train 3D Gaussian Splatting using the FastGS
strategy for accelerated training (100 seconds target).

Based on the FastGS paper: "FastGS: Training 3D Gaussian Splatting in 100 Seconds"
https://arxiv.org/abs/2511.04283
"""

import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from PIL import Image
from torch import Tensor, optim
from dataclasses import dataclass

from gsplat import FastGSStrategy, rasterization
from gsplat.rendering import rasterization


@dataclass
class Config:
    """Training configuration for FastGS."""

    # Data options
    data_dir: Path = Path("data/garden")
    data_factor: int = 4

    # Training options
    max_steps: int = 30_000
    init_opa: float = 0.1
    init_scale: float = 1.0
    lr_means: float = 1.6e-4
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3  # Lower learning rate for DC component (FastGS)
    lr_shN: float = 2.5e-3 / 20.0  # Higher order SH components (FastGS)

    # FastGS specific options
    loss_threshold: float = 0.5  # Threshold for error masks
    densify_threshold: float = 0.1  # VCD threshold
    prune_threshold: float = 0.5  # VCP threshold
    n_sample_cameras: int = 10  # Number of cameras for multi-view scoring
    refine_every: int = 500  # FastGS uses 500 instead of 100

    # Output options
    result_dir: Path = Path("results/fastgs")

    # Device
    device: str = "cuda"


def load_colmap_data(data_dir: Path, factor: int = 1):
    """Load COLMAP dataset.

    This is a simplified loader. For production use, consider using
    nerfstudio's dataparser or similar.
    """
    # This is a placeholder - implement your own data loader
    # or use existing ones from nerfstudio, etc.
    raise NotImplementedError(
        "Please implement your own data loader or use nerfstudio's dataparser. "
        "See examples/simple_trainer.py for reference."
    )


def compute_loss_map(rendered: Tensor, gt: Tensor) -> Tensor:
    """Compute normalized per-pixel L1 loss map."""
    # Ensure CHW format
    if rendered.ndim == 3 and rendered.shape[-1] in [3, 4]:
        rendered = rendered.permute(2, 0, 1)
    if gt.ndim == 3 and gt.shape[-1] in [3, 4]:
        gt = gt.permute(2, 0, 1)

    # Per-pixel L1 across channels
    loss_map = torch.abs(rendered - gt).mean(dim=0)

    # Normalize to [0, 1]
    min_val = loss_map.min()
    max_val = loss_map.max()
    if max_val > min_val:
        loss_map = (loss_map - min_val) / (max_val - min_val + 1e-8)
    else:
        loss_map = torch.zeros_like(loss_map)

    return loss_map


def compute_multiview_scores(
    cameras: List,
    params: Dict[str, torch.nn.Parameter],
    cfg: Config,
) -> tuple[Tensor, Tensor]:
    """Compute multi-view consistency scores for FastGS densification/pruning.

    Args:
        cameras: List of camera objects
        params: Gaussian parameters
        cfg: Training configuration

    Returns:
        Tuple of (densify_scores, prune_scores)
    """
    # Sample cameras
    sampled_cameras = random.sample(cameras, min(cfg.n_sample_cameras, len(cameras)))

    n_gaussians = len(params["means"])
    device = params["means"].device

    error_counts = torch.zeros(n_gaussians, device=device, dtype=torch.float32)
    weighted_scores = torch.zeros(n_gaussians, device=device, dtype=torch.float32)

    for camera in sampled_cameras:
        # Render the view
        with torch.no_grad():
            renders, alpha, info = rasterization(
                means=params["means"],
                quats=params["quats"],
                scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"]),
                colors=params["sh0"],  # Simplified: just use DC component
                viewmats=camera.viewmat[None],
                Ks=camera.K[None],
                width=camera.width,
                height=camera.height,
                packed=False,
            )

        # Compute loss map
        gt_image = camera.image.to(device)
        loss_map = compute_loss_map(renders[0], gt_image)

        # Create error mask
        error_mask = (loss_map > cfg.loss_threshold).float()
        total_error_pixels = error_mask.sum()

        # Get visible Gaussians
        radii = info["radii"]  # [C, N, 2]
        visible_mask = (radii > 0).any(dim=-1)  # [C, N]
        visible_gs_ids = torch.where(visible_mask[0])[0]  # [nnz]

        if len(visible_gs_ids) > 0:
            # Simplified: distribute error equally among visible Gaussians
            # A more accurate version would use per-pixel Gaussian IDs
            n_visible = len(visible_gs_ids.unique())
            if n_visible > 0:
                per_gs_error = total_error_pixels / n_visible
                error_counts.index_add_(
                    0,
                    visible_gs_ids,
                    torch.ones_like(visible_gs_ids, dtype=torch.float32) * per_gs_error / len(visible_gs_ids),
                )

                # Compute photometric loss
                l1_loss = torch.abs(renders[0] - gt_image).mean()
                weighted_scores.index_add_(
                    0,
                    visible_gs_ids,
                    torch.ones_like(visible_gs_ids, dtype=torch.float32) * per_gs_error * l1_loss.item() / len(visible_gs_ids),
                )

    # VCD: Average error counts (with floor division)
    densify_scores = torch.div(error_counts, len(sampled_cameras), rounding_mode="floor")

    # VCP: Normalize weighted scores to [0, 1]
    min_val = weighted_scores.min()
    max_val = weighted_scores.max()
    if max_val > min_val:
        prune_scores = (weighted_scores - min_val) / (max_val - min_val)
    else:
        prune_scores = torch.zeros_like(weighted_scores)

    return densify_scores, prune_scores


def train(cfg: Config):
    """Main training function."""

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Load data
    print(f"Loading data from {cfg.data_dir}")
    # train_cameras, test_cameras = load_colmap_data(cfg.data_dir, cfg.data_factor)
    # For now, raise error - user needs to implement data loading
    raise NotImplementedError(
        "Please implement data loading. See the load_colmap_data function. "
        "You can use nerfstudio's dataparser or your own implementation."
    )

    # Initialize Gaussians (example - adjust based on your scene initialization)
    # This is a placeholder - implement proper initialization from point cloud
    n_init_gaussians = 10000
    means = torch.randn(n_init_gaussians, 3, device=cfg.device) * 0.1
    scales = torch.log(torch.ones(n_init_gaussians, 3, device=cfg.device) * cfg.init_scale)
    quats = torch.nn.functional.normalize(
        torch.randn(n_init_gaussians, 4, device=cfg.device), dim=-1
    )
    opacities = torch.logit(torch.ones(n_init_gaussians, 1, device=cfg.device) * cfg.init_opa)
    sh0 = torch.randn(n_init_gaussians, 3, device=cfg.device) * 0.1
    shN = torch.zeros(n_init_gaussians, 9, 3, device=cfg.device)  # Up to degree 3 SH

    params = {
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "sh0": torch.nn.Parameter(sh0),
        "shN": torch.nn.Parameter(shN),
    }

    # Setup optimizers with FastGS learning rates
    optimizers = {
        "means": optim.Adam([params["means"]], lr=cfg.lr_means),
        "scales": optim.Adam([params["scales"]], lr=cfg.lr_scales),
        "quats": optim.Adam([params["quats"]], lr=cfg.lr_quats),
        "opacities": optim.Adam([params["opacities"]], lr=cfg.lr_opacities),
        "sh0": optim.Adam([params["sh0"]], lr=cfg.lr_sh0),  # Lower LR for DC
        "shN": optim.Adam([params["shN"]], lr=cfg.lr_shN),  # Much lower for higher order
    }

    # Initialize FastGS strategy
    strategy = FastGSStrategy(
        loss_threshold=cfg.loss_threshold,
        densify_threshold=cfg.densify_threshold,
        prune_threshold=cfg.prune_threshold,
        refine_every=cfg.refine_every,
        verbose=True,
    )
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=1.0)

    # Training loop
    print("Starting training...")
    start_time = time.time()

    for step in range(cfg.max_steps):
        # Sample a random camera
        # camera = random.choice(train_cameras)

        # Render
        # renders, alpha, info = rasterization(...)

        # Compute multi-view scores every refine_every steps
        # if step % cfg.refine_every == 0 and step > 0:
        #     info["densify_scores"], info["prune_scores"] = compute_multiview_scores(
        #         train_cameras, params, cfg
        #     )

        # Strategy pre-backward
        # strategy.step_pre_backward(params, optimizers, strategy_state, step, info)

        # Compute loss
        # loss = ...

        # Backward
        # for optimizer in optimizers.values():
        #     optimizer.zero_grad()
        # loss.backward()

        # Strategy post-backward
        # strategy.step_post_backward(params, optimizers, strategy_state, step, info)

        # Optimizer step
        # for optimizer in optimizers.values():
        #     optimizer.step()

        # Logging
        if step % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step}/{cfg.max_steps} | Time: {elapsed:.2f}s | "
                  f"N_gaussians: {len(params['means'])}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"FastGS target: 100s. Your result: {total_time:.2f}s")

    # Save results
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in params.items()},
               cfg.result_dir / "final_gaussians.pt")
    print(f"Saved results to {cfg.result_dir}")


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    train(cfg)
