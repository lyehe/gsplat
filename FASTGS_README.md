# FastGS Implementation for gsplat

This directory contains a clean-room implementation of the FastGS algorithm for accelerated 3D Gaussian Splatting training, integrated into gsplat's Apache 2.0 licensed codebase.

## Overview

**FastGS: Training 3D Gaussian Splatting in 100 Seconds**
Paper: https://arxiv.org/abs/2511.04283

FastGS accelerates 3D Gaussian Splatting training through a multi-view consistency based densification and pruning strategy, achieving **3.32× speedup** compared to DashGaussian and **15.45× speedup** compared to vanilla 3DGS.

## Key Features

- **Multi-View Consistency**: Uses view-consistent densification (VCD) and pruning (VCP) instead of gradient-based methods
- **Fast Training**: Target training time of ~100 seconds per scene
- **Apache 2.0 Licensed**: Clean-room implementation avoiding Max Planck licensed code
- **gsplat Native**: Fully integrated with gsplat's infrastructure

## Implementation Details

### Core Components

1. **`gsplat/strategy/fastgs.py`**: `FastGSStrategy` class
   - Implements VCD (View-Consistent Densification)
   - Implements VCP (View-Consistent Pruning)
   - Compatible with gsplat's strategy interface

2. **`gsplat/strategy/fastgs_utils.py`**: Utility functions
   - `compute_loss_map()`: Per-pixel L1 loss computation
   - `create_error_mask()`: Binary error mask creation
   - `compute_gaussian_importance_scores()`: Multi-view consistency scoring

3. **`examples/simple_trainer_fastgs.py`**: Example training script
   - Demonstrates FastGS usage
   - Shows multi-view scoring integration

### Algorithm

FastGS replaces gradient-based densification with multi-view consistency:

**View-Consistent Densification (VCD):**
```
s_i^+ = 1/K ∑(j=1 to K) ∑(p∈Ω_i) 𝕀(M_mask^j(p)=1)
```
- Samples K training views (default: 10)
- Computes per-pixel L1 loss maps
- Creates binary masks for high-error pixels (threshold τ)
- Counts error pixels within each Gaussian's 2D footprint
- Densifies Gaussians with high error counts

**View-Consistent Pruning (VCP):**
```
s_i^- = N(∑(j=1 to K) (∑(p∈Ω_i) 𝕀(M_mask^j(p)=1)) · E_photo^j)
```
- Weights error counts by photometric loss per view
- Normalizes to [0, 1]
- Prunes Gaussians with high weighted scores (redundant)

## Usage

### Basic Usage

```python
from gsplat import FastGSStrategy, rasterization

# Initialize parameters
params = {
    "means": torch.nn.Parameter(...),
    "scales": torch.nn.Parameter(...),
    "quats": torch.nn.Parameter(...),
    "opacities": torch.nn.Parameter(...),
}

# Setup optimizers
optimizers = {
    "means": optim.Adam([params["means"]], lr=1.6e-4),
    "scales": optim.Adam([params["scales"]], lr=5e-3),
    # ... other optimizers
}

# Initialize FastGS strategy
strategy = FastGSStrategy(
    loss_threshold=0.5,      # Error mask threshold
    densify_threshold=0.1,   # VCD threshold
    prune_threshold=0.5,     # VCP threshold
    refine_every=500,        # FastGS uses 500 instead of 100
    verbose=True,
)
strategy.check_sanity(params, optimizers)
strategy_state = strategy.initialize_state(scene_scale=1.0)

# Training loop
for step in range(30_000):
    # Render
    renders, alpha, info = rasterization(...)

    # Compute multi-view scores every refine_every steps
    if step % strategy.refine_every == 0:
        info["densify_scores"], info["prune_scores"] = compute_multiview_scores(...)

    # Strategy callbacks
    strategy.step_pre_backward(params, optimizers, strategy_state, step, info)

    loss = ...
    loss.backward()

    strategy.step_post_backward(params, optimizers, strategy_state, step, info)

    # Optimizer step
    for optimizer in optimizers.values():
        optimizer.step()
```

### Multi-View Scoring

The key difference from vanilla 3DGS is computing multi-view consistency scores:

```python
def compute_multiview_scores(cameras, params, loss_threshold=0.5, n_samples=10):
    """Compute VCD and VCP scores."""
    sampled_cameras = random.sample(cameras, n_samples)

    error_counts = torch.zeros(len(params["means"]))
    weighted_scores = torch.zeros(len(params["means"]))

    for camera in sampled_cameras:
        # Render
        with torch.no_grad():
            renders, _, info = rasterization(...)

        # Compute loss map
        loss_map = compute_loss_map(renders, camera.image)

        # Create error mask
        error_mask = (loss_map > loss_threshold).float()

        # Accumulate per-Gaussian error counts
        # (simplified - see fastgs_utils.py for full implementation)
        visible_gs_ids = get_visible_gaussians(info)
        error_counts[visible_gs_ids] += error_mask.sum()

        # Weight by photometric loss
        photo_loss = compute_photo_loss(renders, camera.image)
        weighted_scores[visible_gs_ids] += error_mask.sum() * photo_loss

    # VCD scores
    densify_scores = error_counts / n_samples

    # VCP scores (normalized)
    prune_scores = normalize(weighted_scores)

    return densify_scores, prune_scores
```

## Hyperparameters

FastGS uses different hyperparameters compared to vanilla 3DGS:

| Parameter | Vanilla 3DGS | FastGS | Description |
|-----------|--------------|--------|-------------|
| `refine_every` | 100 | 500 | Densification interval |
| `lr_sh_dc` | 0.0025 | 0.0025 | DC component learning rate |
| `lr_sh_rest` | 0.0025 / 20 | 0.0025 / 20 | Higher-order SH LR |
| `loss_threshold` | N/A | 0.5 | Error mask threshold |
| `densify_threshold` | N/A | 0.1 | VCD threshold |
| `prune_threshold` | N/A | 0.5 | VCP threshold |
| `n_sample_cameras` | N/A | 10 | Multi-view samples |

## License and Attribution

This implementation is **Apache 2.0 licensed** and developed as a clean-room implementation based solely on the published FastGS paper. It does **not** contain any code from:
- The original Max Planck/INRIA licensed 3D Gaussian Splatting implementation
- The FastGS repository's Max Planck licensed CUDA extensions

### Paper Citation

If you use this implementation, please cite both the FastGS paper and gsplat:

```bibtex
@article{ren2025fastgs,
  title={FastGS: Training 3D Gaussian Splatting in 100 Seconds},
  author={Ren, Shiwei and Wen, Tianci and Fang, Yongchun and Lu, Biao},
  journal={arXiv preprint arXiv:2511.04283},
  year={2025}
}

@article{ye2024gsplat,
  title={gsplat: An Open-Source Library for Gaussian Splatting},
  author={Ye, Vickie and Ni, Ruilong and Kerbl, Bernhard and Chen, Anpei and Kopanas, Georgios and Drettakis, George and Liu, Lingjie and Kanazawa, Angjoo and Tancik, Matthew},
  journal={arXiv preprint arXiv:2409.06765},
  year={2024}
}
```

## Differences from Original FastGS

This implementation differs from the original FastGS repository in:

1. **License**: Apache 2.0 instead of Max Planck license
2. **CUDA Kernels**: Uses gsplat's existing CUDA infrastructure instead of custom kernels
3. **API**: Follows gsplat's strategy pattern for consistency
4. **Dependencies**: Pure gsplat dependencies, no Max Planck licensed code

## Performance Notes

- Training time depends on scene complexity and hardware
- Target: ~100 seconds on RTX 4090 for MipNeRF360 scenes
- Multi-view scoring adds overhead every `refine_every` steps
- Consider GPU memory when increasing `n_sample_cameras`

## Troubleshooting

**Q: Training is slower than expected**
A: Try reducing `n_sample_cameras` or increasing `refine_every`. FastGS performance depends on efficient multi-view rendering.

**Q: Not enough densification**
A: Lower `densify_threshold` or `loss_threshold` to be more aggressive.

**Q: Too many Gaussians**
A: Increase `prune_threshold` or decrease `densify_threshold`.

**Q: Missing multi-view scores error**
A: Ensure you compute and pass `densify_scores` and `prune_scores` in the `info` dict during refinement steps.

## Contributing

This is a research implementation. Contributions welcome for:
- Performance optimizations
- Better multi-view scoring methods
- Integration with different datasets
- Benchmark results

## Acknowledgments

- FastGS paper authors for the algorithm
- gsplat team for the excellent infrastructure
- Gaussian Splatting community for the foundational work
