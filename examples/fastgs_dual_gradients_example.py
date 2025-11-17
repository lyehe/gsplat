"""Example: Using FastGS with True Dual Gradient Accumulation.

This example demonstrates how to use the FastGSDualStrategy which tracks
xy (screen-space) and depth gradients separately, exactly as the original
FastGS implementation does.

This provides more accurate FastGS behavior compared to the standard
FastGSStrategy which uses dual thresholds on the same gradient.
"""

import torch
from gsplat import FastGSDualStrategy, rasterization_fastgs
from gsplat.rendering_fastgs import DualGradientCapture
from gsplat.strategy.fastgs_utils import compute_gaussian_score_fastgs, sample_cameras


def train_with_dual_gradients():
    """Training loop example with dual gradient tracking."""

    # Initialize scene
    n_gaussians = 10000
    params = {
        "means": torch.nn.Parameter(torch.randn(n_gaussians, 3, device="cuda")),
        "quats": torch.nn.Parameter(torch.randn(n_gaussians, 4, device="cuda")),
        "scales": torch.nn.Parameter(torch.randn(n_gaussians, 3, device="cuda")),
        "opacities": torch.nn.Parameter(torch.randn(n_gaussians, 1, device="cuda")),
    }

    optimizers = {
        key: torch.optim.Adam([param], lr=1e-3) for key, param in params.items()
    }

    # Initialize FastGS strategy with dual gradients
    strategy = FastGSDualStrategy(verbose=True)
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=1.0)

    # Create dual gradient capture object
    capture = DualGradientCapture()

    # Mock camera data
    viewmats = torch.eye(4, device="cuda").unsqueeze(0)  # [1, 4, 4]
    Ks = torch.tensor([[[1000, 0, 400], [0, 1000, 300], [0, 0, 1]]], device="cuda")
    width, height = 800, 600

    # Training loop
    for step in range(30_000):
        # Use FastGS-specific rasterization with dual gradient capture
        renders, alphas, info = rasterization_fastgs(
            means=params["means"],
            quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=torch.rand(n_gaussians, 3, device="cuda"),  # Mock colors
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            dual_gradient_capture=capture,  # Enable dual gradient tracking
            absgrad=True,  # FastGS uses absolute gradients
        )

        # Compute loss (mock)
        gt_images = torch.rand_like(renders)
        loss = torch.nn.functional.l1_loss(renders, gt_images)

        # Backward pass
        loss.backward()

        # Extract dual gradients
        xy_grads, depth_grads, gs_ids = capture.get_gradients(info)

        # Pass gradients to strategy via info dict
        info["xy_grads"] = xy_grads
        info["depth_grads"] = depth_grads
        info["gradient_ids"] = gs_ids

        # Compute multi-view scores periodically
        if step % strategy.refine_every == 0 and step > strategy.refine_start_iter:
            # Mock multi-view scoring (in real code, use actual cameras)
            importance_score = torch.randint(0, 10, (len(params["means"]),), device="cuda").float()
            pruning_score = torch.rand(len(params["means"]), device="cuda")

            info["importance_score"] = importance_score
            info["pruning_score"] = pruning_score

        # Strategy callback
        strategy.step_post_backward(params, optimizers, strategy_state, step, info)

        # Optional: FastGS optimizer scheduling
        if strategy.should_update_optimizers(step):
            for opt in optimizers.values():
                opt.step()
                opt.zero_grad()

        # Reset capture for next iteration
        capture.reset()

        if step % 1000 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}, GSs = {len(params['means'])}")


def compare_dual_vs_single_threshold():
    """Compare dual gradient tracking vs single gradient with dual thresholds."""
    print("=" * 70)
    print("Dual Gradient Tracking Comparison")
    print("=" * 70)

    # Create mock gradients
    n_gaussians = 100
    xy_grads = torch.rand(n_gaussians) * 0.001  # Screen-space gradients
    depth_grads = torch.rand(n_gaussians) * 0.002  # Depth gradients

    # Dual gradient approach (FastGSDualStrategy)
    grad_thresh_xy = 0.0002
    grad_thresh_depth = 0.0012

    is_high_xy = xy_grads > grad_thresh_xy
    is_high_depth = depth_grads > grad_thresh_depth

    n_clone_candidates = is_high_xy.sum().item()
    n_split_candidates = is_high_depth.sum().item()

    print("\n✅ Dual Gradient Tracking (FastGSDualStrategy):")
    print(f"  - XY gradients > {grad_thresh_xy}: {n_clone_candidates} GSs → cloning")
    print(f"  - Depth gradients > {grad_thresh_depth}: {n_split_candidates} GSs → splitting")
    print("  - Separate decisions based on gradient TYPE")

    # Single gradient approach (FastGSStrategy - standard)
    combined_grads = (xy_grads + depth_grads) / 2  # Approximate

    is_high_combined_clone = combined_grads > grad_thresh_xy
    is_high_combined_split = combined_grads > grad_thresh_depth

    n_clone_candidates_single = is_high_combined_clone.sum().item()
    n_split_candidates_single = is_high_combined_split.sum().item()

    print("\n⚠️  Single Gradient with Dual Thresholds (FastGSStrategy):")
    print(f"  - Combined grads > {grad_thresh_xy}: {n_clone_candidates_single} GSs → cloning")
    print(f"  - Combined grads > {grad_thresh_depth}: {n_split_candidates_single} GSs → splitting")
    print("  - Same decisions based on gradient MAGNITUDE")

    print("\n📊 Difference:")
    print(f"  - Clone candidates differ by: {abs(n_clone_candidates - n_clone_candidates_single)}")
    print(f"  - Split candidates differ by: {abs(n_split_candidates - n_split_candidates_single)}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    print("FastGS Dual Gradient Tracking Example")
    print()

    # Show comparison
    compare_dual_vs_single_threshold()

    print("\nNote: The training loop is commented out to avoid requiring a full scene.")
    print("To run training, uncomment train_with_dual_gradients() below.")
    print()

    # Uncomment to run training (requires more setup)
    # train_with_dual_gradients()
