"""FastGS Strategy with Dual Gradient Accumulation.

This is an enhanced version of FastGSStrategy that uses true dual gradient
accumulation (xy vs depth gradients) as in the original FastGS implementation.

Unlike the standard FastGSStrategy which uses dual thresholds on the same
gradient, this version separately tracks:
- XY screen-space gradients → for cloning decisions
- Depth gradients → for splitting decisions

This more closely matches the original FastGS behavior.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch
from typing_extensions import Literal

from ..rendering_fastgs import (
    DualGradientCapture,
    create_fastgs_dual_state,
    get_fastgs_dual_thresholds,
    update_fastgs_dual_gradients,
)
from .base import Strategy
from .ops import duplicate, remove, reset_opa, split


@dataclass
class FastGSDualStrategy(Strategy):
    """FastGS strategy with true dual gradient accumulation (xy + depth).

    This enhanced version uses separate gradient tracking for xy (screen-space)
    and depth components, exactly as the original FastGS implementation does.

    **Differences from FastGSStrategy**:
    - Tracks xy gradients separately from depth gradients
    - Uses `rasterization_fastgs()` with `DualGradientCapture`
    - More accurate to original FastGS behavior
    - Slightly more memory (2 accumulators instead of 1)

    **Algorithm**:
    1. Track xy screen-space gradients → cloning threshold (0.0002)
    2. Track depth gradients → splitting threshold (0.0012)
    3. Apply multi-view filtering as usual
    4. Clone where: high_xy & small & high_importance
    5. Split where: high_depth & large & high_importance

    Args:
        prune_opa (float): Opacity threshold for pruning. Default: 0.005.
        grow_grad2d_xy (float): XY gradient threshold for cloning. Default: 0.0002.
        grow_grad2d_depth (float): Depth gradient threshold for splitting. Default: 0.0012.
        grow_scale3d (float): Scale threshold. Default: 0.001.
        prune_scale3d (float): Scale threshold for pruning. Default: 0.1.
        loss_threshold (float): Loss threshold for error masks. Default: 0.1.
        importance_thresh (float): Importance score threshold. Default: 5.0.
        prune_score_thresh (float): Pruning score threshold. Default: 0.9.
        refine_start_iter (int): Start densification iteration. Default: 500.
        refine_stop_iter (int): Stop densification iteration. Default: 15_000.
        refine_every (int): Densification interval. Default: 100.
        reset_every (int): Opacity reset interval. Default: 3000.
        aggressive_prune_start (int): Aggressive pruning start. Default: 15_000.
        aggressive_prune_interval (int): Aggressive pruning interval. Default: 3000.
        aggressive_prune_opa (float): Aggressive pruning opacity. Default: 0.1.
        n_sample_cameras (int): Cameras for multi-view scoring. Default: 10.
        verbose (bool): Verbose output. Default: False.

    Examples:
        >>> from gsplat import FastGSDualStrategy
        >>> from gsplat.rendering_fastgs import rasterization_fastgs, DualGradientCapture
        >>>
        >>> strategy = FastGSDualStrategy()
        >>> strategy_state = strategy.initialize_state()
        >>> capture = DualGradientCapture()
        >>>
        >>> for step in range(30000):
        ...     # Use FastGS-specific rendering
        ...     renders, alphas, info = rasterization_fastgs(
        ...         means, quats, scales, opacities, colors,
        ...         viewmats, Ks, width, height,
        ...         dual_gradient_capture=capture,
        ...         absgrad=True,
        ...     )
        ...
        ...     loss = compute_loss(renders, gt_images)
        ...     loss.backward()
        ...
        ...     # Get dual gradients
        ...     xy_grads, depth_grads, gs_ids = capture.get_gradients(info)
        ...     info["xy_grads"] = xy_grads
        ...     info["depth_grads"] = depth_grads
        ...     info["gradient_ids"] = gs_ids
        ...
        ...     # Compute multi-view scores when needed
        ...     if step % 100 == 0 and step > 500:
        ...         info["importance_score"], info["pruning_score"] = compute_scores(...)
        ...
        ...     strategy.step_post_backward(params, optimizers, strategy_state, step, info)
        ...
        ...     # Optional: FastGS optimizer scheduling
        ...     if strategy.should_update_optimizers(step):
        ...         for opt in optimizers.values():
        ...             opt.step()
        ...             opt.zero_grad()
        ...
        ...     capture.reset()

    Note:
        This requires using `rasterization_fastgs()` instead of the standard
        `rasterization()` function. The dual gradient capture hooks into the
        backward pass to separately track xy and depth gradients.
    """

    prune_opa: float = 0.005
    grow_grad2d_xy: float = 0.0002  # For cloning (original FastGS threshold)
    grow_grad2d_depth: float = 0.0012  # For splitting (original FastGS threshold)
    grow_scale3d: float = 0.001
    prune_scale3d: float = 0.1
    loss_threshold: float = 0.1
    importance_thresh: float = 5.0
    prune_score_thresh: float = 0.9
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    refine_every: int = 100
    reset_every: int = 3000
    aggressive_prune_start: int = 15_000
    aggressive_prune_interval: int = 3000
    aggressive_prune_opa: float = 0.1
    n_sample_cameras: int = 10
    verbose: bool = False

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """Initialize state with dual gradient accumulators."""
        state = {
            "grad2d_xy": None,  # XY gradients for cloning
            "grad2d_depth": None,  # Depth gradients for splitting
            "count": None,  # Visibility count
            "scene_scale": scene_scale,
        }
        return state

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Sanity check for parameters and optimizers."""
        super().check_sanity(params, optimizers)
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def should_update_optimizers(self, step: int) -> bool:
        """FastGS optimizer scheduling (same as FastGSStrategy)."""
        if step <= 15_000:
            return True
        elif step <= 20_000:
            return step % 32 == 0
        else:
            return step % 64 == 0

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        """Callback after backward pass with dual gradient tracking."""
        if step >= self.refine_stop_iter:
            return

        # Update gradient state with dual gradients
        self._update_state_dual(params, state, info)

        # Densification
        if step > self.refine_start_iter and step % self.refine_every == 0:
            if "importance_score" not in info or "pruning_score" not in info:
                if self.verbose:
                    print(
                        f"Step {step}: Multi-view scores not provided. "
                        f"Skipping FastGS densification."
                    )
                return

            # Grow with dual gradients
            n_dupli, n_split = self._grow_gs_dual(params, optimizers, state, step, info)
            if self.verbose:
                print(
                    f"Step {step}: {n_dupli} GSs cloned, {n_split} GSs split. "
                    f"Now having {len(params['means'])} GSs."
                )

            # Standard pruning
            n_prune = self._prune_gs(params, optimizers, state, step, info)
            if self.verbose:
                print(
                    f"Step {step}: {n_prune} GSs pruned. "
                    f"Now having {len(params['means'])} GSs."
                )

            # Reset running stats
            state["grad2d_xy"].zero_()
            state["grad2d_depth"].zero_()
            state["count"].zero_()
            torch.cuda.empty_cache()

        # Aggressive pruning
        if (
            step % self.aggressive_prune_interval == 0
            and step >= self.aggressive_prune_start
            and step < 30_000
        ):
            if "pruning_score" in info:
                n_prune_aggr = self._aggressive_prune(params, optimizers, state, info)
                if self.verbose:
                    print(
                        f"Step {step}: Aggressive pruning removed {n_prune_aggr} GSs. "
                        f"Now having {len(params['means'])} GSs."
                    )

        # Opacity reset
        if step % self.reset_every == 0 and step > 0:
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=self.prune_opa * 2.0,
            )

    def _update_state_dual(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
    ):
        """Update dual gradient accumulators."""
        # Check for dual gradients in info
        if "xy_grads" not in info or "depth_grads" not in info or "gradient_ids" not in info:
            raise ValueError(
                "Dual gradients not found in info. Make sure you're using "
                "rasterization_fastgs() with DualGradientCapture and passing "
                "the gradients via info dict."
            )

        xy_grads = info["xy_grads"]
        depth_grads = info["depth_grads"]
        gs_ids = info["gradient_ids"]

        # Initialize state on first run
        n_gaussians = len(params["means"])
        if state["grad2d_xy"] is None:
            state["grad2d_xy"] = torch.zeros(n_gaussians, device=xy_grads.device)
        if state["grad2d_depth"] is None:
            state["grad2d_depth"] = torch.zeros(n_gaussians, device=depth_grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussians, device=gs_ids.device)

        # Accumulate dual gradients
        update_fastgs_dual_gradients(state, xy_grads, depth_grads, gs_ids)

    @torch.no_grad()
    def _grow_gs_dual(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ) -> Tuple[int, int]:
        """Grow Gaussians using TRUE dual gradients (xy vs depth)."""
        importance_score = info["importance_score"]
        device = importance_score.device

        # Get dual gradient thresholds
        is_high_xy, is_high_depth = get_fastgs_dual_thresholds(
            state,
            grad_thresh_xy=self.grow_grad2d_xy,
            grad_thresh_depth=self.grow_grad2d_depth,
        )

        # Scale-based separation
        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        is_large = ~is_small

        # Multi-view consistency filter
        is_high_error = importance_score > self.importance_thresh

        # FastGS hybrid logic with TRUE dual gradients:
        # Clone: high XY gradients & small size & high error
        # Split: high DEPTH gradients & large size & high error
        is_dupli = is_high_xy & is_small & is_high_error
        is_split = is_high_depth & is_large & is_high_error

        n_dupli = is_dupli.sum().item()
        n_split = is_split.sum().item()

        # Duplicate first
        if n_dupli > 0:
            duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)

        # New GSs added by duplication will not be split
        is_split = torch.cat(
            [is_split, torch.zeros(n_dupli, dtype=torch.bool, device=device)]
        )

        # Then split
        if n_split > 0:
            split(params=params, optimizers=optimizers, state=state, mask=is_split)

        return n_dupli, n_split

    @torch.no_grad()
    def _prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ) -> int:
        """Standard pruning with budget-based sampling (same as FastGSStrategy)."""
        is_prune_opa = torch.sigmoid(params["opacities"]).squeeze(-1) < self.prune_opa
        is_prune_scale = (
            torch.exp(params["scales"]).max(dim=-1).values
            > self.prune_scale3d * state["scene_scale"]
        )

        is_prune = is_prune_opa | is_prune_scale
        n_candidates = is_prune.sum().item()

        if n_candidates == 0:
            return 0

        # Budget-based pruning if pruning_score available
        if "pruning_score" in info and n_candidates > 1:
            pruning_score = info["pruning_score"]
            n_gaussians = len(params["means"])

            scores = 1.0 - pruning_score
            remove_budget = max(1, int(0.5 * n_candidates))

            padded_importance = torch.zeros(
                n_gaussians, dtype=torch.float32, device=scores.device
            )
            padded_importance[:scores.shape[0]] = 1.0 / (1e-6 + scores)
            padded_importance = padded_importance * is_prune.float()

            if padded_importance.sum() > 0:
                padded_importance = padded_importance / padded_importance.sum()

                sampled_indices = torch.multinomial(
                    padded_importance, remove_budget, replacement=False
                )

                selected_mask = torch.zeros_like(is_prune, dtype=torch.bool)
                selected_mask[sampled_indices] = True
                final_prune = is_prune & selected_mask

                n_prune = final_prune.sum().item()
                if n_prune > 0:
                    remove(params=params, optimizers=optimizers, state=state, mask=final_prune)

                return n_prune

        # Fallback: remove all candidates
        if n_candidates > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_candidates

    @torch.no_grad()
    def _aggressive_prune(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        info: Dict[str, Any],
    ) -> int:
        """Aggressive pruning (same as FastGSStrategy)."""
        pruning_score = info["pruning_score"]

        is_prune_opa = (
            torch.sigmoid(params["opacities"]).squeeze(-1) < self.aggressive_prune_opa
        )
        is_prune_score = pruning_score > self.prune_score_thresh

        is_prune = is_prune_opa | is_prune_score
        n_prune = is_prune.sum().item()

        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune
