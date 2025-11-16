"""FastGS Strategy for 3D Gaussian Splatting.

Based on the FastGS paper: "FastGS: Training 3D Gaussian Splatting in 100 Seconds"
https://arxiv.org/abs/2511.04283

This is a clean-room implementation following the algorithm described in the paper,
using gsplat's Apache 2.0 licensed infrastructure.

**IMPORTANT**: FastGS is a HYBRID strategy that combines gradient-based densification
(like vanilla 3DGS) with multi-view consistency filtering. It does NOT replace gradients!

The original FastGS uses custom CUDA modifications to accumulate per-Gaussian error counts,
which we approximate here using gsplat's existing capabilities.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch
from typing_extensions import Literal

from .base import Strategy
from .ops import duplicate, remove, reset_opa, split


@dataclass
class FastGSStrategy(Strategy):
    """FastGS strategy: gradient-based densification + multi-view consistency filtering.

    FastGS accelerates 3D Gaussian Splatting by combining the proven gradient-based
    densification from vanilla 3DGS with an additional multi-view consistency filter.

    **Algorithm**:

    1. **Gradient Accumulation** (every iteration, like vanilla 3DGS):
       - Accumulate image plane gradients for each Gaussian
       - Track visibility counts

    2. **Densification** (every `refine_every` iterations, default 100):
       - Identify candidates using gradients: `grad >= grad_thresh` OR `absgrad >= grad_abs_thresh`
       - Identify candidates using scale: small (clone) vs large (split)
       - **FastGS contribution**: Filter by multi-view metric `importance_score > importance_thresh`
       - Densify only where BOTH gradient AND metric conditions are met

    3. **Pruning** (two stages):
       - Stage 1 (iterations < 15000): Standard pruning (low opacity, large scale)
       - Stage 2 (iterations >= 15000, every 3000): Aggressive pruning using `pruning_score > prune_score_thresh`

    **Multi-View Consistency Scoring**:

    The user must compute multi-view scores periodically (every `refine_every` iterations):

    - Sample K cameras (default 10)
    - For each camera:
      * Render and compute per-pixel L1 loss
      * Create binary error mask (loss > `loss_thresh`)
      * Count error pixels each Gaussian contributes to
    - `importance_score`: Average error pixel count per Gaussian (VCD)
    - `pruning_score`: Normalized photometric loss weighted by counts (VCP)

    Args:
        prune_opa (float): Gaussians with opacity below this will be pruned. Default: 0.005.
        grow_grad2d (float): 2D gradient threshold for densification. Default: 0.0002.
        grow_grad2d_abs (float): Absolute 2D gradient threshold for split. Default: 0.0012.
        grow_scale3d (float): 3D scale threshold (normalized). Below=clone, above=split. Default: 0.001.
        prune_scale3d (float): 3D scale threshold for pruning. Default: 0.1.
        loss_threshold (float): Threshold for creating binary error masks from loss maps. Default: 0.1.
        importance_thresh (float): Gaussians with importance_score > this will be densified.
            This is a COUNT (not normalized). Default: 5.0 (means >5 error pixels across views).
        prune_score_thresh (float): Gaussians with pruning_score > this will be pruned in stage 2.
            This is NORMALIZED to [0, 1]. Default: 0.9.
        refine_start_iter (int): Start densification after this iteration. Default: 500.
        refine_stop_iter (int): Stop densification after this iteration. Default: 15_000.
        refine_every (int): Densify every this many iterations. Default: 100.
        reset_every (int): Reset opacities every this many iterations. Default: 3000.
        aggressive_prune_start (int): Start aggressive pruning (stage 2) after this. Default: 15_000.
        aggressive_prune_interval (int): Interval for aggressive pruning. Default: 3000.
        aggressive_prune_opa (float): Opacity threshold for aggressive pruning. Default: 0.1.
        n_sample_cameras (int): Number of cameras to sample for multi-view scoring. Default: 10.
        absgrad (bool): Whether to use absolute gradients. Default: True (FastGS uses abs gradients).
        verbose (bool): Whether to print verbose information. Default: False.
        key_for_gradient (str): Which gradient to use for densification. Default: "means2d".

    Examples:

        >>> from gsplat import FastGSStrategy, rasterization
        >>> params: Dict[str, torch.nn.Parameter] = ...
        >>> optimizers: Dict[str, torch.optim.Optimizer] = ...
        >>> strategy = FastGSStrategy()
        >>> strategy.check_sanity(params, optimizers)
        >>> strategy_state = strategy.initialize_state()
        >>>
        >>> for step in range(30000):
        ...     # Regular rendering
        ...     renders, alphas, info = rasterization(..., absgrad=True)
        ...     strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        ...     loss = ...
        ...     loss.backward()
        ...
        ...     # Compute multi-view scores when needed (every refine_every iterations)
        ...     if step % strategy.refine_every == 0 and step > strategy.refine_start_iter:
        ...         info["importance_score"], info["pruning_score"] = compute_multiview_scores(...)
        ...
        ...     strategy.step_post_backward(params, optimizers, strategy_state, step, info)
        ...
        ...     for opt in optimizers.values():
        ...         opt.step()
        ...         opt.zero_grad()

    Note:
        Unlike vanilla 3DGS which only uses gradients, FastGS requires the training loop
        to compute multi-view consistency scores periodically. These scores should be passed
        via `info["importance_score"]` and `info["pruning_score"]` during refinement steps.

        The original FastGS uses custom CUDA modifications to efficiently compute per-Gaussian
        error counts. This implementation approximates those counts using gsplat's existing
        capabilities. See `fastgs_utils.py` for helper functions.
    """

    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    grow_grad2d_abs: float = 0.0012
    grow_scale3d: float = 0.001  # FastGS uses 0.001, not 0.01
    prune_scale3d: float = 0.1
    loss_threshold: float = 0.1  # FastGS uses 0.1, not 0.5!
    importance_thresh: float = 5.0  # Count threshold, not normalized!
    prune_score_thresh: float = 0.9  # Normalized threshold
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    refine_every: int = 100  # FastGS uses 100, same as vanilla!
    reset_every: int = 3000
    aggressive_prune_start: int = 15_000
    aggressive_prune_interval: int = 3000
    aggressive_prune_opa: float = 0.1
    n_sample_cameras: int = 10
    absgrad: bool = True  # FastGS uses absolute gradients
    verbose: bool = False
    key_for_gradient: Literal["means2d", "gradient_2dgs"] = "means2d"

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy."""
        # Same as DefaultStrategy - we still accumulate gradients
        state = {"grad2d": None, "count": None, "scene_scale": scene_scale}
        return state

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Sanity check for the parameters and optimizers."""
        super().check_sanity(params, optimizers)
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        """Callback function before `loss.backward()`.

        Retain gradients for means2d to accumulate later.
        """
        assert (
            self.key_for_gradient in info
        ), "The 2D means of the Gaussians is required but missing."
        info[self.key_for_gradient].retain_grad()

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """Callback function after `loss.backward()`.

        Implements the FastGS hybrid densification and pruning strategy.
        """
        if step >= self.refine_stop_iter:
            return

        # Always update gradient state (like vanilla 3DGS)
        self._update_state(params, state, info, packed=packed)

        # Densification (iterations 500-15000, every 100)
        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
        ):
            # Check if multi-view scores are provided
            if "importance_score" not in info or "pruning_score" not in info:
                if self.verbose:
                    print(
                        f"Step {step}: Multi-view scores not provided. "
                        f"Skipping FastGS densification. Please compute and pass "
                        f"info['importance_score'] and info['pruning_score']."
                    )
                return

            # Grow GSs using hybrid gradient + multi-view filtering
            n_dupli, n_split = self._grow_gs(params, optimizers, state, step, info)
            if self.verbose:
                print(
                    f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
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
            state["grad2d"].zero_()
            state["count"].zero_()
            torch.cuda.empty_cache()

        # Aggressive pruning (iterations 15000-30000, every 3000)
        if (
            step % self.aggressive_prune_interval == 0
            and step >= self.aggressive_prune_start
            and step < 30_000  # FastGS stops at 30k
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

    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """Update gradient accumulation state (same as DefaultStrategy)."""
        for key in [
            "width",
            "height",
            "n_cameras",
            "radii",
            "gaussian_ids",
            self.key_for_gradient,
        ]:
            assert key in info, f"{key} is required but missing."

        # Normalize grads to [-1, 1] screen space
        if self.absgrad:
            grads = info[self.key_for_gradient].absgrad.clone()
        else:
            grads = info[self.key_for_gradient].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        # Initialize state on first run
        n_gaussian = len(list(params.values())[0])

        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian, device=grads.device)

        # Update running state
        if packed:
            gs_ids = info["gaussian_ids"]
            radii = info["radii"].max(dim=-1).values
        else:
            sel = (info["radii"] > 0.0).all(dim=-1)
            gs_ids = torch.where(sel)[1]
            grads = grads[sel]
            radii = info["radii"][sel].max(dim=-1).values

        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        state["count"].index_add_(
            0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
        )

    @torch.no_grad()
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ) -> Tuple[int, int]:
        """Grow Gaussians using FastGS hybrid approach: gradients + multi-view metrics."""
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device

        importance_score = info["importance_score"]

        # FastGS Step 1: Gradient-based candidates (like vanilla 3DGS, but with abs grad for split)
        is_grad_high = grads > self.grow_grad2d  # For cloning
        is_grad_high_abs = grads > self.grow_grad2d_abs  # For splitting (FastGS uses higher thresh)

        # FastGS Step 2: Scale-based separation
        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        is_large = ~is_small

        # FastGS Step 3: Multi-view consistency filter
        is_high_error = importance_score > self.importance_thresh  # >5 error pixels

        # FastGS Step 4: Combine conditions (KEY: AND operation, not OR!)
        is_dupli = is_grad_high & is_small & is_high_error  # Gradient AND small AND high error
        is_split = is_grad_high_abs & is_large & is_high_error  # AbsGrad AND large AND high error

        n_dupli = is_dupli.sum().item()
        n_split = is_split.sum().item()

        # Duplicate first
        if n_dupli > 0:
            duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)

        # New GSs added by duplication will not be split
        is_split = torch.cat(
            [
                is_split,
                torch.zeros(n_dupli, dtype=torch.bool, device=device),
            ]
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
        """Standard pruning with budget-based sampling (FastGS optimization).

        Identifies candidates by low opacity OR large scale, then uses pruning_score
        to intelligently select which 50% to actually remove, preserving Gaussians
        that contribute well to reconstruction.
        """
        is_prune_opa = torch.sigmoid(params["opacities"]).squeeze(-1) < self.prune_opa
        is_prune_scale = (
            torch.exp(params["scales"]).max(dim=-1).values
            > self.prune_scale3d * state["scene_scale"]
        )

        is_prune = is_prune_opa | is_prune_scale
        n_candidates = is_prune.sum().item()

        if n_candidates == 0:
            return 0

        # Budget-based pruning if pruning_score available (FastGS optimization)
        if "pruning_score" in info and n_candidates > 1:
            pruning_score = info["pruning_score"]
            n_gaussians = len(params["means"])

            # Invert score: low score = good Gaussian, high score = bad Gaussian
            # We want to preserve good Gaussians even if they meet prune criteria
            scores = 1.0 - pruning_score

            # Only remove 50% of candidates, weighted by quality
            remove_budget = max(1, int(0.5 * n_candidates))

            # Weight by inverse score: bad Gaussians more likely to be removed
            padded_importance = torch.zeros(n_gaussians, dtype=torch.float32, device=scores.device)
            padded_importance[:scores.shape[0]] = 1.0 / (1e-6 + scores)

            # Only sample from prune candidates
            padded_importance = padded_importance * is_prune.float()

            # Normalize to valid probability distribution
            if padded_importance.sum() > 0:
                padded_importance = padded_importance / padded_importance.sum()

                # Sample Gaussians to remove
                sampled_indices = torch.multinomial(
                    padded_importance,
                    remove_budget,
                    replacement=False
                )

                # Create final prune mask
                selected_mask = torch.zeros_like(is_prune, dtype=torch.bool)
                selected_mask[sampled_indices] = True
                final_prune = is_prune & selected_mask

                n_prune = final_prune.sum().item()
                if n_prune > 0:
                    remove(params=params, optimizers=optimizers, state=state, mask=final_prune)

                return n_prune

        # Fallback: remove all candidates if no pruning_score
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
        """Aggressive pruning using multi-view consistency scores (FastGS stage 2)."""
        pruning_score = info["pruning_score"]

        # Prune by opacity OR multi-view consistency
        is_prune_opa = (
            torch.sigmoid(params["opacities"]).squeeze(-1) < self.aggressive_prune_opa
        )
        is_prune_score = pruning_score > self.prune_score_thresh

        is_prune = is_prune_opa | is_prune_score
        n_prune = is_prune.sum().item()

        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune
