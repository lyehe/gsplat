"""FastMCMC Strategy: MCMC with Multi-View Consistency Filtering.

This is a novel hybrid strategy that combines:
- MCMC's probabilistic densification (no gradients needed)
- FastGS's multi-view consistency filtering

The result is faster, more efficient MCMC training with better Gaussian budgeting.

Based on:
- MCMC: "3D Gaussian Splatting as Markov Chain Monte Carlo" (https://arxiv.org/abs/2404.09591)
- FastGS: "FastGS: Training 3D Gaussian Splatting in 100 Seconds" (https://arxiv.org/abs/2511.04283)

This is a clean-room implementation using gsplat's Apache 2.0 licensed infrastructure.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Union

import torch
from torch import Tensor

from .base import Strategy
from .ops import inject_noise_to_position, remove, sample_add


@dataclass
class FastMCMCStrategy(Strategy):
    """FastMCMC: MCMC with multi-view consistency filtering for faster training.

    This strategy combines the best of MCMC and FastGS:

    **From MCMC:**
    - Probabilistic Gaussian management (no gradients needed)
    - Sampling new Gaussians based on opacity
    - Noise injection for MCMC exploration
    - Compatible with 3DGut and distorted cameras

    **From FastGS:**
    - Multi-view consistency scoring to filter operations
    - Prune redundant Gaussians (high pruning_score)
    - Relocate under-reconstructed Gaussians (low importance_score)
    - Sample in high-error areas (high importance_score)

    **Algorithm:**

    1. Every `refine_every` iterations (default 100):
       - Compute multi-view scores (like FastGS)
       - **Relocate** Gaussians with low opacity OR low importance_score
       - **Add new** Gaussians sampled by opacity, weighted by importance_score
       - **Prune** Gaussians with low opacity OR high pruning_score

    2. Every iteration:
       - Inject noise to positions (MCMC sampling)

    **Key Differences from MCMC:**
    - Uses multi-view consistency to guide probabilistic operations
    - Prunes redundant Gaussians (MCMC doesn't prune much)
    - More efficient Gaussian budgeting

    **Key Differences from FastGS:**
    - No gradient accumulation (compatible with 3DGut!)
    - Probabilistic instead of deterministic
    - Sampling instead of split/clone

    Args:
        cap_max (int): Maximum number of GSs. Default: 1_000_000.
        noise_lr (float): MCMC sampling noise learning rate. Default: 5e5.
        refine_start_iter (int): Start refining after this iteration. Default: 500.
        refine_stop_iter (int): Stop refining after this iteration. Default: 25_000.
        refine_every (int): Refine every this many steps. Default: 100.
        min_opacity (float): GSs with opacity below this will be relocated/pruned. Default: 0.005.
        loss_threshold (float): Threshold for error masks. Default: 0.1 (FastGS).
        importance_thresh (float): Gaussians with importance_score < this may be relocated.
            Lower = more aggressive relocation. Default: 2.0.
        prune_score_thresh (float): Gaussians with pruning_score > this will be pruned.
            Normalized to [0, 1]. Default: 0.8.
        n_sample_cameras (int): Number of cameras for multi-view scoring. Default: 10.
        verbose (bool): Whether to print verbose information. Default: False.

    Examples:

        >>> from gsplat import FastMCMCStrategy, rasterization
        >>> from gsplat.strategy.fastgs_utils import sample_cameras, compute_gaussian_score_fastgs
        >>> params: Dict[str, torch.nn.Parameter] = ...
        >>> optimizers: Dict[str, torch.optim.Optimizer] = ...
        >>> strategy = FastMCMCStrategy()
        >>> strategy.check_sanity(params, optimizers)
        >>> strategy_state = strategy.initialize_state()
        >>>
        >>> for step in range(30000):
        ...     # Regular rendering (no absgrad needed!)
        ...     renders, alphas, info = rasterization(
        ...         ...,
        ...         with_ut=True,      # Works with 3DGut!
        ...         with_eval3d=True,  # Works with 3DGut!
        ...     )
        ...     loss = ...
        ...     loss.backward()
        ...
        ...     # Compute multi-view scores when needed
        ...     if step % strategy.refine_every == 0 and step > strategy.refine_start_iter:
        ...         camlist = sample_cameras(train_cameras, n_samples=10)
        ...         importance_score, pruning_score = compute_gaussian_score_fastgs(
        ...             camlist, render_fn, n_gaussians, loss_threshold=0.1, densify_mode=True
        ...         )
        ...         info["importance_score"] = importance_score
        ...         info["pruning_score"] = pruning_score
        ...
        ...     # Strategy callback (needs lr for noise injection)
        ...     strategy.step_post_backward(params, optimizers, strategy_state, step, info, lr=1e-3)
        ...
        ...     for opt in optimizers.values():
        ...         opt.step()
        ...         opt.zero_grad()

    Note:
        Unlike vanilla MCMC which only uses opacity, FastMCMC uses multi-view consistency
        scores to make smarter decisions about which Gaussians to relocate, sample, and prune.
        This results in faster convergence and better quality.
    """

    cap_max: int = 1_000_000
    noise_lr: float = 5e5
    refine_start_iter: int = 500
    refine_stop_iter: int = 25_000
    refine_every: int = 100
    min_opacity: float = 0.005
    loss_threshold: float = 0.1
    importance_thresh: float = 2.0  # Count threshold
    prune_score_thresh: float = 0.8  # Normalized threshold
    n_sample_cameras: int = 10
    verbose: bool = False

    def initialize_state(self) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy."""
        # Binomial coefficients for relocation (from MCMC)
        n_max = 51
        binoms = torch.zeros((n_max, n_max))
        for n in range(n_max):
            for k in range(n + 1):
                binoms[n, k] = math.comb(n, k)
        return {"binoms": binoms}

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Sanity check for the parameters and optimizers."""
        super().check_sanity(params, optimizers)
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        lr: float,
    ):
        """Callback function after `loss.backward()`.

        Args:
            lr (float): Learning rate for "means" attribute (used for noise scaling).
        """
        # Move binoms to correct device
        state["binoms"] = state["binoms"].to(params["means"].device)
        binoms = state["binoms"]

        # FastMCMC refinement
        if (
            step < self.refine_stop_iter
            and step > self.refine_start_iter
            and step % self.refine_every == 0
        ):
            # Check if multi-view scores are provided
            if "importance_score" not in info or "pruning_score" not in info:
                if self.verbose:
                    print(
                        f"Step {step}: Multi-view scores not provided. "
                        f"Falling back to vanilla MCMC behavior."
                    )
                # Fallback to vanilla MCMC
                n_relocated = self._relocate_gs_vanilla(params, optimizers, binoms)
                n_new = self._add_new_gs_vanilla(params, optimizers, binoms)
            else:
                # Use FastMCMC with multi-view filtering
                n_relocated = self._relocate_gs_fastmcmc(
                    params, optimizers, binoms, info
                )
                n_new = self._add_new_gs_fastmcmc(
                    params, optimizers, binoms, info
                )
                n_pruned = self._prune_gs_fastmcmc(params, optimizers, state, info)

                if self.verbose:
                    print(
                        f"Step {step}: Relocated {n_relocated} GSs, "
                        f"Added {n_new} GSs, Pruned {n_pruned} GSs. "
                        f"Now having {len(params['means'])} GSs."
                    )

            torch.cuda.empty_cache()

        # MCMC noise injection (every iteration)
        inject_noise_to_position(
            params=params,
            optimizers=optimizers,
            state={},
            scaler=lr * self.noise_lr,
        )

    @torch.no_grad()
    def _relocate_gs_vanilla(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        binoms: Tensor,
    ) -> int:
        """Vanilla MCMC relocation: move low-opacity Gaussians."""
        from .ops import relocate

        opacities = torch.sigmoid(params["opacities"].flatten())
        dead_mask = opacities <= self.min_opacity
        n_gs = dead_mask.sum().item()
        if n_gs > 0:
            relocate(
                params=params,
                optimizers=optimizers,
                state={},
                mask=dead_mask,
                binoms=binoms,
                min_opacity=self.min_opacity,
            )
        return n_gs

    @torch.no_grad()
    def _relocate_gs_fastmcmc(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        binoms: Tensor,
        info: Dict[str, Any],
    ) -> int:
        """FastMCMC relocation: move Gaussians with low opacity OR low importance."""
        from .ops import relocate

        opacities = torch.sigmoid(params["opacities"].flatten())
        importance_score = info["importance_score"]

        # Relocate if low opacity OR low importance (under-reconstructed)
        low_opacity_mask = opacities <= self.min_opacity
        low_importance_mask = importance_score < self.importance_thresh

        relocate_mask = low_opacity_mask | low_importance_mask
        n_gs = relocate_mask.sum().item()

        if n_gs > 0:
            relocate(
                params=params,
                optimizers=optimizers,
                state={},
                mask=relocate_mask,
                binoms=binoms,
                min_opacity=self.min_opacity,
            )
        return n_gs

    @torch.no_grad()
    def _add_new_gs_vanilla(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        binoms: Tensor,
    ) -> int:
        """Vanilla MCMC sampling: add Gaussians sampled by opacity."""
        current_n_points = len(params["means"])
        n_target = min(self.cap_max, int(1.05 * current_n_points))
        n_gs = max(0, n_target - current_n_points)
        if n_gs > 0:
            sample_add(
                params=params,
                optimizers=optimizers,
                state={},
                n=n_gs,
                binoms=binoms,
                min_opacity=self.min_opacity,
            )
        return n_gs

    @torch.no_grad()
    def _add_new_gs_fastmcmc(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        binoms: Tensor,
        info: Dict[str, Any],
    ) -> int:
        """FastMCMC sampling: weight by opacity AND importance_score."""
        current_n_points = len(params["means"])
        n_target = min(self.cap_max, int(1.05 * current_n_points))
        n_gs = max(0, n_target - current_n_points)

        if n_gs > 0:
            # Weight sampling by both opacity and importance
            opacities = torch.sigmoid(params["opacities"].flatten())
            importance_score = info["importance_score"]

            # Normalize importance_score to [0, 1] for weighting
            imp_norm = importance_score / (importance_score.max() + 1e-8)

            # Combined probability: opacity * importance
            # This favors high-opacity Gaussians in high-error areas
            probs = opacities * (1.0 + imp_norm)  # Boost by importance
            probs = probs / (probs.sum() + 1e-8)  # Normalize

            # Sample using weighted probabilities
            from .ops import _multinomial_sample
            from gsplat.relocation import compute_relocation

            sampled_idxs = _multinomial_sample(probs, n_gs, replacement=True)

            # Use MCMC relocation logic
            new_opacities, new_scales = compute_relocation(
                opacities=opacities[sampled_idxs].unsqueeze(-1),
                scales=torch.exp(params["scales"])[sampled_idxs],
                ratios=torch.bincount(sampled_idxs, minlength=len(opacities))[
                    sampled_idxs
                ]
                + 1,
                binoms=binoms,
            )

            eps = torch.finfo(torch.float32).eps
            new_opacities = torch.clamp(
                new_opacities, max=1.0 - eps, min=self.min_opacity
            )

            # Add sampled Gaussians
            from .ops import _update_param_with_optimizer

            def param_fn(name: str, p: Tensor) -> Tensor:
                if name == "opacities":
                    p[sampled_idxs] = torch.logit(new_opacities.squeeze(-1))
                elif name == "scales":
                    p[sampled_idxs] = torch.log(new_scales)
                p_new = torch.cat([p, p[sampled_idxs]])
                return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)

            def optimizer_fn(key: str, v: Tensor) -> Tensor:
                v_new = torch.zeros(
                    (len(sampled_idxs), *v.shape[1:]), device=v.device
                )
                return torch.cat([v, v_new])

            _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)

        return n_gs

    @torch.no_grad()
    def _prune_gs_fastmcmc(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        info: Dict[str, Any],
    ) -> int:
        """FastMCMC pruning: remove Gaussians with low opacity OR high redundancy."""
        opacities = torch.sigmoid(params["opacities"].flatten())
        pruning_score = info["pruning_score"]

        # Prune if low opacity OR high redundancy
        low_opacity_mask = opacities <= self.min_opacity
        high_redundancy_mask = pruning_score > self.prune_score_thresh

        prune_mask = low_opacity_mask | high_redundancy_mask
        n_prune = prune_mask.sum().item()

        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=prune_mask)

        return n_prune
