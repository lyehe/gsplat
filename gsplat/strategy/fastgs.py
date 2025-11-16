"""FastGS Strategy for 3D Gaussian Splatting.

Based on the FastGS paper: "FastGS: Training 3D Gaussian Splatting in 100 Seconds"
https://arxiv.org/abs/2511.04283

This is a clean-room implementation following the algorithm described in the paper,
using gsplat's Apache 2.0 licensed infrastructure.

The FastGS strategy differs from the default 3DGS strategy by using multi-view
consistency based densification and pruning instead of gradient-based methods.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch
from typing_extensions import Literal

from .base import Strategy
from .ops import duplicate, remove, reset_opa, split


@dataclass
class FastGSStrategy(Strategy):
    """FastGS strategy for accelerated 3D Gaussian Splatting training.

    This strategy implements the FastGS paper's multi-view consistency based
    densification and pruning approach. Instead of using image plane gradients,
    it uses:

    - **View-Consistent Densification (VCD)**: Counts high-error pixels across
      multiple views to identify under-reconstructed regions.
    - **View-Consistent Pruning (VCP)**: Uses photometric loss weighted by error
      counts to identify redundant Gaussians.

    The strategy requires the user to periodically compute multi-view consistency
    scores and pass them via the `info` dict in `step_post_backward()`.

    Args:
        loss_threshold (float): Threshold for identifying high-error pixels in
            loss maps. Default is 0.5 (after normalization to [0, 1]).
        densify_threshold (float): Gaussians with VCD score above this threshold
            will be densified. Default is 0.1.
        prune_threshold (float): Gaussians with VCP score above this threshold
            will be pruned. Default is 0.5.
        prune_opa (float): GSs with opacity below this value will be pruned. Default is 0.005.
        grow_scale3d (float): GSs with 3d scale (normalized by scene_scale) below this
            value will be duplicated. Above will be split. Default is 0.01.
        prune_scale3d (float): GSs with 3d scale (normalized by scene_scale) above this
            value will be pruned. Default is 0.1.
        refine_start_iter (int): Start refining GSs after this iteration. Default is 500.
        refine_stop_iter (int): Stop refining GSs after this iteration. Default is 15_000.
        refine_every (int): Refine GSs every this steps. Default is 500 (FastGS uses
            500 instead of 100 from vanilla 3DGS).
        reset_every (int): Reset opacities every this steps. Default is 3000.
        n_sample_cameras (int): Number of cameras to sample for multi-view consistency
            scoring. Default is 10.
        verbose (bool): Whether to print verbose information. Default is False.

    Examples:

        >>> from gsplat import FastGSStrategy, rasterization
        >>> params: Dict[str, torch.nn.Parameter] | torch.nn.ParameterDict = ...
        >>> optimizers: Dict[str, torch.optim.Optimizer] = ...
        >>> strategy = FastGSStrategy()
        >>> strategy.check_sanity(params, optimizers)
        >>> strategy_state = strategy.initialize_state()
        >>> for step in range(30000):
        ...     render_image, render_alpha, info = rasterization(...)
        ...     # Compute multi-view scores every refine_every steps
        ...     if step % strategy.refine_every == 0:
        ...         info["densify_scores"], info["prune_scores"] = compute_mv_scores(...)
        ...     strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        ...     loss = ...
        ...     loss.backward()
        ...     strategy.step_post_backward(params, optimizers, strategy_state, step, info)

    Note:
        Unlike DefaultStrategy which accumulates gradients automatically, FastGSStrategy
        requires the training loop to explicitly compute multi-view consistency scores
        and pass them via `info["densify_scores"]` and `info["prune_scores"]` during
        refinement steps. See `fastgs_utils.py` for helper functions.
    """

    loss_threshold: float = 0.5
    densify_threshold: float = 0.1
    prune_threshold: float = 0.5
    prune_opa: float = 0.005
    grow_scale3d: float = 0.01
    prune_scale3d: float = 0.1
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    refine_every: int = 500
    reset_every: int = 3000
    n_sample_cameras: int = 10
    verbose: bool = False

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy.

        Args:
            scene_scale: Scale of the scene for normalizing 3D scales. Default is 1.0.

        Returns:
            Dictionary containing strategy state.
        """
        state = {"scene_scale": scene_scale}
        return state

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Sanity check for the parameters and optimizers.

        Check if:
            * `params` and `optimizers` have the same keys.
            * Each optimizer has exactly one param_group, corresponding to each parameter.
            * The following keys are present: {"means", "scales", "quats", "opacities"}.

        Raises:
            AssertionError: If any of the above conditions is not met.
        """
        super().check_sanity(params, optimizers)
        # The following keys are required for this strategy.
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
        """Callback function to be executed before the `loss.backward()` call.

        For FastGS, this is a no-op since we don't need to retain gradients
        for multi-view consistency scoring.
        """
        pass

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """Callback function to be executed after the `loss.backward()` call.

        This function performs FastGS-based densification and pruning using
        multi-view consistency scores that should be provided in `info`.

        Args:
            params: Dictionary of parameters.
            optimizers: Dictionary of optimizers.
            state: Strategy state dictionary.
            step: Current training step.
            info: Dictionary containing rendering info. Should include:
                - "densify_scores": Tensor of shape [n_gaussians] with VCD scores (required
                  during refinement steps).
                - "prune_scores": Tensor of shape [n_gaussians] with VCP scores (required
                  during refinement steps).
            packed: Whether using packed mode. Default is False.
        """
        if step >= self.refine_stop_iter:
            return

        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
        ):
            # Check if multi-view scores are provided
            if "densify_scores" not in info or "prune_scores" not in info:
                if self.verbose:
                    print(
                        f"Step {step}: Multi-view consistency scores not provided in info. "
                        f"Skipping densification and pruning. Please compute scores using "
                        f"fastgs_utils and pass via info['densify_scores'] and info['prune_scores']."
                    )
                return

            # Grow GSs based on multi-view consistency
            n_dupli, n_split = self._grow_gs(params, optimizers, state, step, info)
            if self.verbose:
                print(
                    f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                    f"Now having {len(params['means'])} GSs."
                )

            # Prune GSs based on multi-view consistency and opacity
            n_prune = self._prune_gs(params, optimizers, state, step, info)
            if self.verbose:
                print(
                    f"Step {step}: {n_prune} GSs pruned. "
                    f"Now having {len(params['means'])} GSs."
                )

            torch.cuda.empty_cache()

        if step % self.reset_every == 0 and step > 0:
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=self.prune_opa * 2.0,
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
        """Grow Gaussians based on multi-view consistency scores (VCD).

        Args:
            params: Dictionary of parameters.
            optimizers: Dictionary of optimizers.
            state: Strategy state dictionary.
            step: Current training step.
            info: Dictionary containing "densify_scores".

        Returns:
            Tuple of (n_duplicated, n_split).
        """
        densify_scores = info["densify_scores"]
        device = densify_scores.device

        # Identify Gaussians with high error scores
        is_high_error = densify_scores > self.densify_threshold

        # Separate into small (duplicate) and large (split) based on 3D scale
        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )

        is_dupli = is_high_error & is_small
        n_dupli = is_dupli.sum().item()

        is_large = ~is_small
        is_split = is_high_error & is_large
        n_split = is_split.sum().item()

        # First duplicate
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
        """Prune Gaussians based on multi-view consistency scores (VCP) and other criteria.

        Args:
            params: Dictionary of parameters.
            optimizers: Dictionary of optimizers.
            state: Strategy state dictionary.
            step: Current training step.
            info: Dictionary containing "prune_scores".

        Returns:
            Number of Gaussians pruned.
        """
        prune_scores = info["prune_scores"]

        # Prune based on multi-view consistency scores
        is_prune_mv = prune_scores > self.prune_threshold

        # Prune low opacity Gaussians
        is_prune_opa = torch.sigmoid(params["opacities"]).squeeze(-1) < self.prune_opa

        # Prune large 3D scale Gaussians
        is_prune_scale = (
            torch.exp(params["scales"]).max(dim=-1).values
            > self.prune_scale3d * state["scene_scale"]
        )

        # Combine all pruning criteria
        is_prune = is_prune_mv | is_prune_opa | is_prune_scale
        n_prune = is_prune.sum().item()

        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune
