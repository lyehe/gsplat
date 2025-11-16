#!/usr/bin/env python3
"""Quick test for FastMCMC strategy implementation."""

import torch
from gsplat import FastMCMCStrategy

def test_fastmcmc_basic():
    """Test basic FastMCMC instantiation and initialization."""
    print("Testing FastMCMC basic functionality...")

    # Test instantiation
    strategy = FastMCMCStrategy(
        cap_max=1_000_000,
        noise_lr=5e5,
        refine_start_iter=500,
        refine_stop_iter=25_000,
        refine_every=100,
        min_opacity=0.005,
        loss_threshold=0.1,
        importance_thresh=2.0,
        prune_score_thresh=0.8,
        n_sample_cameras=10,
        verbose=True,
    )
    print("✓ FastMCMC strategy instantiated successfully")

    # Test state initialization
    state = strategy.initialize_state()
    assert "binoms" in state, "State should contain binoms"
    assert state["binoms"].shape == (51, 51), "Binoms should be 51x51"
    print("✓ State initialized successfully")

    # Test sanity check with mock parameters
    params = {
        "means": torch.nn.Parameter(torch.randn(100, 3)),
        "scales": torch.nn.Parameter(torch.randn(100, 3)),
        "quats": torch.nn.Parameter(torch.randn(100, 4)),
        "opacities": torch.nn.Parameter(torch.randn(100, 1)),
    }
    optimizers = {
        "means": torch.optim.Adam([params["means"]], lr=1e-3),
        "scales": torch.optim.Adam([params["scales"]], lr=1e-3),
        "quats": torch.optim.Adam([params["quats"]], lr=1e-3),
        "opacities": torch.optim.Adam([params["opacities"]], lr=1e-3),
    }

    try:
        strategy.check_sanity(params, optimizers)
        print("✓ Sanity check passed")
    except Exception as e:
        print(f"✗ Sanity check failed: {e}")
        return False

    print("\n✅ All basic tests passed!")
    return True


def test_fastmcmc_import():
    """Test that FastMCMC can be imported from gsplat."""
    print("Testing FastMCMC import...")

    try:
        from gsplat import FastMCMCStrategy as FMS1
        from gsplat.strategy import FastMCMCStrategy as FMS2
        from gsplat.strategy.fastmcmc import FastMCMCStrategy as FMS3

        assert FMS1 is FMS2, "Import paths should be consistent"
        assert FMS2 is FMS3, "Import paths should be consistent"
        print("✓ All import paths work correctly")
        return True
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("FastMCMC Strategy Test Suite")
    print("=" * 60)
    print()

    success = True
    success &= test_fastmcmc_import()
    print()
    success &= test_fastmcmc_basic()
    print()

    if success:
        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        exit(0)
    else:
        print("=" * 60)
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        exit(1)
