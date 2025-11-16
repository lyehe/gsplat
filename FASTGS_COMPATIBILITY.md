# FastGS Compatibility with gsplat Features

## Summary Table

| Feature | Compatible? | Status | Notes |
|---------|-------------|--------|-------|
| **MCMC** | ❌ No | Fundamentally incompatible | MCMC uses probabilistic sampling, FastGS uses gradients |
| **3DGut** | ✅ Yes* | Should work, untested | No technical blockers found |
| **DefaultStrategy** | ✅ Yes | By design | FastGS extends DefaultStrategy logic |
| **Distributed** | ✅ Yes | Should work | FastGS is strategy-level, distributed is rendering-level |
| **Sparse Gradients** | ✅ Yes | Should work | FastGS compatible with sparse_grad=True |
| **AbsGrad** | ✅ Yes | Required | FastGS uses absgrad=True by default |

---

## Detailed Analysis

### 1. MCMC Strategy ❌

**Why they're incompatible:**

MCMC and FastGS are **competing densification strategies** that use fundamentally different approaches:

**MCMC (Probabilistic):**
```python
# From gsplat/strategy/mcmc.py
class MCMCStrategy:
    def step_post_backward(...):
        # 1. Relocate low-opacity Gaussians (opacity-based)
        self._relocate_gs(params, optimizers, binoms)

        # 2. Sample new Gaussians (opacity distribution)
        self._add_new_gs(params, optimizers, binoms)

        # 3. Inject noise (MCMC sampling)
        inject_noise_to_position(params, optimizers, scaler=lr * noise_lr)

        # NO gradient accumulation!
        # NO split/clone based on gradients!
```

**FastGS (Gradient + Multi-View Hybrid):**
```python
# From gsplat/strategy/fastgs.py
class FastGSStrategy:
    def step_pre_backward(...):
        info["means2d"].retain_grad()  # REQUIRES gradients

    def step_post_backward(...):
        # Accumulate gradients
        grads = state["grad2d"] / state["count"]

        # Gradient-based candidates
        is_grad_high = grads > 0.0002

        # Multi-view filtering
        is_high_error = importance_score > 5.0

        # Densify where BOTH conditions met
        densify_where = is_grad_high & is_high_error
```

**Conclusion:** You must choose one or the other. Cannot use both simultaneously.

---

### 2. 3DGut (3D Gaussian Unscented Transform) ✅*

**What 3DGut does:**
- Enables distorted camera models (fisheye, rolling shutter)
- Supports secondary lighting (reflections, refractions)
- Uses Unscented Transform for nonlinear projections
- Operates at the **rendering/projection level**

**What FastGS does:**
- Hybrid gradient + multi-view densification
- Operates at the **strategy/densification level**

**Why they should be compatible:**

```
┌─────────────────────────────────┐
│  Rendering Level (3DGut)        │ ← Modifies projection & rasterization
│  - with_ut=True                 │
│  - with_eval3d=True             │
│  - Distortion parameters        │
└─────────────────────────────────┘
              ↓
      Gradient Flow
              ↓
┌─────────────────────────────────┐
│  Strategy Level (FastGS)        │ ← Uses gradients for densification
│  - Accumulate gradients         │
│  - Multi-view scoring           │
│  - Hybrid densification         │
└─────────────────────────────────┘
```

They operate at **different levels** with no direct coupling!

**Technical verification:**
```bash
# Check for coupling
$ grep -r "with_ut\|with_eval3d" gsplat/strategy/
# Result: No matches! Strategies don't reference 3DGut flags

$ grep "absgrad" gsplat/rendering.py
# Result: absgrad is independent parameter, works with all rendering modes
```

**Current limitation:**
The gsplat docs say:
> "note in gsplat we only support MCMC densification strategy for 3DGUT"

This is a **documentation/testing limitation**, not a technical one. It likely means:
- 3DGut was only tested with MCMC
- Nobody has tried FastGS + 3DGut yet
- Conservative documentation

**How to use FastGS + 3DGut:**

```python
from gsplat import FastGSStrategy, rasterization
from gsplat.strategy.fastgs_utils import sample_cameras, compute_gaussian_score_fastgs

# Initialize FastGS strategy
strategy = FastGSStrategy(
    absgrad=True,  # Required for FastGS
    verbose=True,
)
strategy_state = strategy.initialize_state()

for step in range(30_000):
    # Render with 3DGut enabled
    renders, alphas, info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        # 3DGut flags
        with_ut=True,              # Enable Unscented Transform
        with_eval3d=True,          # Enable 3D evaluation
        radial_coeffs=radial,      # Distortion parameters
        camera_model="fisheye",    # Or "pinhole", "ftheta"
        # FastGS requirements
        absgrad=True,              # Absolute gradients
        packed=True,
    )

    # FastGS strategy callbacks
    strategy.step_pre_backward(params, optimizers, strategy_state, step, info)

    loss = compute_loss(renders, gt_images)
    loss.backward()

    # Multi-view scoring every 100 iterations
    if step % 100 == 0 and step > 500:
        camlist = sample_cameras(train_cameras, n_samples=10)
        importance_score, pruning_score = compute_gaussian_score_fastgs(
            camlist,
            lambda cam: rasterization(..., with_ut=True, with_eval3d=True),
            n_gaussians=len(params["means"]),
        )
        info["importance_score"] = importance_score
        info["pruning_score"] = pruning_score

    strategy.step_post_backward(params, optimizers, strategy_state, step, info)

    for opt in optimizers.values():
        opt.step()
        opt.zero_grad()
```

**Expected behavior:**
- ✅ Gradients flow correctly through 3DGut projection
- ✅ FastGS accumulates gradients as normal
- ✅ Multi-view scoring works with distorted cameras
- ✅ Densification uses hybrid approach
- ⚠️ May need to tune `loss_threshold` for distorted images

**Potential issues:**
1. **Performance**: 3DGut rendering is slower, multi-view scoring will be slower
2. **Hyperparameters**: May need different thresholds for distorted cameras
3. **Untested**: This combination hasn't been validated

---

### 3. Other Features

**Distributed Training** ✅
- FastGS operates at strategy level
- Distributed is at rendering level
- Should work together

**Sparse Gradients** ✅
- FastGS accumulates gradients normally
- Compatible with `sparse_grad=True`
- May reduce memory usage

**AbsGrad** ✅
- FastGS **requires** `absgrad=True` by default
- Uses absolute gradients for splitting (0.0012 threshold)
- Already part of FastGS design

---

## Recommendations

### For MCMC Users
If you need MCMC's probabilistic approach, stick with MCMCStrategy. FastGS is fundamentally different.

### For 3DGut Users
**You can try FastGS + 3DGut!** There are no technical blockers:

1. Use the code example above
2. Start with standard FastGS hyperparameters
3. Monitor Gaussian count growth
4. Tune `loss_threshold` if needed (distorted cameras may have different error distributions)
5. Report results back to the community!

### Contribution Opportunity
If you successfully use FastGS + 3DGut:
1. Validate that it works
2. Benchmark against MCMC + 3DGut
3. Submit results/docs to gsplat
4. Help remove the "only MCMC" limitation from docs

---

## Testing Guide

To test FastGS + 3DGut compatibility:

```bash
# 1. Modify simple_trainer.py or create new script
python examples/simple_trainer_fastgs_3dgut.py \
    --data_dir data/garden \
    --with_ut \
    --with_eval3d \
    --camera_model fisheye \
    --strategy fastgs \
    --verbose

# 2. Monitor metrics
# - Gaussian count should grow controlled (FastGS filtering)
# - PSNR should improve with distortion handling (3DGut)
# - Training should complete without errors

# 3. Compare against baselines
# - MCMC + 3DGut (current recommended)
# - DefaultStrategy + 3DGut (if it works)
# - FastGS + standard projection
```

---

## Conclusion

- **MCMC + FastGS**: ❌ Incompatible (competing strategies)
- **3DGut + FastGS**: ✅ Should work (different levels, no coupling)
- **Recommendation**: Try it and report results!

The "only MCMC" limitation for 3DGut appears to be documentation/testing, not technical. FastGS should work with 3DGut since they operate at different levels of the pipeline.
