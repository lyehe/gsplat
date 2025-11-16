# FastMCMC Implementation Validation Report

## Executive Summary

**Status**: ⚠️ **CRITICAL BUG FOUND** - Shape handling error in `_add_new_gs_fastmcmc`

The FastMCMC implementation has several issues that need to be addressed:
- **1 Critical Bug**: Incorrect tensor shape manipulation in sampling
- **2 Major Issues**: Missing input validation, potential inefficiency
- **3 Minor Issues**: Documentation inconsistencies

## Detailed Analysis

### ✅ What's Correct

1. **Overall Algorithm Structure**: Sound hybrid approach combining MCMC + FastGS
2. **MCMC Base Implementation**: Correctly uses `relocate`, `sample_add`, and `inject_noise_to_position`
3. **Binomial Coefficients**: Properly initialized (51x51 lookup table)
4. **Fallback Behavior**: Gracefully falls back to vanilla MCMC when scores missing
5. **Integration**: Properly exported in `__init__.py` files
6. **Hyperparameters**: Reasonable defaults matching MCMC and FastGS

---

## 🔴 Critical Bug

### Bug #1: Unnecessary unsqueeze/squeeze in `_add_new_gs_fastmcmc`

**Location**: `gsplat/strategy/fastmcmc.py:328` and `gsplat/strategy/fastmcmc.py:347`

**Issue**:
```python
# Line 328 - WRONG: adds unnecessary dimension
new_opacities, new_scales = compute_relocation(
    opacities=opacities[sampled_idxs].unsqueeze(-1),  # ❌ Makes it [N, 1]
    scales=torch.exp(params["scales"])[sampled_idxs],
    ratios=torch.bincount(sampled_idxs, minlength=len(opacities))[sampled_idxs] + 1,
    binoms=binoms,
)

# Line 347 - Then has to squeeze it back
if name == "opacities":
    p[sampled_idxs] = torch.logit(new_opacities.squeeze(-1))  # ❌ Squeezes back
```

**Why it's wrong**:
- `compute_relocation` expects `opacities` with shape `[N]`, not `[N, 1]`
- The vanilla MCMC implementations (`relocate`, `sample_add`) don't use `unsqueeze`
- Adding/removing dimensions unnecessarily is error-prone

**Comparison with vanilla MCMC**:
```python
# From ops.py sample_add (lines 314-324) - CORRECT
new_opacities, new_scales = compute_relocation(
    opacities=opacities[sampled_idxs],  # ✅ Shape [N]
    scales=torch.exp(params["scales"])[sampled_idxs],
    ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
    binoms=binoms,
)

def param_fn(name: str, p: Tensor) -> Tensor:
    if name == "opacities":
        p[sampled_idxs] = torch.logit(new_opacities)  # ✅ No squeeze needed
```

**Impact**:
- May cause shape mismatch errors or silent broadcasting issues
- Inconsistent with gsplat conventions
- Potentially breaks with certain parameter shapes

**Fix**:
```python
# Line 328 - Remove .unsqueeze(-1)
new_opacities, new_scales = compute_relocation(
    opacities=opacities[sampled_idxs],  # ✅ Correct
    scales=torch.exp(params["scales"])[sampled_idxs],
    ratios=torch.bincount(sampled_idxs, minlength=len(opacities))[sampled_idxs] + 1,
    binoms=binoms,
)

# Line 347 - Remove .squeeze(-1)
if name == "opacities":
    p[sampled_idxs] = torch.logit(new_opacities)  # ✅ Correct
```

---

## ⚠️ Major Issues

### Issue #1: Missing Input Validation for Multi-View Scores

**Location**: `_relocate_gs_fastmcmc` (line 252), `_prune_gs_fastmcmc` (line 373)

**Problem**: No validation that `importance_score` and `pruning_score` have correct shapes

```python
def _relocate_gs_fastmcmc(self, params, optimizers, binoms, info):
    opacities = torch.sigmoid(params["opacities"].flatten())
    importance_score = info["importance_score"]  # ❌ No shape check!

    # If importance_score has wrong shape, this will fail cryptically
    low_importance_mask = importance_score < self.importance_thresh
    relocate_mask = low_opacity_mask | low_importance_mask  # ❌ Shape mismatch possible
```

**Expected behavior**:
- `importance_score` should have shape `[N]` where N = number of Gaussians
- `pruning_score` should have shape `[N]`

**Potential failures**:
- If scores have wrong shape: RuntimeError during boolean operations
- If scores are shorter: IndexError
- If scores are longer: Silent broadcasting issues

**Fix**: Add validation
```python
def _relocate_gs_fastmcmc(self, params, optimizers, binoms, info):
    opacities = torch.sigmoid(params["opacities"].flatten())
    importance_score = info["importance_score"]

    # ✅ Validate shape
    n_gaussians = len(opacities)
    assert importance_score.shape == (n_gaussians,), (
        f"importance_score shape mismatch: expected ({n_gaussians},), "
        f"got {importance_score.shape}"
    )

    low_opacity_mask = opacities <= self.min_opacity
    low_importance_mask = importance_score < self.importance_thresh
    relocate_mask = low_opacity_mask | low_importance_mask
```

### Issue #2: Inefficient Relocation of Gaussians That Will Be Pruned

**Location**: `step_post_backward` (lines 191-197)

**Problem**: Gaussians may be relocated then immediately pruned

**Scenario**:
1. A Gaussian has `low_importance` → gets relocated (moved to new position, opacity updated)
2. Same Gaussian has `high_redundancy` → gets pruned immediately after

**Result**: Wasted computation relocating a Gaussian that gets deleted

**Example**:
```python
# Gaussian 42:
# - importance_score = 1.5 (< 2.0 threshold) → will relocate
# - pruning_score = 0.85 (> 0.8 threshold) → will prune
#
# Execution:
# 1. Relocate Gaussian 42 to new location (expensive)
# 2. Update its opacity and scales (expensive)
# 3. Prune Gaussian 42 (deletes it)
#
# Result: Wasted work!
```

**Impact**:
- Not a bug (still produces correct results)
- Reduces efficiency, especially with aggressive pruning thresholds
- May matter for very large scenes (1M+ Gaussians)

**Potential fix** (not critical):
```python
# Check prune candidates first, exclude from relocation
prune_mask = low_opacity_mask | high_redundancy_mask
relocate_mask = (low_opacity_mask | low_importance_mask) & ~prune_mask
```

---

## 📝 Minor Issues

### Issue #1: Inconsistent `minlength` in bincount

**Location**: Line 330

```python
ratios=torch.bincount(sampled_idxs, minlength=len(opacities))[sampled_idxs] + 1,
```

vs vanilla `sample_add` (line 317):
```python
ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
```

**Analysis**:
- `minlength=len(opacities)` makes bincount return a tensor of length N (all Gaussians)
- Then we index with `[sampled_idxs]` to get only the counts we need
- This is **correct but wasteful** - creates a larger tensor than needed

**Impact**: Minor memory overhead, no functional issue

**Fix** (optional):
```python
ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,  # More efficient
```

### Issue #2: Documentation says "CORRECTION" in commit message

The latest commit message says:
> "CORRECTION: Fix FastGS to match actual hybrid algorithm"

But this was for FastGS, not FastMCMC. The FastMCMC commit doesn't mention it's a novel strategy.

**Impact**: None (just documentation)

### Issue #3: No handling of edge case when all Gaussians should be pruned

If `prune_mask.sum() == len(params["means"])`, all Gaussians would be pruned, leaving the scene empty.

**Impact**: Unlikely in practice (would only happen with extreme hyperparameters)

**Potential safeguard**:
```python
# Don't prune if it would remove everything
if n_prune < len(params["means"]):
    remove(params=params, optimizers=optimizers, state=state, mask=prune_mask)
else:
    print("Warning: Attempted to prune all Gaussians, skipping")
```

---

## Algorithm Validation

### ✅ MCMC Components

Compared against `gsplat/strategy/mcmc.py`:

| Component | MCMC | FastMCMC | Status |
|-----------|------|----------|--------|
| `initialize_state()` | Binomial 51x51 | Binomial 51x51 | ✅ Identical |
| `check_sanity()` | means/scales/quats/opacities | means/scales/quats/opacities | ✅ Identical |
| `inject_noise_to_position()` | Every iter | Every iter | ✅ Identical |
| Relocation condition | `opacity <= min_opacity` | `opacity <= min_opacity OR importance < thresh` | ✅ Extension |
| Sampling weights | `opacity` | `opacity * (1 + importance_norm)` | ✅ Extension |
| Pruning | None | `opacity <= min_opacity OR pruning > thresh` | ✅ Extension |

**Verdict**: FastMCMC correctly extends MCMC base logic

### ✅ FastGS Multi-View Filtering

Compared against `gsplat/strategy/fastgs.py`:

| Component | FastGS | FastMCMC | Status |
|-----------|--------|----------|--------|
| Multi-view scoring | ✅ Uses | ✅ Uses | ✅ Match |
| `importance_score` usage | Densify where `importance > 5` | Relocate/sample where importance relevant | ✅ Adapted |
| `pruning_score` usage | Prune where `pruning > 0.8` | Prune where `pruning > 0.8` | ✅ Match |
| Gradient accumulation | ✅ Required | ❌ Not used | ✅ Intentional (MCMC is gradient-free) |

**Verdict**: FastMCMC correctly adapts FastGS filtering for gradient-free MCMC

### ✅ Hybrid Logic Correctness

**Relocation** (low opacity OR low importance):
- ✅ Makes sense: Move dead Gaussians OR under-reconstructed Gaussians
- ✅ Uses MCMC `relocate()` with multi-view filtering

**Sampling** (weighted by opacity AND importance):
- ✅ Makes sense: Sample where Gaussians are confident AND errors are high
- ✅ Uses MCMC `sample_add()` logic with weighted probabilities

**Pruning** (low opacity OR high redundancy):
- ✅ Makes sense: Remove dead OR redundant Gaussians
- ✅ Uses MCMC-style removal with multi-view filtering

**Noise injection**:
- ✅ Preserved from MCMC (every iteration)
- ✅ Maintains exploration properties

**Verdict**: Algorithm logic is sound

---

## Compatibility Analysis

### ✅ 3DGut Compatibility

**Claim**: "Compatible with 3DGut (no gradients needed)"

**Validation**:
- FastMCMC doesn't use `absgrad` or gradient accumulation ✅
- Only requires rendering outputs and multi-view consistency ✅
- Multi-view scoring works with any rendering backend ✅

**Verdict**: ✅ **Compatible** - No technical blockers for 3DGut

### ✅ Integration with gsplat

**ops.py integration**:
- Uses `inject_noise_to_position` ✅
- Uses `relocate` ✅
- Uses `sample_add` logic ✅
- Uses `remove` ✅
- Uses `_multinomial_sample` ✅
- Uses `_update_param_with_optimizer` ✅

**relocation.py integration**:
- Uses `compute_relocation` ✅ (but with shape bug)

**Verdict**: ✅ Properly integrated with gsplat infrastructure

---

## Recommendations

### Critical (Must Fix)

1. **Remove unsqueeze/squeeze in `_add_new_gs_fastmcmc`**
   - Line 328: Remove `.unsqueeze(-1)`
   - Line 347: Remove `.squeeze(-1)`
   - This is the most important fix

### High Priority (Should Fix)

2. **Add input validation for multi-view scores**
   - Validate shape of `importance_score` matches number of Gaussians
   - Validate shape of `pruning_score` matches number of Gaussians
   - Add helpful error messages

### Medium Priority (Nice to Have)

3. **Optimize relocation order**
   - Check prune candidates before relocation
   - Avoid relocating Gaussians that will be pruned
   - Only matters for large scenes with aggressive pruning

4. **Add safeguards for edge cases**
   - Don't prune if it would remove all Gaussians
   - Warn if relocation/pruning affects >50% of Gaussians

### Low Priority (Optional)

5. **Remove `minlength` from bincount**
   - Minor memory optimization
   - Makes code consistent with vanilla MCMC

---

## Testing Checklist

Before merging, test:

- [ ] Basic instantiation and initialization
- [ ] Relocation with multi-view scores
- [ ] Sampling with weighted probabilities
- [ ] Pruning with redundancy scores
- [ ] Fallback to vanilla MCMC when scores missing
- [ ] Shape validation with different Gaussian counts
- [ ] Edge case: All Gaussians have low importance
- [ ] Edge case: All Gaussians have high redundancy
- [ ] Integration with 3DGut rendering
- [ ] Memory usage with 1M+ Gaussians

---

## Conclusion

FastMCMC is a **sound algorithm** with **correct hybrid logic**, but has **1 critical shape handling bug** that must be fixed before deployment.

**Overall Assessment**: ⚠️ **Needs fixes before production use**

**Confidence**: High - Compared against vanilla MCMC and FastGS implementations

**Next Steps**:
1. Fix the unsqueeze/squeeze bug (critical)
2. Add input validation (high priority)
3. Consider optimization for relocation order (medium priority)
4. Test thoroughly with real datasets
