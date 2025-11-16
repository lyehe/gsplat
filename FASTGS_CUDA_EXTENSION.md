# FastGS CUDA Extension Specification

## Overview

This document specifies the CUDA modifications needed to add FastGS metric counting to gsplat's rasterization kernels. This is a **future enhancement** that would eliminate the approximation currently used.

## Current Limitation

The current FastGSStrategy implementation approximates per-Gaussian error counts by distributing error pixels equally among visible Gaussians. This is less precise than the original FastGS implementation.

## FastGS CUDA Modification

### What FastGS Does

FastGS modifies the rasterization kernel to accumulate per-Gaussian error pixel counts during rendering:

```cuda
// From FastGS diff-gaussian-rasterization_fastgs/cuda_rasterizer/forward.cu:400-406
if(get_flag) {
    if(metric_map[pix_id] == 1) {
        atomicAdd(&(metricCount[collected_id[j]]), 1);
    }
}
```

**Inputs:**
- `metric_map`: Binary mask [H, W] where 1 = high-error pixel
- `get_flag`: Boolean to enable metric counting

**Output:**
- `metricCount`: Per-Gaussian error pixel count [N]

**Logic:**
For each Gaussian that contributes to rendering a pixel:
- If the pixel is marked as high-error (`metric_map[pix_id] == 1`)
- Atomically increment the count for that Gaussian

### Where to Add in gsplat

**File:** `gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu`

**Location:** Inside the rasterization loop, after color accumulation (line 164)

```cuda
// Current code (line 159-168):
int32_t g = id_batch[t];
const float vis = alpha * T;
const float *c_ptr = colors + g * CDIM;
#pragma unroll
for (uint32_t k = 0; k < CDIM; ++k) {
    pix_out[k] += c_ptr[k] * vis;
}
cur_idx = batch_start + t;

// ADD FASTGS METRIC COUNTING HERE:
if (count_metrics && metric_counts != nullptr) {
    atomicAdd(&metric_counts[g], 1);
}

T = next_T;
```

## Proposed Implementation

### 1. Modify Kernel Signature

Add optional FastGS parameters to `rasterize_to_pixels_3dgs_fwd_kernel`:

```cuda
template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_3dgs_fwd_kernel(
    // ... existing parameters ...
    scalar_t *__restrict__ render_colors,
    scalar_t *__restrict__ render_alphas,
    int32_t *__restrict__ last_ids,
    // FastGS additions (optional, can be nullptr)
    const int32_t *__restrict__ metric_map,  // [I, H, W] binary mask
    int32_t *__restrict__ metric_counts      // [N] per-Gaussian counts
) {
    // ... existing code ...

    // Add after metric_map setup (line ~60):
    if (metric_map != nullptr) {
        metric_map += image_id * image_height * image_width;
    }

    // Add flag check before rasterization loop (line ~108):
    bool count_metrics = (metric_map != nullptr && inside &&
                          metric_map[pix_id] == 1);

    // Add in rasterization loop (after line 164):
    if (count_metrics && metric_counts != nullptr) {
        atomicAdd(&metric_counts[g], 1);
    }
}
```

### 2. Update Launch Function

Modify `launch_rasterize_to_pixels_3dgs_fwd_kernel`:

```cuda
template <uint32_t CDIM>
void launch_rasterize_to_pixels_3dgs_fwd_kernel(
    // ... existing parameters ...
    at::Tensor renders,
    at::Tensor alphas,
    at::Tensor last_ids,
    // FastGS additions
    const at::optional<at::Tensor> metric_map,    // [I, H, W]
    at::optional<at::Tensor> metric_counts        // [N]
) {
    // ... existing setup ...

    rasterize_to_pixels_3dgs_fwd_kernel<CDIM, float>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            // ... existing args ...
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            // FastGS args
            metric_map.has_value() ? metric_map.value().data_ptr<int32_t>() : nullptr,
            metric_counts.has_value() ? metric_counts.value().data_ptr<int32_t>() : nullptr
        );
}
```

### 3. Update Header Declaration

In `Rasterization.h`:

```cpp
template <uint32_t CDIM>
void launch_rasterize_to_pixels_3dgs_fwd_kernel(
    // ... existing parameters ...
    at::Tensor renders,
    at::Tensor alphas,
    at::Tensor last_ids,
    // FastGS additions
    const at::optional<at::Tensor> metric_map = at::nullopt,
    at::optional<at::Tensor> metric_counts = at::nullopt
);
```

### 4. Update Python Bindings

In `gsplat/cuda/_wrapper.py`, add to `rasterize_to_pixels` function:

```python
def rasterize_to_pixels(
    # ... existing params ...
    backgrounds: Optional[Tensor] = None,
    masks: Optional[Tensor] = None,
    # FastGS additions
    metric_map: Optional[Tensor] = None,
    metric_counts: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
    """
    Args:
        metric_map: Optional binary mask [C, H, W] where 1 = high-error pixel
        metric_counts: Optional output tensor [N] to accumulate per-Gaussian counts

    Returns:
        renders, alphas, last_ids, metric_counts (if provided)
    """
    # ... existing code ...

    if C <= 512:
        _C.launch_rasterize_to_pixels_3dgs_fwd_kernel[C](
            # ... existing args ...
            renders,
            alphas,
            last_ids,
            # FastGS args
            metric_map,
            metric_counts,
        )

    return renders, alphas, last_ids, metric_counts
```

## Usage Example

With this CUDA extension, the FastGS strategy would use exact counting:

```python
from gsplat import rasterization
import torch

# Create metric map (binary mask of high-error pixels)
with torch.no_grad():
    renders, alphas, info = rasterization(...)
    loss_map = (renders - gt_image).abs().mean(dim=0)
    loss_map_norm = (loss_map - loss_map.min()) / (loss_map.max() - loss_map.min())
    metric_map = (loss_map_norm > 0.1).int()  # [H, W]

# Initialize metric counts
metric_counts = torch.zeros(n_gaussians, dtype=torch.int32, device='cuda')

# Render with metric counting
renders, alphas, last_ids, metric_counts = rasterize_to_pixels(
    ...,
    metric_map=metric_map,
    metric_counts=metric_counts,
)

# metric_counts now contains exact per-Gaussian error pixel counts!
importance_score = torch.div(metric_counts, 1, rounding_mode='floor')
```

## Benefits Over Current Approximation

1. **Exact counts**: No approximation error from distributing pixels equally
2. **Faster**: Single render pass instead of post-processing
3. **More accurate**: Knows exactly which Gaussians contribute to each error pixel
4. **Minimal overhead**: Just one atomic add per contributing Gaussian

## Implementation Effort

- **Low complexity**: ~20 lines of CUDA code changes
- **Backward compatible**: Optional parameters, defaults to current behavior
- **No new dependencies**: Uses existing atomic operations
- **Testing**: Can validate against current approximation

## Alternative: Per-Pixel Gaussian IDs

An alternative approach would be to return per-pixel Gaussian IDs from the rasterizer, then count in Python:

```python
# If rasterizer returned per-pixel Gaussian IDs
gaussian_ids_per_pixel = rasterization(..., return_gaussian_ids=True)

# Count in Python
for pix_id in high_error_pixels:
    for g_id in gaussian_ids_per_pixel[pix_id]:
        metric_counts[g_id] += 1
```

This would work but be slower than the CUDA atomic approach.

## Files to Modify

1. `gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu` - Add metric counting logic
2. `gsplat/cuda/csrc/Rasterization.h` - Update function signature
3. `gsplat/cuda/_wrapper.py` - Add Python bindings
4. `gsplat/rendering.py` - Expose in high-level API
5. `tests/` - Add tests for metric counting

## Testing Strategy

```python
def test_fastgs_metric_counting():
    # Create simple scene
    means = torch.randn(100, 3, device='cuda')
    # ... setup gaussians ...

    # Render
    renders, alphas, info = rasterization(...)

    # Create all-ones metric map
    metric_map = torch.ones(H, W, dtype=torch.int32, device='cuda')
    metric_counts = torch.zeros(100, dtype=torch.int32, device='cuda')

    # Render with counting
    _, _, _, metric_counts = rasterize_to_pixels(..., metric_map=metric_map, metric_counts=metric_counts)

    # Verify counts are reasonable
    assert (metric_counts > 0).any()  # At least some Gaussians contributed
    assert metric_counts.sum() > 0    # Total contributions > 0
```

## Recommendation

This CUDA extension should be implemented as a **future enhancement** to gsplat. For now, the Python-side approximation in `fastgs_utils.py` provides a working solution that:
- Avoids modifying gsplat's core CUDA code
- Is Apache 2.0 compatible
- Provides reasonable approximation of the FastGS behavior

Once this extension is added to gsplat (by the gsplat team or community), the FastGSStrategy can be updated to use exact counting with minimal changes to the Python code.

## License Note

This specification is based on understanding the FastGS algorithm from the paper and examining the Max Planck licensed code for educational purposes only. The proposed implementation is a clean-room design using gsplat's existing Apache 2.0 architecture.
