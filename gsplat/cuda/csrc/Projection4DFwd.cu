/*
 * Copyright (C) 2024, gsplat team
 *
 * This file implements 4D screenspace projection for FastGS dual gradient tracking.
 * Based on ProjectionEWA3DGSFused.cu but outputs 4D points [x, y, depth, scale_metric]
 * instead of separate 2D points + depths.
 */

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Projection.h"
#include "Utils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

/**
 * Compute scale metric for 4th component of screenspace points.
 *
 * This metric captures how much the Gaussian's scale projects to screen space.
 * High gradients in this component indicate the Gaussian needs splitting.
 *
 * @param scales: [3] Gaussian scales in object space
 * @param R: [3, 3] Rotation matrix from viewmat
 * @param z: depth in camera space
 * @return scale metric (screen-space scale magnitude)
 */
inline __device__ float compute_scale_metric(
    const vec3 &scales,
    const mat3 &R,
    float z
) {
    // Transform scales to camera space
    vec3 scale_cam = R * scales;

    // Project to screen space: scale in screen = scale in cam / depth
    // We use the xy magnitude (ignoring z scale)
    float scale_screen = glm::length(vec2(scale_cam.x, scale_cam.y)) / (z + 1e-6f);

    return scale_screen;
}

/**
 * Forward pass: Project 3D Gaussians to 4D screenspace points.
 *
 * Output format [N, 4]:
 *   [0]: screen x coordinate
 *   [1]: screen y coordinate
 *   [2]: depth (z in camera space)
 *   [3]: scale metric (screen-space scale magnitude)
 *
 * This enables dual gradient tracking: xy gradients vs depth/scale gradients.
 */
template <typename scalar_t>
__global__ void projection_4d_fused_fwd_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const scalar_t *__restrict__ means,    // [B, N, 3]
    const scalar_t *__restrict__ covars,   // [B, N, 6] optional
    const scalar_t *__restrict__ quats,    // [B, N, 4] optional
    const scalar_t *__restrict__ scales,   // [B, N, 3] optional
    const scalar_t *__restrict__ opacities, // [B, N] optional
    const scalar_t *__restrict__ viewmats, // [B, C, 4, 4]
    const scalar_t *__restrict__ Ks,       // [B, C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const CameraModelType camera_model,
    // outputs
    int32_t *__restrict__ radii,           // [B, C, N, 2]
    scalar_t *__restrict__ means4d,        // [B, C, N, 4] - 4D output!
    scalar_t *__restrict__ depths,         // [B, C, N] - redundant but kept for compatibility
    scalar_t *__restrict__ conics,         // [B, C, N, 3]
    scalar_t *__restrict__ compensations   // [B, C, N] optional
) {
    // parallelize over B * C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= B * C * N) {
        return;
    }
    const uint32_t bid = idx / (C * N); // batch id
    const uint32_t cid = (idx / N) % C; // camera id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += bid * N * 3 + gid * 3;
    viewmats += bid * C * 16 + cid * 16;
    Ks += bid * C * 9 + cid * 9;

    // glm is column-major but input is row-major
    mat3 R = mat3(
        viewmats[0],
        viewmats[4],
        viewmats[8], // 1st column
        viewmats[1],
        viewmats[5],
        viewmats[9], // 2nd column
        viewmats[2],
        viewmats[6],
        viewmats[10] // 3rd column
    );
    vec3 t = vec3(viewmats[3], viewmats[7], viewmats[11]);

    // transform Gaussian center to camera space
    vec3 mean_c;
    posW2C(R, t, glm::make_vec3(means), mean_c);
    if (mean_c.z < near_plane || mean_c.z > far_plane) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // transform Gaussian covariance to camera space
    mat3 covar;
    vec3 scale; // Need to keep scale for metric computation
    if (covars != nullptr) {
        covars += bid * N * 6 + gid * 6;
        covar = mat3(
            covars[0],
            covars[1],
            covars[2], // 1st column
            covars[1],
            covars[3],
            covars[4], // 2nd column
            covars[2],
            covars[4],
            covars[5] // 3rd column
        );
        // For covars, we can't compute scale metric properly
        // Use a simple approximation from covariance eigenvalues
        scale = vec3(
            sqrtf(covar[0][0]),
            sqrtf(covar[1][1]),
            sqrtf(covar[2][2])
        );
    } else {
        // compute from quaternions and scales
        quats += bid * N * 4 + gid * 4;
        scales += bid * N * 3 + gid * 3;
        scale = glm::make_vec3(scales);
        quat_scale_to_covar_preci(
            glm::make_vec4(quats), scale, &covar, nullptr
        );
    }
    mat3 covar_c;
    covarW2C(R, covar, covar_c);

    // perspective projection
    mat2 covar2d;
    vec2 mean2d;

    switch (camera_model) {
    case CameraModelType::PINHOLE: // perspective projection
        persp_proj(
            mean_c,
            covar_c,
            Ks[0],
            Ks[4],
            Ks[2],
            Ks[5],
            image_width,
            image_height,
            covar2d,
            mean2d
        );
        break;
    case CameraModelType::ORTHO: // orthographic projection
        ortho_proj(
            mean_c,
            covar_c,
            Ks[0],
            Ks[4],
            Ks[2],
            Ks[5],
            image_width,
            image_height,
            covar2d,
            mean2d
        );
        break;
    case CameraModelType::FISHEYE: // fisheye projection
        fisheye_proj(
            mean_c,
            covar_c,
            Ks[0],
            Ks[4],
            Ks[2],
            Ks[5],
            image_width,
            image_height,
            covar2d,
            mean2d
        );
        break;
    }

    float compensation;
    float det = add_blur(eps2d, covar2d, compensation);
    if (det <= 0.f) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // compute the inverse of the 2d covariance
    mat2 covar2d_inv = glm::inverse(covar2d);

    float extend = 3.33f;
    if (opacities != nullptr) {
        float opacity = opacities[bid * N + gid];
        if (compensations != nullptr) {
            // we assume compensation term will be applied later on.
            opacity *= compensation;
        }
        if (opacity < ALPHA_THRESHOLD) {
            radii[idx * 2] = 0;
            radii[idx * 2 + 1] = 0;
            return;
        }
        // Compute opacity-aware bounding box.
        // https://arxiv.org/pdf/2402.00525 Section B.2
        extend = min(extend, sqrt(2.0f * __logf(opacity / ALPHA_THRESHOLD)));
    }

    // compute tight rectangular bounding box (non differentiable)
    // https://arxiv.org/pdf/2402.00525
    float radius_x = ceilf(extend * sqrtf(covar2d[0][0]));
    float radius_y = ceilf(extend * sqrtf(covar2d[1][1]));

    if (radius_x <= radius_clip && radius_y <= radius_clip) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // mask out gaussians outside the image region
    if (mean2d.x + radius_x <= 0 || mean2d.x - radius_x >= image_width ||
        mean2d.y + radius_y <= 0 || mean2d.y - radius_y >= image_height) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // Compute scale metric for 4th component
    float scale_metric = compute_scale_metric(scale, R, mean_c.z);

    // write to outputs
    radii[idx * 2] = (int32_t)radius_x;
    radii[idx * 2 + 1] = (int32_t)radius_y;

    // 4D screenspace point [x, y, depth, scale_metric]
    means4d[idx * 4] = mean2d.x;
    means4d[idx * 4 + 1] = mean2d.y;
    means4d[idx * 4 + 2] = mean_c.z;         // depth
    means4d[idx * 4 + 3] = scale_metric;     // scale metric

    // Also write to depths for compatibility with 2D code
    depths[idx] = mean_c.z;

    conics[idx * 3] = covar2d_inv[0][0];
    conics[idx * 3 + 1] = covar2d_inv[0][1];
    conics[idx * 3 + 2] = covar2d_inv[1][1];
    if (compensations != nullptr) {
        compensations[idx] = compensation;
    }
}

} // namespace gsplat
