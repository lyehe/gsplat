/*
 * Copyright (C) 2024, gsplat team
 *
 * This file implements backward pass for 4D screenspace projection (FastGS dual gradients).
 * Computes gradients w.r.t. the 4th component (scale_metric) in addition to standard 2D projection.
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
 * Vector-Jacobian Product for scale_metric computation.
 *
 * Forward:
 *   scale_cam = R * scales
 *   scale_screen = ||scale_cam_{xy}|| / z
 *
 * Backward:
 *   Given v_scale_screen (gradient w.r.t. output), compute:
 *   - v_scales: gradient w.r.t. input scales
 *   - v_R: gradient w.r.t. rotation matrix
 *   - v_z: gradient w.r.t. depth
 */
inline __device__ void compute_scale_metric_vjp(
    const vec3 &scales,
    const mat3 &R,
    float z,
    float v_scale_screen,
    vec3 &v_scales,
    mat3 &v_R,
    float &v_z
) {
    // Forward computation (recompute for efficiency)
    vec3 scale_cam = R * scales;

    // Compute norm of xy components
    float norm_xy = glm::length(vec2(scale_cam.x, scale_cam.y));

    // Handle numerical stability
    if (norm_xy < 1e-6f) {
        // If norm is too small, gradients are ill-defined; set to zero
        v_scales = vec3(0.0f);
        v_R = mat3(0.0f);
        v_z = 0.0f;
        return;
    }

    // ∂scale_screen/∂z = -||scale_cam_{xy}|| / z^2
    v_z = -v_scale_screen * norm_xy / (z * z + 1e-6f);

    // ∂scale_screen/∂scale_cam
    // scale_screen = norm_xy / z
    // ∂norm_xy/∂scale_cam.x = scale_cam.x / norm_xy
    // ∂norm_xy/∂scale_cam.y = scale_cam.y / norm_xy
    // ∂norm_xy/∂scale_cam.z = 0
    vec3 v_scale_cam = vec3(
        v_scale_screen * scale_cam.x / (norm_xy * z + 1e-6f),
        v_scale_screen * scale_cam.y / (norm_xy * z + 1e-6f),
        0.0f
    );

    // ∂scale_cam/∂scales = R^T
    // So v_scales = R^T * v_scale_cam
    v_scales = glm::transpose(R) * v_scale_cam;

    // ∂scale_cam/∂R: scale_cam_i = sum_j R[i][j] * scales[j]
    // ∂scale_cam_i/∂R[i][j] = scales[j]
    // So v_R[i][j] += v_scale_cam[i] * scales[j]
    // This is the outer product: v_R = v_scale_cam ⊗ scales
    v_R = glm::outerProduct(v_scale_cam, scales);
}

/**
 * Backward pass: Compute gradients for 4D projection.
 *
 * Takes gradients w.r.t. 4D output [x, y, depth, scale_metric] and computes
 * gradients w.r.t. input parameters (means, quats, scales).
 */
template <typename scalar_t>
__global__ void projection_4d_fused_bwd_kernel(
    // fwd inputs
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const scalar_t *__restrict__ means,    // [B, N, 3]
    const scalar_t *__restrict__ covars,   // [B, N, 6] optional
    const scalar_t *__restrict__ quats,    // [B, N, 4] optional
    const scalar_t *__restrict__ scales,   // [B, N, 3] optional
    const scalar_t *__restrict__ viewmats, // [B, C, 4, 4]
    const scalar_t *__restrict__ Ks,       // [B, C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const int32_t *__restrict__ radii,          // [B, C, N, 2]
    const scalar_t *__restrict__ conics,        // [B, C, N, 3]
    const scalar_t *__restrict__ compensations, // [B, C, N] optional
    // grad outputs
    const scalar_t *__restrict__ v_means4d,       // [B, C, N, 4]
    const scalar_t *__restrict__ v_depths,        // [B, C, N]
    const scalar_t *__restrict__ v_conics,        // [B, C, N, 3]
    const scalar_t *__restrict__ v_compensations, // [B, C, N] optional
    const bool viewmats_requires_grad,
    // grad inputs
    scalar_t *__restrict__ v_means,    // [B, N, 3]
    scalar_t *__restrict__ v_covars,   // [B, N, 6] optional
    scalar_t *__restrict__ v_quats,    // [B, N, 4] optional
    scalar_t *__restrict__ v_scales,   // [B, N, 3] optional
    scalar_t *__restrict__ v_viewmats  // [B, C, 4, 4] optional
) {
    // parallelize over B * C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= B * C * N || radii[idx * 2] <= 0 || radii[idx * 2 + 1] <= 0) {
        return;
    }
    const uint32_t bid = idx / (C * N); // batch id
    const uint32_t cid = (idx / N) % C; // camera id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += bid * N * 3 + gid * 3;
    viewmats += bid * C * 16 + cid * 16;
    Ks += bid * C * 9 + cid * 9;

    conics += idx * 3;

    v_means4d += idx * 4;
    v_conics += idx * 3;

    // vjp: compute the inverse of the 2d covariance
    mat2 covar2d_inv = mat2(conics[0], conics[1], conics[1], conics[2]);
    mat2 v_covar2d_inv =
        mat2(v_conics[0], v_conics[1] * .5f, v_conics[1] * .5f, v_conics[2]);
    mat2 v_covar2d(0.f);

    // inverse VJP: if Y = inv(X), then dL/dX = -Y^T * dL/dY * Y^T
    mat2 conic_t = glm::transpose(covar2d_inv);
    v_covar2d = -conic_t * v_covar2d_inv * conic_t;

    if (v_compensations != nullptr && compensations != nullptr) {
        // vjp: compensation term
        const float compensation = compensations[idx];
        const float v_compensation = v_compensations[idx];
        add_blur_vjp(
            eps2d, covar2d_inv, compensation, v_compensation, v_covar2d
        );
    }

    // transform Gaussian to camera space
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

    mat3 covar;
    vec4 quat;
    vec3 scale;
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
        scale = vec3(
            sqrtf(covar[0][0]),
            sqrtf(covar[1][1]),
            sqrtf(covar[2][2])
        );
    } else {
        // compute from quaternions and scales
        quat = glm::make_vec4(quats + bid * N * 4 + gid * 4);
        scale = glm::make_vec3(scales + bid * N * 3 + gid * 3);
        quat_scale_to_covar_preci(quat, scale, &covar, nullptr);
    }
    vec3 mean_c;
    posW2C(R, t, glm::make_vec3(means), mean_c);
    mat3 covar_c;
    covarW2C(R, covar, covar_c);

    // vjp: perspective projection
    float fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
    mat3 v_covar_c(0.f);
    vec3 v_mean_c(0.f);

    vec2 v_mean2d = vec2(v_means4d[0], v_means4d[1]);

    switch (camera_model) {
    case CameraModelType::PINHOLE: // perspective projection
        persp_proj_vjp(
            mean_c,
            covar_c,
            fx,
            fy,
            cx,
            cy,
            image_width,
            image_height,
            v_covar2d,
            v_mean2d,
            v_mean_c,
            v_covar_c
        );
        break;
    case CameraModelType::ORTHO: // orthographic projection
        ortho_proj_vjp(
            mean_c,
            covar_c,
            fx,
            fy,
            cx,
            cy,
            image_width,
            image_height,
            v_covar2d,
            v_mean2d,
            v_mean_c,
            v_covar_c
        );
        break;
    case CameraModelType::FISHEYE: // fisheye projection
        fisheye_proj_vjp(
            mean_c,
            covar_c,
            fx,
            fy,
            cx,
            cy,
            image_width,
            image_height,
            v_covar2d,
            v_mean2d,
            v_mean_c,
            v_covar_c
        );
        break;
    }

    // add contribution from v_depths (v_means4d[2])
    v_mean_c.z += v_means4d[2];

    // add contribution from separate v_depths if provided
    if (v_depths != nullptr) {
        v_mean_c.z += v_depths[idx];
    }

    // NEW: Add gradient from scale_metric component (v_means4d[3])
    float v_scale_metric = v_means4d[3];

    vec3 v_scales_from_metric(0.0f);
    mat3 v_R_from_metric(0.0f);
    float v_z_from_metric = 0.0f;

    compute_scale_metric_vjp(
        scale, R, mean_c.z,
        v_scale_metric,
        v_scales_from_metric,
        v_R_from_metric,
        v_z_from_metric
    );

    // Accumulate depth gradient from scale_metric
    v_mean_c.z += v_z_from_metric;

    // vjp: transform Gaussian covariance to camera space
    vec3 v_mean(0.f);
    mat3 v_covar(0.f);
    mat3 v_R(0.f);
    vec3 v_t(0.f);
    posW2C_VJP(R, t, glm::make_vec3(means), v_mean_c, v_R, v_t, v_mean);
    covarW2C_VJP(R, covar, v_covar_c, v_R, v_covar);

    // Add contribution from scale_metric
    v_R += v_R_from_metric;

    // write out results with warp-level reduction
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    auto warp_group_g = cg::labeled_partition(warp, gid);
    if (v_means != nullptr) {
        warpSum(v_mean, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_means += bid * N * 3 + gid * 3;
#pragma unroll
            for (uint32_t i = 0; i < 3; i++) {
                gpuAtomicAdd(v_means + i, v_mean[i]);
            }
        }
    }
    if (v_covars != nullptr) {
        // Output gradients w.r.t. the covariance matrix
        warpSum(v_covar, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_covars += bid * N * 6 + gid * 6;
            gpuAtomicAdd(v_covars, v_covar[0][0]);
            gpuAtomicAdd(v_covars + 1, v_covar[0][1] + v_covar[1][0]);
            gpuAtomicAdd(v_covars + 2, v_covar[0][2] + v_covar[2][0]);
            gpuAtomicAdd(v_covars + 3, v_covar[1][1]);
            gpuAtomicAdd(v_covars + 4, v_covar[1][2] + v_covar[2][1]);
            gpuAtomicAdd(v_covars + 5, v_covar[2][2]);
        }
    } else {
        // Output gradients w.r.t. the quaternion and scale
        vec4 v_quat(0.f);
        vec3 v_scale(0.f);
        quat_scale_to_covar_vjp(quat, scale, v_covar, v_quat, v_scale);

        // Add contribution from scale_metric
        v_scale += v_scales_from_metric;

        if (v_quats != nullptr) {
            warpSum(v_quat, warp_group_g);
            if (warp_group_g.thread_rank() == 0) {
                v_quats += bid * N * 4 + gid * 4;
#pragma unroll
                for (uint32_t i = 0; i < 4; i++) {
                    gpuAtomicAdd(v_quats + i, v_quat[i]);
                }
            }
        }
        if (v_scales != nullptr) {
            warpSum(v_scale, warp_group_g);
            if (warp_group_g.thread_rank() == 0) {
                v_scales += bid * N * 3 + gid * 3;
#pragma unroll
                for (uint32_t i = 0; i < 3; i++) {
                    gpuAtomicAdd(v_scales + i, v_scale[i]);
                }
            }
        }
    }

    if (viewmats_requires_grad && v_viewmats != nullptr) {
        // Accumulate gradients w.r.t. viewmat
        warpSum(v_R, warp_group_g);
        warpSum(v_t, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_viewmats += bid * C * 16 + cid * 16;
            // glm is column-major, so write columns
            gpuAtomicAdd(v_viewmats + 0, v_R[0][0]);
            gpuAtomicAdd(v_viewmats + 1, v_R[1][0]);
            gpuAtomicAdd(v_viewmats + 2, v_R[2][0]);
            gpuAtomicAdd(v_viewmats + 3, v_t[0]);
            gpuAtomicAdd(v_viewmats + 4, v_R[0][1]);
            gpuAtomicAdd(v_viewmats + 5, v_R[1][1]);
            gpuAtomicAdd(v_viewmats + 6, v_R[2][1]);
            gpuAtomicAdd(v_viewmats + 7, v_t[1]);
            gpuAtomicAdd(v_viewmats + 8, v_R[0][2]);
            gpuAtomicAdd(v_viewmats + 9, v_R[1][2]);
            gpuAtomicAdd(v_viewmats + 10, v_R[2][2]);
            gpuAtomicAdd(v_viewmats + 11, v_t[2]);
        }
    }
}

} // namespace gsplat
