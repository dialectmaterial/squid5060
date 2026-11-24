#include "kernels.hpp"
#include "kernels_common.cuh"

#include <cstdio>

namespace squid {

namespace {

// ============================================================================
// Fast F32 -> F16 conversion (vectorized float4 -> half2x2)
// ============================================================================
__global__ void f32_to_f16_kernel(const float* __restrict__ src, __half* __restrict__ dst, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        float4 v = *reinterpret_cast<const float4*>(src + i);
        __half2 h0 = __floats2half2_rn(v.x, v.y);
        __half2 h1 = __floats2half2_rn(v.z, v.w);
        *reinterpret_cast<__half2*>(dst + i) = h0;
        *reinterpret_cast<__half2*>(dst + i + 2) = h1;
    } else {
        for (int k = i; k < n && k < i + 4; ++k) dst[k] = __float2half(src[k]);
    }
}

// ============================================================================
// ZERO-SMEM L1-CACHED NVFP4 / FP8 / BF16 Linear Kernels (B = 1..8)
// Uses Blackwell sm_120a hardware PTX cvt.rn.f16x2.e2m1x2, 0 bytes SMEM, 0 spills,
// 48 warps/SM max occupancy, and reads small activations directly from L1 cache!
// ============================================================================

template <int B_CONST, bool ADD_RES, int TPR = 32>
__global__ void __launch_bounds__(128)
linear_nvfp4_h_kernel(
    const uint8_t* __restrict__ w_data,
    const uint8_t* __restrict__ w_scales,
    float global_scale,
    const __half* __restrict__ x_h, // [B, K]
    float* __restrict__ y,          // [B, N]
    int N, int K) {
    static_assert(TPR == 16 || TPR == 32);
    constexpr int ROWS_PER_BLK = 128 / TPR;
    int r_in_blk = (TPR == 32) ? (threadIdx.x >> 5) : (threadIdx.x >> 4);
    int lane = threadIdx.x & (TPR - 1);
    int row = blockIdx.x * ROWS_PER_BLK + r_in_blk;
    if (row >= N) return;

    const uint2* w_row = reinterpret_cast<const uint2*>(w_data + (size_t)row * (K >> 1));
    const uint8_t* s_row = w_scales + (size_t)row * (K >> 4);
    int n_blocks16 = K >> 4;

    float acc[B_CONST];
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) acc[b] = 0.0f;

    for (int blk = lane; blk < n_blocks16; blk += TPR) {
        uint2 w8 = __ldg(w_row + blk);
        float sc = cvt_e4m3_to_float(__ldg(s_row + blk));
        __half2 wh[8];
        cvt_e2m1x8_to_half2x4(w8.x, wh[0], wh[1], wh[2], wh[3]);
        cvt_e2m1x8_to_half2x4(w8.y, wh[4], wh[5], wh[6], wh[7]);

        int koff = blk << 4;
        #pragma unroll
        for (int b = 0; b < B_CONST; ++b) {
            const uint4* x4 = reinterpret_cast<const uint4*>(x_h + (size_t)b * K + koff);
            uint4 xv0 = __ldg(x4);
            uint4 xv1 = __ldg(x4 + 1);
            const __half2* xb0 = reinterpret_cast<const __half2*>(&xv0);
            const __half2* xb1 = reinterpret_cast<const __half2*>(&xv1);
            __half2 p0 = __hmul2(wh[0], xb0[0]);
            __half2 p1 = __hmul2(wh[1], xb0[1]);
            p0 = __hfma2(wh[2], xb0[2], p0);
            p1 = __hfma2(wh[3], xb0[3], p1);
            p0 = __hfma2(wh[4], xb1[0], p0);
            p1 = __hfma2(wh[5], xb1[1], p1);
            p0 = __hfma2(wh[6], xb1[2], p0);
            p1 = __hfma2(wh[7], xb1[3], p1);
            float2 pf = __half22float2(__hadd2(p0, p1));
            acc[b] = fmaf(pf.x + pf.y, sc, acc[b]);
        }
    }

    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) {
        float s = ((TPR == 32) ? warp_sum(acc[b]) : half_warp_sum(acc[b])) * global_scale;
        if (lane == 0) {
            if constexpr (ADD_RES) y[(size_t)b * N + row] += s;
            else                   y[(size_t)b * N + row] = s;
        }
    }
}

// FUSED Q+R+Gate+Sink (6552) + K (1024) + V (1024) = 8600 rows in ONE single launch!
// HALF-WARP (16 threads/row -> 144 / 16 = 9 EXACT iterations, 0% warp-tail waste, 8 rows/CTA = 1075 CTAs!)
template <int B_CONST>
__global__ void __launch_bounds__(128)
qkvrg_fused_nvfp4_kernel(
    const uint8_t* __restrict__ wq_d, const uint8_t* __restrict__ wq_s, float q_gs,
    const uint8_t* __restrict__ wk_d, const uint8_t* __restrict__ wk_s, float k_gs,
    const uint8_t* __restrict__ wv_d, const uint8_t* __restrict__ wv_s, float v_gs,
    const __half* __restrict__ xq_h,
    const __half* __restrict__ xk_h,
    const __half* __restrict__ xv_h,
    float* __restrict__ y_qrgs,
    float* __restrict__ y_k,
    float* __restrict__ y_v) {
    constexpr int N_Q = Q_DIM + R_DIM + Q_DIM + NUM_HEADS; // 6552
    constexpr int N_K = KV_DIM;                            // 1024
    constexpr int N_TOT = N_Q + N_K + N_K;                 // 8600

    int hw = threadIdx.x >> 4;   // 0..7
    int lane = threadIdx.x & 15; // 0..15
    int g_row = blockIdx.x * 8 + hw;
    if (g_row >= N_TOT) return;

    const uint8_t* w_data;
    const uint8_t* w_scales;
    float gscale;
    const __half* x_h;
    float* y_out;
    int row, N_out;

    if (g_row < N_Q) {
        w_data = wq_d; w_scales = wq_s; gscale = q_gs;
        x_h = xq_h; y_out = y_qrgs; row = g_row; N_out = N_Q;
    } else if (g_row < N_Q + N_K) {
        w_data = wk_d; w_scales = wk_s; gscale = k_gs;
        x_h = xk_h; y_out = y_k; row = g_row - N_Q; N_out = N_K;
    } else {
        w_data = wv_d; w_scales = wv_s; gscale = v_gs;
        x_h = xv_h; y_out = y_v; row = g_row - (N_Q + N_K); N_out = N_K;
    }

    constexpr int NBLK = HIDDEN >> 4; // 144 = 9 * 16 EXACT!
    const uint2* w_row = reinterpret_cast<const uint2*>(w_data + (size_t)row * (HIDDEN >> 1));
    const uint8_t* s_row = w_scales + (size_t)row * NBLK;

    float acc[B_CONST];
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) acc[b] = 0.0f;

    #pragma unroll
    for (int iter = 0; iter < 9; ++iter) {
        int blk = lane + iter * 16;
        uint2 w8 = __ldg(w_row + blk);
        float sc = cvt_e4m3_to_float(__ldg(s_row + blk));
        __half2 wh[8];
        cvt_e2m1x8_to_half2x4(w8.x, wh[0], wh[1], wh[2], wh[3]);
        cvt_e2m1x8_to_half2x4(w8.y, wh[4], wh[5], wh[6], wh[7]);

        int koff = blk << 4;
        #pragma unroll
        for (int b = 0; b < B_CONST; ++b) {
            const uint4* x4 = reinterpret_cast<const uint4*>(x_h + (size_t)b * HIDDEN + koff);
            uint4 xv0 = __ldg(x4), xv1 = __ldg(x4 + 1);
            const __half2* xb0 = reinterpret_cast<const __half2*>(&xv0);
            const __half2* xb1 = reinterpret_cast<const __half2*>(&xv1);
            __half2 p0 = __hmul2(wh[0], xb0[0]);
            __half2 p1 = __hmul2(wh[1], xb0[1]);
            p0 = __hfma2(wh[2], xb0[2], p0);
            p1 = __hfma2(wh[3], xb0[3], p1);
            p0 = __hfma2(wh[4], xb1[0], p0);
            p1 = __hfma2(wh[5], xb1[1], p1);
            p0 = __hfma2(wh[6], xb1[2], p0);
            p1 = __hfma2(wh[7], xb1[3], p1);
            float2 pf = __half22float2(__hadd2(p0, p1));
            acc[b] = fmaf(pf.x + pf.y, sc, acc[b]);
        }
    }

    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) {
        float s = half_warp_sum(acc[b]) * gscale;
        if (lane == 0) y_out[(size_t)b * N_out + row] = s;
    }
}

template <int B_CONST, bool ADD_RES>
__global__ void __launch_bounds__(128)
linear_fp8_h_kernel(
    const uint8_t* __restrict__ w_data,
    const float* __restrict__ row_scales,
    const __half* __restrict__ x_h,
    float* __restrict__ y,
    int N, int K) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 4 + warp;
    if (row >= N) return;

    const uint4* w_row = reinterpret_cast<const uint4*>(w_data + (size_t)row * K);
    int n_vec16 = K >> 4;
    float rscale = row_scales[row];

    float acc[B_CONST];
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) acc[b] = 0.0f;

    for (int v = lane; v < n_vec16; v += 32) {
        uint4 w16 = __ldg(w_row + v);
        __half2 wh[8];
        wh[0] = cvt_e4m3x2_to_half2(uint16_t(w16.x & 0xFFFFu));
        wh[1] = cvt_e4m3x2_to_half2(uint16_t(w16.x >> 16));
        wh[2] = cvt_e4m3x2_to_half2(uint16_t(w16.y & 0xFFFFu));
        wh[3] = cvt_e4m3x2_to_half2(uint16_t(w16.y >> 16));
        wh[4] = cvt_e4m3x2_to_half2(uint16_t(w16.z & 0xFFFFu));
        wh[5] = cvt_e4m3x2_to_half2(uint16_t(w16.z >> 16));
        wh[6] = cvt_e4m3x2_to_half2(uint16_t(w16.w & 0xFFFFu));
        wh[7] = cvt_e4m3x2_to_half2(uint16_t(w16.w >> 16));

        int koff = v << 4;
        #pragma unroll
        for (int b = 0; b < B_CONST; ++b) {
            const uint4* x4 = reinterpret_cast<const uint4*>(x_h + (size_t)b * K + koff);
            uint4 xv0 = __ldg(x4), xv1 = __ldg(x4 + 1);
            const __half2* xb0 = reinterpret_cast<const __half2*>(&xv0);
            const __half2* xb1 = reinterpret_cast<const __half2*>(&xv1);
            __half2 p0 = __hmul2(wh[0], xb0[0]);
            __half2 p1 = __hmul2(wh[1], xb0[1]);
            p0 = __hfma2(wh[2], xb0[2], p0);
            p1 = __hfma2(wh[3], xb0[3], p1);
            p0 = __hfma2(wh[4], xb1[0], p0);
            p1 = __hfma2(wh[5], xb1[1], p1);
            p0 = __hfma2(wh[6], xb1[2], p0);
            p1 = __hfma2(wh[7], xb1[3], p1);
            float2 pf = __half22float2(__hadd2(p0, p1));
            acc[b] += pf.x + pf.y;
        }
    }
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) {
        float s = warp_sum(acc[b]) * rscale;
        if (lane == 0) {
            if constexpr (ADD_RES) y[(size_t)b * N + row] += s;
            else                   y[(size_t)b * N + row] = s;
        }
    }
}

template <int B_CONST, bool ADD_RES>
__global__ void __launch_bounds__(128)
linear_bf16_nosmem_t_kernel(
    const __nv_bfloat16* __restrict__ w_data,
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int K) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 4 + warp;
    if (row >= N) return;

    const uint4* w_row = reinterpret_cast<const uint4*>(w_data + (size_t)row * K);
    int n_vec8 = K >> 3;

    float acc[B_CONST];
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) acc[b] = 0.0f;

    for (int v = lane; v < n_vec8; v += 32) {
        uint4 w8 = __ldg(w_row + v);
        const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&w8);
        float2 wf0 = __bfloat1622float2(wb[0]);
        float2 wf1 = __bfloat1622float2(wb[1]);
        float2 wf2 = __bfloat1622float2(wb[2]);
        float2 wf3 = __bfloat1622float2(wb[3]);
        int koff = v << 3;
        #pragma unroll
        for (int b = 0; b < B_CONST; ++b) {
            const float4* xf = reinterpret_cast<const float4*>(x + (size_t)b * K + koff);
            float4 x0 = __ldg(xf), x1 = __ldg(xf + 1);
            acc[b] = fmaf(wf0.x, x0.x, acc[b]); acc[b] = fmaf(wf0.y, x0.y, acc[b]);
            acc[b] = fmaf(wf1.x, x0.z, acc[b]); acc[b] = fmaf(wf1.y, x0.w, acc[b]);
            acc[b] = fmaf(wf2.x, x1.x, acc[b]); acc[b] = fmaf(wf2.y, x1.y, acc[b]);
            acc[b] = fmaf(wf3.x, x1.z, acc[b]); acc[b] = fmaf(wf3.y, x1.w, acc[b]);
        }
    }
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) {
        float s = warp_sum(acc[b]);
        if (lane == 0) {
            if constexpr (ADD_RES) y[(size_t)b * N + row] += s;
            else                   y[(size_t)b * N + row] = s;
        }
    }
}

template <int B_CONST, bool ADD_RES>
__global__ void __launch_bounds__(128)
linear_nvfp4_f32_kernel(
    const uint8_t* __restrict__ w_data,
    const uint8_t* __restrict__ w_scales,
    float global_scale,
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int K) {
    // Zero-SMEM: converts x on the fly from L1 (used for DSpark/general fallback)
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 4 + warp;
    if (row >= N) return;

    const uint2* w_row = reinterpret_cast<const uint2*>(w_data + (size_t)row * (K >> 1));
    const uint8_t* s_row = w_scales + (size_t)row * (K >> 4);
    int n_blocks16 = K >> 4;

    float acc[B_CONST];
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) acc[b] = 0.0f;

    for (int blk = lane; blk < n_blocks16; blk += 32) {
        uint2 w8 = __ldg(w_row + blk);
        float sc = cvt_e4m3_to_float(__ldg(s_row + blk));
        __half2 wh[8];
        cvt_e2m1x8_to_half2x4(w8.x, wh[0], wh[1], wh[2], wh[3]);
        cvt_e2m1x8_to_half2x4(w8.y, wh[4], wh[5], wh[6], wh[7]);
        int koff = blk << 4;
        #pragma unroll
        for (int b = 0; b < B_CONST; ++b) {
            const float4* xf = reinterpret_cast<const float4*>(x + (size_t)b * K + koff);
            float4 f0 = __ldg(xf), f1 = __ldg(xf + 1), f2 = __ldg(xf + 2), f3 = __ldg(xf + 3);
            __half2 p0 = __hmul2(wh[0], __floats2half2_rn(f0.x, f0.y));
            __half2 p1 = __hmul2(wh[1], __floats2half2_rn(f0.z, f0.w));
            p0 = __hfma2(wh[2], __floats2half2_rn(f1.x, f1.y), p0);
            p1 = __hfma2(wh[3], __floats2half2_rn(f1.z, f1.w), p1);
            p0 = __hfma2(wh[4], __floats2half2_rn(f2.x, f2.y), p0);
            p1 = __hfma2(wh[5], __floats2half2_rn(f2.z, f2.w), p1);
            p0 = __hfma2(wh[6], __floats2half2_rn(f3.x, f3.y), p0);
            p1 = __hfma2(wh[7], __floats2half2_rn(f3.z, f3.w), p1);
            float2 pf = __half22float2(__hadd2(p0, p1));
            acc[b] = fmaf(pf.x + pf.y, sc, acc[b]);
        }
    }
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) {
        float s = warp_sum(acc[b]) * global_scale;
        if (lane == 0) {
            if constexpr (ADD_RES) y[(size_t)b * N + row] += s;
            else                   y[(size_t)b * N + row] = s;
        }
    }
}

template <int B_CONST, bool ADD_RES>
__global__ void __launch_bounds__(128)
linear_fp8_f32_kernel(
    const uint8_t* __restrict__ w_data,
    const float* __restrict__ row_scales,
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int K) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * 4 + warp;
    if (row >= N) return;

    const uint4* w_row = reinterpret_cast<const uint4*>(w_data + (size_t)row * K);
    int n_vec16 = K >> 4;
    float rscale = row_scales[row];

    float acc[B_CONST];
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) acc[b] = 0.0f;

    for (int v = lane; v < n_vec16; v += 32) {
        uint4 w16 = __ldg(w_row + v);
        __half2 wh[8];
        wh[0] = cvt_e4m3x2_to_half2(uint16_t(w16.x & 0xFFFFu));
        wh[1] = cvt_e4m3x2_to_half2(uint16_t(w16.x >> 16));
        wh[2] = cvt_e4m3x2_to_half2(uint16_t(w16.y & 0xFFFFu));
        wh[3] = cvt_e4m3x2_to_half2(uint16_t(w16.y >> 16));
        wh[4] = cvt_e4m3x2_to_half2(uint16_t(w16.z & 0xFFFFu));
        wh[5] = cvt_e4m3x2_to_half2(uint16_t(w16.z >> 16));
        wh[6] = cvt_e4m3x2_to_half2(uint16_t(w16.w & 0xFFFFu));
        wh[7] = cvt_e4m3x2_to_half2(uint16_t(w16.w >> 16));
        int koff = v << 4;
        #pragma unroll
        for (int b = 0; b < B_CONST; ++b) {
            const float4* xf = reinterpret_cast<const float4*>(x + (size_t)b * K + koff);
            float4 f0 = __ldg(xf), f1 = __ldg(xf + 1), f2 = __ldg(xf + 2), f3 = __ldg(xf + 3);
            __half2 p0 = __hmul2(wh[0], __floats2half2_rn(f0.x, f0.y));
            __half2 p1 = __hmul2(wh[1], __floats2half2_rn(f0.z, f0.w));
            p0 = __hfma2(wh[2], __floats2half2_rn(f1.x, f1.y), p0);
            p1 = __hfma2(wh[3], __floats2half2_rn(f1.z, f1.w), p1);
            p0 = __hfma2(wh[4], __floats2half2_rn(f2.x, f2.y), p0);
            p1 = __hfma2(wh[5], __floats2half2_rn(f2.z, f2.w), p1);
            p0 = __hfma2(wh[6], __floats2half2_rn(f3.x, f3.y), p0);
            p1 = __hfma2(wh[7], __floats2half2_rn(f3.z, f3.w), p1);
            float2 pf = __half22float2(__hadd2(p0, p1));
            acc[b] += pf.x + pf.y;
        }
    }
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) {
        float s = warp_sum(acc[b]) * rscale;
        if (lane == 0) {
            if constexpr (ADD_RES) y[(size_t)b * N + row] += s;
            else                   y[(size_t)b * N + row] = s;
        }
    }
}

template <int B_CONST>
static void dispatch_linear_b(
    const LinearWeight& w, const float* x, float* y, bool add_res, cudaStream_t stream) {
    int grid = (w.rows + 3) / 4;
    if (w.kind == WeightKind::NVFP4) {
        if (add_res) linear_nvfp4_f32_kernel<B_CONST, true><<<grid, 128, 0, stream>>>(
            static_cast<const uint8_t*>(w.data), w.scales, w.global_scale, x, y, w.rows, w.cols);
        else linear_nvfp4_f32_kernel<B_CONST, false><<<grid, 128, 0, stream>>>(
            static_cast<const uint8_t*>(w.data), w.scales, w.global_scale, x, y, w.rows, w.cols);
    } else if (w.kind == WeightKind::FP8) {
        if (add_res) linear_fp8_f32_kernel<B_CONST, true><<<grid, 128, 0, stream>>>(
            static_cast<const uint8_t*>(w.data), w.row_scales, x, y, w.rows, w.cols);
        else linear_fp8_f32_kernel<B_CONST, false><<<grid, 128, 0, stream>>>(
            static_cast<const uint8_t*>(w.data), w.row_scales, x, y, w.rows, w.cols);
    } else {
        if (add_res) linear_bf16_nosmem_t_kernel<B_CONST, true><<<grid, 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(w.data), x, y, w.rows, w.cols);
        else linear_bf16_nosmem_t_kernel<B_CONST, false><<<grid, 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(w.data), x, y, w.rows, w.cols);
    }
}

template <int B_CONST>
static void dispatch_linear_h_b(
    const LinearWeight& w, const __half* x_h, float* y, bool add_res, cudaStream_t stream) {
    if (w.kind == WeightKind::NVFP4) {
        bool use_half_warp = (((w.cols >> 4) & 31) != 0);
        if (use_half_warp) {
            int grid = (w.rows + 7) / 8;
            if (add_res) linear_nvfp4_h_kernel<B_CONST, true, 16><<<grid, 128, 0, stream>>>(
                static_cast<const uint8_t*>(w.data), w.scales, w.global_scale, x_h, y, w.rows, w.cols);
            else linear_nvfp4_h_kernel<B_CONST, false, 16><<<grid, 128, 0, stream>>>(
                static_cast<const uint8_t*>(w.data), w.scales, w.global_scale, x_h, y, w.rows, w.cols);
        } else {
            int grid = (w.rows + 3) / 4;
            if (add_res) linear_nvfp4_h_kernel<B_CONST, true, 32><<<grid, 128, 0, stream>>>(
                static_cast<const uint8_t*>(w.data), w.scales, w.global_scale, x_h, y, w.rows, w.cols);
            else linear_nvfp4_h_kernel<B_CONST, false, 32><<<grid, 128, 0, stream>>>(
                static_cast<const uint8_t*>(w.data), w.scales, w.global_scale, x_h, y, w.rows, w.cols);
        }
    } else if (w.kind == WeightKind::FP8) {
        int grid = (w.rows + 3) / 4;
        if (add_res) linear_fp8_h_kernel<B_CONST, true><<<grid, 128, 0, stream>>>(
            static_cast<const uint8_t*>(w.data), w.row_scales, x_h, y, w.rows, w.cols);
        else linear_fp8_h_kernel<B_CONST, false><<<grid, 128, 0, stream>>>(
            static_cast<const uint8_t*>(w.data), w.row_scales, x_h, y, w.rows, w.cols);
    }
}

// ============================================================================
// Fused LM Head + Softcap(30) + GPU Argmax (HALF-WARP NVFP4 -> 0% tail waste!)
// ============================================================================

constexpr int LM_HEAD_BLOCKS = 208; // 8 * 26 SMs on RTX 5060 Laptop (0% wave tail!)

template <int B_CONST, int KIND>
__global__ void __launch_bounds__(256)
lm_head_stage1_kernel(
    const void* __restrict__ w_data,
    const uint8_t* __restrict__ w_scales,
    const float* __restrict__ row_scales,
    float global_scale,
    const float* __restrict__ x,
    float* __restrict__ out_logits,
    float* __restrict__ blk_max_val,
    int32_t* __restrict__ blk_max_idx,
    int V) {
    __shared__ uint16_t s_x16[B_CONST * HIDDEN];
    int tid = threadIdx.x;
    int total_x = B_CONST * HIDDEN;
    if constexpr (KIND == 0) {
        for (int i = tid; i < total_x; i += 256) {
            __nv_bfloat16 bv = __float2bfloat16(x[i]);
            s_x16[i] = *reinterpret_cast<uint16_t*>(&bv);
        }
    } else {
        for (int i = tid; i < total_x; i += 256) {
            __half hv = __float2half(x[i]);
            s_x16[i] = *reinterpret_cast<uint16_t*>(&hv);
        }
    }
    __syncthreads();
    const __half* s_xh = reinterpret_cast<const __half*>(s_x16);
    const __nv_bfloat16* s_xbf = reinterpret_cast<const __nv_bfloat16*>(s_x16);

    constexpr int TPR = (KIND == 2) ? 16 : 32;
    constexpr int RPB = 256 / TPR; // 16 for NVFP4, 8 for FP8/BF16
    int r_in_blk = (TPR == 16) ? (tid >> 4) : (tid >> 5);
    int lane = tid & (TPR - 1);
    int global_r = blockIdx.x * RPB + r_in_blk;
    int total_r = gridDim.x * RPB;

    float best_val[B_CONST];
    int32_t best_idx[B_CONST];
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) { best_val[b] = -1e30f; best_idx[b] = 0; }

    for (int row = global_r; row < V; row += total_r) {
        float acc[B_CONST];
        #pragma unroll
        for (int b = 0; b < B_CONST; ++b) acc[b] = 0.0f;

        if constexpr (KIND == 2) {
            const uint2* wr = reinterpret_cast<const uint2*>(
                static_cast<const uint8_t*>(w_data) + (size_t)row * (HIDDEN >> 1));
            const uint8_t* sr = w_scales + (size_t)row * (HIDDEN >> 4);
            #pragma unroll
            for (int iter = 0; iter < 9; ++iter) {
                int blk = lane + iter * 16;
                uint2 w8 = __ldg(wr + blk);
                float sc = cvt_e4m3_to_float(__ldg(sr + blk));
                __half2 wh[8];
                cvt_e2m1x8_to_half2x4(w8.x, wh[0], wh[1], wh[2], wh[3]);
                cvt_e2m1x8_to_half2x4(w8.y, wh[4], wh[5], wh[6], wh[7]);
                int koff = blk << 4;
                #pragma unroll
                for (int b = 0; b < B_CONST; ++b) {
                    const __half2* xb = reinterpret_cast<const __half2*>(s_xh + b * HIDDEN + koff);
                    __half2 p0 = __hmul2(wh[0], xb[0]), p1 = __hmul2(wh[1], xb[1]);
                    p0 = __hfma2(wh[2], xb[2], p0); p1 = __hfma2(wh[3], xb[3], p1);
                    p0 = __hfma2(wh[4], xb[4], p0); p1 = __hfma2(wh[5], xb[5], p1);
                    p0 = __hfma2(wh[6], xb[6], p0); p1 = __hfma2(wh[7], xb[7], p1);
                    float2 pf = __half22float2(__hadd2(p0, p1));
                    acc[b] = fmaf(pf.x + pf.y, sc, acc[b]);
                }
            }
        } else if constexpr (KIND == 1) {
            const uint4* wr = reinterpret_cast<const uint4*>(
                static_cast<const uint8_t*>(w_data) + (size_t)row * HIDDEN);
            constexpr int NVEC = HIDDEN >> 4;
            for (int v = lane; v < NVEC; v += 32) {
                uint4 w16 = __ldg(wr + v);
                __half2 wh[8];
                wh[0] = cvt_e4m3x2_to_half2(uint16_t(w16.x & 0xFFFFu));
                wh[1] = cvt_e4m3x2_to_half2(uint16_t(w16.x >> 16));
                wh[2] = cvt_e4m3x2_to_half2(uint16_t(w16.y & 0xFFFFu));
                wh[3] = cvt_e4m3x2_to_half2(uint16_t(w16.y >> 16));
                wh[4] = cvt_e4m3x2_to_half2(uint16_t(w16.z & 0xFFFFu));
                wh[5] = cvt_e4m3x2_to_half2(uint16_t(w16.z >> 16));
                wh[6] = cvt_e4m3x2_to_half2(uint16_t(w16.w & 0xFFFFu));
                wh[7] = cvt_e4m3x2_to_half2(uint16_t(w16.w >> 16));
                int koff = v << 4;
                #pragma unroll
                for (int b = 0; b < B_CONST; ++b) {
                    const __half2* xb = reinterpret_cast<const __half2*>(s_xh + b * HIDDEN + koff);
                    __half2 p0 = __hmul2(wh[0], xb[0]), p1 = __hmul2(wh[1], xb[1]);
                    p0 = __hfma2(wh[2], xb[2], p0); p1 = __hfma2(wh[3], xb[3], p1);
                    p0 = __hfma2(wh[4], xb[4], p0); p1 = __hfma2(wh[5], xb[5], p1);
                    p0 = __hfma2(wh[6], xb[6], p0); p1 = __hfma2(wh[7], xb[7], p1);
                    float2 pf = __half22float2(__hadd2(p0, p1));
                    acc[b] += pf.x + pf.y;
                }
            }
        } else {
            const uint4* wr = reinterpret_cast<const uint4*>(
                static_cast<const __nv_bfloat16*>(w_data) + (size_t)row * HIDDEN);
            constexpr int NVEC = HIDDEN >> 3;
            for (int v = lane; v < NVEC; v += 32) {
                uint4 w8 = __ldg(wr + v);
                const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&w8);
                float2 wf0 = __bfloat1622float2(wb[0]), wf1 = __bfloat1622float2(wb[1]);
                float2 wf2 = __bfloat1622float2(wb[2]), wf3 = __bfloat1622float2(wb[3]);
                int koff = v << 3;
                #pragma unroll
                for (int b = 0; b < B_CONST; ++b) {
                    uint4 x8 = *reinterpret_cast<const uint4*>(s_xbf + b * HIDDEN + koff);
                    const __nv_bfloat162* xbf = reinterpret_cast<const __nv_bfloat162*>(&x8);
                    float2 xf0 = __bfloat1622float2(xbf[0]), xf1 = __bfloat1622float2(xbf[1]);
                    float2 xf2 = __bfloat1622float2(xbf[2]), xf3 = __bfloat1622float2(xbf[3]);
                    acc[b] = fmaf(wf0.x, xf0.x, acc[b]); acc[b] = fmaf(wf0.y, xf0.y, acc[b]);
                    acc[b] = fmaf(wf1.x, xf1.x, acc[b]); acc[b] = fmaf(wf1.y, xf1.y, acc[b]);
                    acc[b] = fmaf(wf2.x, xf2.x, acc[b]); acc[b] = fmaf(wf2.y, xf2.y, acc[b]);
                    acc[b] = fmaf(wf3.x, xf3.x, acc[b]); acc[b] = fmaf(wf3.y, xf3.y, acc[b]);
                }
            }
        }

        float scale = (KIND == 2) ? global_scale : ((KIND == 1) ? row_scales[row] : 1.0f);
        #pragma unroll
        for (int b = 0; b < B_CONST; ++b) {
            float val = ((TPR == 16) ? half_warp_sum(acc[b]) : warp_sum(acc[b])) * scale;
            if (lane == 0) {
                if (out_logits) {
                    float cap_val = fast_tanh(val * (1.0f / FINAL_SOFTCAP)) * FINAL_SOFTCAP;
                    out_logits[(size_t)b * V + row] = cap_val;
                    if (cap_val > best_val[b]) { best_val[b] = cap_val; best_idx[b] = row; }
                } else {
                    if (val > best_val[b]) { best_val[b] = val; best_idx[b] = row; }
                }
            }
        }
    }

    __shared__ float sm_val[B_CONST][RPB];
    __shared__ int32_t sm_idx[B_CONST][RPB];
    if (lane == 0) {
        #pragma unroll
        for (int b = 0; b < B_CONST; ++b) { sm_val[b][r_in_blk] = best_val[b]; sm_idx[b][r_in_blk] = best_idx[b]; }
    }
    __syncthreads();
    if (tid < B_CONST) {
        int b = tid;
        float bv = sm_val[b][0];
        int32_t bi = sm_idx[b][0];
        for (int w = 1; w < RPB; ++w) {
            if (sm_val[b][w] > bv || (sm_val[b][w] == bv && sm_idx[b][w] < bi)) { bv = sm_val[b][w]; bi = sm_idx[b][w]; }
        }
        blk_max_val[b * LM_HEAD_BLOCKS + blockIdx.x] = bv;
        blk_max_idx[b * LM_HEAD_BLOCKS + blockIdx.x] = bi;
    }
}

__global__ void lm_head_stage2_argmax_kernel(
    const float* __restrict__ blk_max_val,
    const int32_t* __restrict__ blk_max_idx,
    int32_t* __restrict__ out_tokens,
    int B) {
    int b = blockIdx.x;
    if (b >= B) return;
    int tid = threadIdx.x;
    float bv = -1e30f;
    int32_t bi = 0;
    for (int i = tid; i < LM_HEAD_BLOCKS; i += 256) {
        float v = blk_max_val[b * LM_HEAD_BLOCKS + i];
        int32_t idx = blk_max_idx[b * LM_HEAD_BLOCKS + i];
        if (v > bv || (v == bv && idx < bi)) { bv = v; bi = idx; }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, bv, off);
        int32_t oi = __shfl_xor_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    __shared__ float sv[8];
    __shared__ int32_t si[8];
    if ((tid & 31) == 0) { sv[tid >> 5] = bv; si[tid >> 5] = bi; }
    __syncthreads();
    if (tid == 0) {
        bv = sv[0]; bi = si[0];
        for (int w = 1; w < 8; ++w) {
            if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) { bv = sv[w]; bi = si[w]; }
        }
        out_tokens[b] = bi;
    }
}

// ============================================================================
// ZERO-SMEM, ZERO-SPILL MoE Expert Kernels (B=1 AND B=2..8 Specialized Grouped!)
// ============================================================================

__global__ void build_expert_groups_kernel(
    const int32_t* __restrict__ route_ids,
    const float* __restrict__ route_w,
    ExpertGroups* __restrict__ g,
    int B) {
    if (threadIdx.x != 0) return;
    int n_uniq = 0;
    int exp_to_u[NUM_EXPERTS];
    for (int e = 0; e < NUM_EXPERTS; ++e) exp_to_u[e] = -1;
    for (int b = 0; b < B; ++b) {
        for (int s = 0; s < TOP_K; ++s) {
            int e = route_ids[b * TOP_K + s];
            float rw = route_w[b * TOP_K + s];
            int u = exp_to_u[e];
            if (u < 0) {
                u = n_uniq++;
                exp_to_u[e] = u;
                g->expert_id[u] = e;
                g->token_mask[u] = 0;
            }
            g->token_mask[u] |= uint8_t(1u << b);
            g->slot_idx[u][b] = uint8_t(s);
            g->route_w[u][b] = rw;
        }
    }
    g->num_unique = n_uniq;
}

// --- B = 1 MoE Gate+Up (NVFP4, ZERO SMEM, 32 threads/row -> max TLP & L1 hiding!) ---
__global__ void __launch_bounds__(128)
moe_gate_up_nvfp4_b1_h_kernel(
    const uint8_t* __restrict__ w_data,
    const uint8_t* __restrict__ w_scales,
    const float* __restrict__ exp_scales,
    const __half* __restrict__ x_h,
    const int32_t* __restrict__ r_ids,
    __half* __restrict__ out_act_h) {
    int slot = blockIdx.y;
    int expert = r_ids[slot];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int j = blockIdx.x * 4 + warp;
    if (j >= INTERMEDIATE) return;

    size_t row_gate = ((size_t)expert * INTERMEDIATE + j) * 2;
    size_t row_up = row_gate + 1;
    constexpr int W_STRIDE = HIDDEN >> 1;
    constexpr int S_STRIDE = HIDDEN >> 4;
    const uint2* wg = reinterpret_cast<const uint2*>(w_data + row_gate * W_STRIDE);
    const uint2* wu = reinterpret_cast<const uint2*>(w_data + row_up * W_STRIDE);
    const uint8_t* sg = w_scales + row_gate * S_STRIDE;
    const uint8_t* su = w_scales + row_up * S_STRIDE;

    float acc_g = 0.0f, acc_u = 0.0f;
    for (int blk = lane; blk < S_STRIDE; blk += 32) {
        const uint4* x4 = reinterpret_cast<const uint4*>(x_h + (blk << 4));
        uint4 xv0 = __ldg(x4), xv1 = __ldg(x4 + 1);
        __half2 xh[8];
        const __half2* p0 = reinterpret_cast<const __half2*>(&xv0);
        const __half2* p1 = reinterpret_cast<const __half2*>(&xv1);
        xh[0] = p0[0]; xh[1] = p0[1]; xh[2] = p0[2]; xh[3] = p0[3];
        xh[4] = p1[0]; xh[5] = p1[1]; xh[6] = p1[2]; xh[7] = p1[3];

        uint2 w8g = __ldg(wg + blk);
        uint2 w8u = __ldg(wu + blk);
        float scg = cvt_e4m3_to_float(__ldg(sg + blk));
        float scu = cvt_e4m3_to_float(__ldg(su + blk));
        acc_g = fmaf(nvfp4_block16_dot(w8g, xh), scg, acc_g);
        acc_u = fmaf(nvfp4_block16_dot(w8u, xh), scu, acc_u);
    }
    float g = warp_sum(acc_g);
    float u = warp_sum(acc_u);
    if (lane == 0) {
        float es = exp_scales[expert];
        g *= es; u *= es;
        float g_cap = fast_tanh(g * (1.0f / GATE_SOFTCAP)) * GATE_SOFTCAP;
        float u_cap = fast_tanh(u * (1.0f / UP_SOFTCAP)) * UP_SOFTCAP;
        out_act_h[slot * INTERMEDIATE + j] = __float2half((g_cap * fast_sigmoid(g)) * u_cap);
    }
}

__global__ void __launch_bounds__(128)
moe_gate_up_mxfp8_b1_h_kernel(
    const uint8_t* __restrict__ w_data,
    const uint8_t* __restrict__ w_scales,
    const __half* __restrict__ x_h,
    const int32_t* __restrict__ r_ids,
    __half* __restrict__ out_act_h) {
    int slot = blockIdx.y;
    int expert = r_ids[slot];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int j = blockIdx.x * 4 + warp;
    if (j >= INTERMEDIATE) return;

    size_t row_gate = ((size_t)expert * INTERMEDIATE + j) * 2;
    size_t row_up = row_gate + 1;
    constexpr int S_STRIDE = HIDDEN >> 5;
    const uint4* wg = reinterpret_cast<const uint4*>(w_data + row_gate * HIDDEN);
    const uint4* wu = reinterpret_cast<const uint4*>(w_data + row_up * HIDDEN);
    const uint8_t* sg = w_scales + row_gate * S_STRIDE;
    const uint8_t* su = w_scales + row_up * S_STRIDE;

    float acc_g = 0.0f, acc_u = 0.0f;
    for (int v = lane; v < (HIDDEN >> 4); v += 32) {
        const uint4* x4 = reinterpret_cast<const uint4*>(x_h + (v << 4));
        uint4 xv0 = __ldg(x4), xv1 = __ldg(x4 + 1);
        const __half2* xb0 = reinterpret_cast<const __half2*>(&xv0);
        const __half2* xb1 = reinterpret_cast<const __half2*>(&xv1);
        uint4 w16g = __ldg(wg + v);
        uint4 w16u = __ldg(wu + v);
        float scg = cvt_ue8m0_to_float(__ldg(sg + (v >> 1)));
        float scu = cvt_ue8m0_to_float(__ldg(su + (v >> 1)));

        auto dot16 = [&](uint4 w16) -> float {
            __half2 p0 = __hmul2(cvt_e4m3x2_to_half2(uint16_t(w16.x & 0xFFFFu)), xb0[0]);
            __half2 p1 = __hmul2(cvt_e4m3x2_to_half2(uint16_t(w16.x >> 16)), xb0[1]);
            p0 = __hfma2(cvt_e4m3x2_to_half2(uint16_t(w16.y & 0xFFFFu)), xb0[2], p0);
            p1 = __hfma2(cvt_e4m3x2_to_half2(uint16_t(w16.y >> 16)), xb0[3], p1);
            p0 = __hfma2(cvt_e4m3x2_to_half2(uint16_t(w16.z & 0xFFFFu)), xb1[0], p0);
            p1 = __hfma2(cvt_e4m3x2_to_half2(uint16_t(w16.z >> 16)), xb1[1], p1);
            p0 = __hfma2(cvt_e4m3x2_to_half2(uint16_t(w16.w & 0xFFFFu)), xb1[2], p0);
            p1 = __hfma2(cvt_e4m3x2_to_half2(uint16_t(w16.w >> 16)), xb1[3], p1);
            float2 pf = __half22float2(__hadd2(p0, p1));
            return pf.x + pf.y;
        };
        acc_g = fmaf(dot16(w16g), scg, acc_g);
        acc_u = fmaf(dot16(w16u), scu, acc_u);
    }
    float g = warp_sum(acc_g), u = warp_sum(acc_u);
    if (lane == 0) {
        float g_cap = fast_tanh(g * (1.0f / GATE_SOFTCAP)) * GATE_SOFTCAP;
        float u_cap = fast_tanh(u * (1.0f / UP_SOFTCAP)) * UP_SOFTCAP;
        out_act_h[slot * INTERMEDIATE + j] = __float2half((g_cap * fast_sigmoid(g)) * u_cap);
    }
}

// --- B = 1 MoE Down (NVFP4, QUARTER-WARP 8 threads/(j,slot) -> 9 EXACT iters, 0% tail waste!) ---
// 4 output rows `j` x 8 slots x 8 threads = 256 threads/CTA, Grid = 2304 / 4 = 576 CTAs!
__global__ void __launch_bounds__(256)
moe_down_nvfp4_b1_h_kernel(
    const uint8_t* __restrict__ w_data,
    const uint8_t* __restrict__ w_scales,
    const float* __restrict__ exp_scales,
    const __half* __restrict__ act_h,  // [TOP_K, INTERMEDIATE] (18 KB -> 100% L1 resident!)
    const int32_t* __restrict__ r_ids,
    const float* __restrict__ r_w,
    float* __restrict__ out_down) {
    int local_j = threadIdx.x >> 6;       // 0..3
    int slot = (threadIdx.x >> 3) & 7;    // 0..7
    int lane = threadIdx.x & 7;           // 0..7
    int j = blockIdx.x * 4 + local_j;
    if (j >= HIDDEN) return;

    int expert = r_ids[slot];
    size_t row = (size_t)expert * HIDDEN + j;
    constexpr int W_STRIDE = INTERMEDIATE >> 1;
    constexpr int S_STRIDE = INTERMEDIATE >> 4; // 72 = 9 * 8 EXACT!
    const uint2* wr = reinterpret_cast<const uint2*>(w_data + row * W_STRIDE);
    const uint8_t* sr = w_scales + row * S_STRIDE;
    const __half* ah = act_h + slot * INTERMEDIATE;

    float acc = 0.0f;
    #pragma unroll
    for (int iter = 0; iter < 9; ++iter) {
        int blk = lane + iter * 8;
        const uint4* x4 = reinterpret_cast<const uint4*>(ah + (blk << 4));
        uint4 xv0 = __ldg(x4), xv1 = __ldg(x4 + 1);
        __half2 xh[8];
        const __half2* p0 = reinterpret_cast<const __half2*>(&xv0);
        const __half2* p1 = reinterpret_cast<const __half2*>(&xv1);
        xh[0] = p0[0]; xh[1] = p0[1]; xh[2] = p0[2]; xh[3] = p0[3];
        xh[4] = p1[0]; xh[5] = p1[1]; xh[6] = p1[2]; xh[7] = p1[3];

        uint2 w8 = __ldg(wr + blk);
        float sc = cvt_e4m3_to_float(__ldg(sr + blk));
        acc = fmaf(nvfp4_block16_dot(w8, xh), sc, acc);
    }
    float dot = quarter_warp_sum(acc);
    __shared__ float partial[4][TOP_K];
    if (lane == 0) partial[local_j][slot] = dot * exp_scales[expert] * r_w[slot];
    __syncthreads();
    if (threadIdx.x < 4) {
        int lj = threadIdx.x;
        float sum = partial[lj][0] + partial[lj][1] + partial[lj][2] + partial[lj][3]
                  + partial[lj][4] + partial[lj][5] + partial[lj][6] + partial[lj][7];
        out_down[blockIdx.x * 4 + lj] = sum;
    }
}

__global__ void __launch_bounds__(256)
moe_down_mxfp8_b1_h_kernel(
    const uint8_t* __restrict__ w_data,
    const uint8_t* __restrict__ w_scales,
    const __half* __restrict__ act_h,
    const int32_t* __restrict__ r_ids,
    const float* __restrict__ r_w,
    float* __restrict__ out_down) {
    int j = blockIdx.x;
    int slot = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    int expert = r_ids[slot];
    size_t row = (size_t)expert * HIDDEN + j;
    constexpr int S_STRIDE = INTERMEDIATE >> 5;
    const uint4* wr = reinterpret_cast<const uint4*>(w_data + row * INTERMEDIATE);
    const uint8_t* sr = w_scales + row * S_STRIDE;
    const __half* ah = act_h + slot * INTERMEDIATE;

    float acc = 0.0f;
    for (int v = lane; v < (INTERMEDIATE >> 4); v += 32) {
        const uint4* x4 = reinterpret_cast<const uint4*>(ah + (v << 4));
        uint4 xv0 = __ldg(x4), xv1 = __ldg(x4 + 1);
        const __half2* xb0 = reinterpret_cast<const __half2*>(&xv0);
        const __half2* xb1 = reinterpret_cast<const __half2*>(&xv1);
        uint4 w16 = __ldg(wr + v);
        float sc = cvt_ue8m0_to_float(__ldg(sr + (v >> 1)));
        __half2 p0 = __hmul2(cvt_e4m3x2_to_half2(uint16_t(w16.x & 0xFFFFu)), xb0[0]);
        __half2 p1 = __hmul2(cvt_e4m3x2_to_half2(uint16_t(w16.x >> 16)), xb0[1]);
        p0 = __hfma2(cvt_e4m3x2_to_half2(uint16_t(w16.y & 0xFFFFu)), xb0[2], p0);
        p1 = __hfma2(cvt_e4m3x2_to_half2(uint16_t(w16.y >> 16)), xb0[3], p1);
        p0 = __hfma2(cvt_e4m3x2_to_half2(uint16_t(w16.z & 0xFFFFu)), xb1[0], p0);
        p1 = __hfma2(cvt_e4m3x2_to_half2(uint16_t(w16.z >> 16)), xb1[1], p1);
        p0 = __hfma2(cvt_e4m3x2_to_half2(uint16_t(w16.w & 0xFFFFu)), xb1[2], p0);
        p1 = __hfma2(cvt_e4m3x2_to_half2(uint16_t(w16.w >> 16)), xb1[3], p1);
        float2 pf = __half22float2(__hadd2(p0, p1));
        acc = fmaf(pf.x + pf.y, sc, acc);
    }
    float dot = warp_sum(acc);
    __shared__ float partial[TOP_K];
    if (lane == 0) partial[slot] = dot * r_w[slot];
    __syncthreads();
    if (threadIdx.x == 0) {
        out_down[j] = partial[0] + partial[1] + partial[2] + partial[3]
                    + partial[4] + partial[5] + partial[6] + partial[7];
    }
}

// --- B = 2..8 ZERO-SMEM, ZERO-SPILL HALF-WARP/QUARTER-WARP GROUPED MoE KERNELS!! ---

template <int B_CONST>
__global__ void __launch_bounds__(128)
moe_gate_up_nvfp4_grouped_h_kernel(
    const uint8_t* __restrict__ w_data,
    const uint8_t* __restrict__ w_scales,
    const float* __restrict__ exp_scales,
    const __half* __restrict__ x_h, // [B, HIDDEN]
    const ExpertGroups* __restrict__ g,
    __half* __restrict__ out_act_h) {
    int u = blockIdx.y;
    if (u >= g->num_unique) return;
    uint32_t mask = g->token_mask[u];
    int expert = g->expert_id[u];

    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int j = blockIdx.x * 4 + warp;
    if (j >= INTERMEDIATE) return;

    size_t row_gate = ((size_t)expert * INTERMEDIATE + j) * 2;
    size_t row_up = row_gate + 1;
    constexpr int W_STRIDE = HIDDEN >> 1;
    constexpr int S_STRIDE = HIDDEN >> 4;
    const uint2* wg = reinterpret_cast<const uint2*>(w_data + row_gate * W_STRIDE);
    const uint2* wu = reinterpret_cast<const uint2*>(w_data + row_up * W_STRIDE);
    const uint8_t* sg = w_scales + row_gate * S_STRIDE;
    const uint8_t* su = w_scales + row_up * S_STRIDE;

    if (__popc(mask) == 1) {
        int b = __ffs(mask) - 1;
        const __half* xb = x_h + (size_t)b * HIDDEN;
        float ag = 0.0f, au = 0.0f;
        for (int blk = lane; blk < S_STRIDE; blk += 32) {
            const uint4* x4 = reinterpret_cast<const uint4*>(xb + (blk << 4));
            uint4 xv0 = __ldg(x4), xv1 = __ldg(x4 + 1);
            __half2 xh[8];
            const __half2* p0 = reinterpret_cast<const __half2*>(&xv0);
            const __half2* p1 = reinterpret_cast<const __half2*>(&xv1);
            xh[0] = p0[0]; xh[1] = p0[1]; xh[2] = p0[2]; xh[3] = p0[3];
            xh[4] = p1[0]; xh[5] = p1[1]; xh[6] = p1[2]; xh[7] = p1[3];
            uint2 w8g = __ldg(wg + blk), w8u = __ldg(wu + blk);
            float scg = cvt_e4m3_to_float(__ldg(sg + blk));
            float scu = cvt_e4m3_to_float(__ldg(su + blk));
            ag = fmaf(nvfp4_block16_dot(w8g, xh), scg, ag);
            au = fmaf(nvfp4_block16_dot(w8u, xh), scu, au);
        }
        float es = exp_scales[expert];
        float gv = warp_sum(ag) * es;
        float uv = warp_sum(au) * es;
        if (lane == 0) {
            int slot = g->slot_idx[u][b];
            float g_cap = fast_tanh(gv * (1.0f / GATE_SOFTCAP)) * GATE_SOFTCAP;
            float u_cap = fast_tanh(uv * (1.0f / UP_SOFTCAP)) * UP_SOFTCAP;
            out_act_h[((size_t)b * TOP_K + slot) * INTERMEDIATE + j] = __float2half((g_cap * fast_sigmoid(gv)) * u_cap);
        }
        return;
    }

    float acc_g[B_CONST], acc_u[B_CONST];
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) { acc_g[b] = 0.0f; acc_u[b] = 0.0f; }

    for (int blk = lane; blk < S_STRIDE; blk += 32) {
        uint2 w8g = __ldg(wg + blk), w8u = __ldg(wu + blk);
        float scg = cvt_e4m3_to_float(__ldg(sg + blk));
        float scu = cvt_e4m3_to_float(__ldg(su + blk));
        __half2 gh[8], uh[8];
        cvt_e2m1x8_to_half2x4(w8g.x, gh[0], gh[1], gh[2], gh[3]);
        cvt_e2m1x8_to_half2x4(w8g.y, gh[4], gh[5], gh[6], gh[7]);
        cvt_e2m1x8_to_half2x4(w8u.x, uh[0], uh[1], uh[2], uh[3]);
        cvt_e2m1x8_to_half2x4(w8u.y, uh[4], uh[5], uh[6], uh[7]);

        int koff = blk << 4;
        #pragma unroll
        for (int b = 0; b < B_CONST; ++b) {
            if (mask & (1u << b)) {
                const uint4* x4 = reinterpret_cast<const uint4*>(x_h + (size_t)b * HIDDEN + koff);
                uint4 xv0 = __ldg(x4), xv1 = __ldg(x4 + 1);
                const __half2* xb0 = reinterpret_cast<const __half2*>(&xv0);
                const __half2* xb1 = reinterpret_cast<const __half2*>(&xv1);
                __half2 pg0 = __hmul2(gh[0], xb0[0]), pg1 = __hmul2(gh[1], xb0[1]);
                __half2 pu0 = __hmul2(uh[0], xb0[0]), pu1 = __hmul2(uh[1], xb0[1]);
                pg0 = __hfma2(gh[2], xb0[2], pg0); pg1 = __hfma2(gh[3], xb0[3], pg1);
                pu0 = __hfma2(uh[2], xb0[2], pu0); pu1 = __hfma2(uh[3], xb0[3], pu1);
                pg0 = __hfma2(gh[4], xb1[0], pg0); pg1 = __hfma2(gh[5], xb1[1], pg1);
                pu0 = __hfma2(uh[4], xb1[0], pu0); pu1 = __hfma2(uh[5], xb1[1], pu1);
                pg0 = __hfma2(gh[6], xb1[2], pg0); pg1 = __hfma2(gh[7], xb1[3], pg1);
                pu0 = __hfma2(uh[6], xb1[2], pu0); pu1 = __hfma2(uh[7], xb1[3], pu1);
                float2 fg = __half22float2(__hadd2(pg0, pg1));
                float2 fu = __half22float2(__hadd2(pu0, pu1));
                acc_g[b] = fmaf(fg.x + fg.y, scg, acc_g[b]);
                acc_u[b] = fmaf(fu.x + fu.y, scu, acc_u[b]);
            }
        }
    }

    float es = exp_scales[expert];
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) {
        if (mask & (1u << b)) {
            float gv = warp_sum(acc_g[b]) * es;
            float uv = warp_sum(acc_u[b]) * es;
            if (lane == 0) {
                int slot = g->slot_idx[u][b];
                float g_cap = fast_tanh(gv * (1.0f / GATE_SOFTCAP)) * GATE_SOFTCAP;
                float u_cap = fast_tanh(uv * (1.0f / UP_SOFTCAP)) * UP_SOFTCAP;
                out_act_h[((size_t)b * TOP_K + slot) * INTERMEDIATE + j] = __float2half((g_cap * fast_sigmoid(gv)) * u_cap);
            }
        }
    }
}

template <int B_CONST>
__global__ void __launch_bounds__(128)
moe_gate_up_mxfp8_grouped_h_kernel(
    const uint8_t* __restrict__ w_data,
    const uint8_t* __restrict__ w_scales,
    const __half* __restrict__ x_h,
    const ExpertGroups* __restrict__ g,
    __half* __restrict__ out_act_h) {
    int u = blockIdx.y;
    if (u >= g->num_unique) return;
    uint32_t mask = g->token_mask[u];
    int expert = g->expert_id[u];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int j = blockIdx.x * 4 + warp;
    if (j >= INTERMEDIATE) return;

    size_t row_gate = ((size_t)expert * INTERMEDIATE + j) * 2;
    size_t row_up = row_gate + 1;
    constexpr int S_STRIDE = HIDDEN >> 5;
    const uint4* wg = reinterpret_cast<const uint4*>(w_data + row_gate * HIDDEN);
    const uint4* wu = reinterpret_cast<const uint4*>(w_data + row_up * HIDDEN);
    const uint8_t* sg = w_scales + row_gate * S_STRIDE;
    const uint8_t* su = w_scales + row_up * S_STRIDE;

    float acc_g[B_CONST], acc_u[B_CONST];
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) { acc_g[b] = 0.0f; acc_u[b] = 0.0f; }

    for (int v = lane; v < (HIDDEN >> 4); v += 32) {
        uint4 w16g = __ldg(wg + v), w16u = __ldg(wu + v);
        float scg = cvt_ue8m0_to_float(__ldg(sg + (v >> 1)));
        float scu = cvt_ue8m0_to_float(__ldg(su + (v >> 1)));
        __half2 gh[8], uh[8];
        gh[0] = cvt_e4m3x2_to_half2(uint16_t(w16g.x & 0xFFFFu)); uh[0] = cvt_e4m3x2_to_half2(uint16_t(w16u.x & 0xFFFFu));
        gh[1] = cvt_e4m3x2_to_half2(uint16_t(w16g.x >> 16));      uh[1] = cvt_e4m3x2_to_half2(uint16_t(w16u.x >> 16));
        gh[2] = cvt_e4m3x2_to_half2(uint16_t(w16g.y & 0xFFFFu)); uh[2] = cvt_e4m3x2_to_half2(uint16_t(w16u.y & 0xFFFFu));
        gh[3] = cvt_e4m3x2_to_half2(uint16_t(w16g.y >> 16));      uh[3] = cvt_e4m3x2_to_half2(uint16_t(w16u.y >> 16));
        gh[4] = cvt_e4m3x2_to_half2(uint16_t(w16g.z & 0xFFFFu)); uh[4] = cvt_e4m3x2_to_half2(uint16_t(w16u.z & 0xFFFFu));
        gh[5] = cvt_e4m3x2_to_half2(uint16_t(w16g.z >> 16));      uh[5] = cvt_e4m3x2_to_half2(uint16_t(w16u.z >> 16));
        gh[6] = cvt_e4m3x2_to_half2(uint16_t(w16g.w & 0xFFFFu)); uh[6] = cvt_e4m3x2_to_half2(uint16_t(w16u.w & 0xFFFFu));
        gh[7] = cvt_e4m3x2_to_half2(uint16_t(w16g.w >> 16));      uh[7] = cvt_e4m3x2_to_half2(uint16_t(w16u.w >> 16));
        int koff = v << 4;
        #pragma unroll
        for (int b = 0; b < B_CONST; ++b) {
            if (mask & (1u << b)) {
                const uint4* x4 = reinterpret_cast<const uint4*>(x_h + (size_t)b * HIDDEN + koff);
                uint4 xv0 = __ldg(x4), xv1 = __ldg(x4 + 1);
                const __half2* xb0 = reinterpret_cast<const __half2*>(&xv0);
                const __half2* xb1 = reinterpret_cast<const __half2*>(&xv1);
                __half2 pg0 = __hmul2(gh[0], xb0[0]), pg1 = __hmul2(gh[1], xb0[1]);
                __half2 pu0 = __hmul2(uh[0], xb0[0]), pu1 = __hmul2(uh[1], xb0[1]);
                pg0 = __hfma2(gh[2], xb0[2], pg0); pg1 = __hfma2(gh[3], xb0[3], pg1);
                pu0 = __hfma2(uh[2], xb0[2], pu0); pu1 = __hfma2(uh[3], xb0[3], pu1);
                pg0 = __hfma2(gh[4], xb1[0], pg0); pg1 = __hfma2(gh[5], xb1[1], pg1);
                pu0 = __hfma2(uh[4], xb1[0], pu0); pu1 = __hfma2(uh[5], xb1[1], pu1);
                pg0 = __hfma2(gh[6], xb1[2], pg0); pg1 = __hfma2(gh[7], xb1[3], pg1);
                pu0 = __hfma2(uh[6], xb1[2], pu0); pu1 = __hfma2(uh[7], xb1[3], pu1);
                float2 fg = __half22float2(__hadd2(pg0, pg1));
                float2 fu = __half22float2(__hadd2(pu0, pu1));
                acc_g[b] = fmaf(fg.x + fg.y, scg, acc_g[b]);
                acc_u[b] = fmaf(fu.x + fu.y, scu, acc_u[b]);
            }
        }
    }
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) {
        if (mask & (1u << b)) {
            float gv = warp_sum(acc_g[b]), uv = warp_sum(acc_u[b]);
            if (lane == 0) {
                int slot = g->slot_idx[u][b];
                float g_cap = fast_tanh(gv * (1.0f / GATE_SOFTCAP)) * GATE_SOFTCAP;
                float u_cap = fast_tanh(uv * (1.0f / UP_SOFTCAP)) * UP_SOFTCAP;
                out_act_h[((size_t)b * TOP_K + slot) * INTERMEDIATE + j] = __float2half((g_cap * fast_sigmoid(gv)) * u_cap);
            }
        }
    }
}

template <int B_CONST>
__global__ void __launch_bounds__(128)
moe_down_nvfp4_grouped_h_kernel(
    const uint8_t* __restrict__ w_data,
    const uint8_t* __restrict__ w_scales,
    const float* __restrict__ exp_scales,
    const __half* __restrict__ act_h,
    const ExpertGroups* __restrict__ g,
    float* __restrict__ out_down) {
    int u = blockIdx.y;
    if (u >= g->num_unique) return;
    uint32_t mask = g->token_mask[u];
    int expert = g->expert_id[u];

    int qw = threadIdx.x >> 3;   // 0..15 (16 rows/CTA!)
    int lane = threadIdx.x & 7;  // 0..7 (72 / 8 = 9 EXACT iters!)
    int j = blockIdx.x * 16 + qw;
    if (j >= HIDDEN) return;

    size_t row = (size_t)expert * HIDDEN + j;
    constexpr int W_STRIDE = INTERMEDIATE >> 1;
    constexpr int S_STRIDE = INTERMEDIATE >> 4;
    const uint2* wr = reinterpret_cast<const uint2*>(w_data + row * W_STRIDE);
    const uint8_t* sr = w_scales + row * S_STRIDE;

    // Fast path for single-token expert!
    if (__popc(mask) == 1) {
        int b = __ffs(mask) - 1;
        int slot = g->slot_idx[u][b];
        const __half* ah = act_h + ((size_t)b * TOP_K + slot) * INTERMEDIATE;
        float acc = 0.0f;
        #pragma unroll
        for (int iter = 0; iter < 9; ++iter) {
            int blk = lane + iter * 8;
            const uint4* x4 = reinterpret_cast<const uint4*>(ah + (blk << 4));
            uint4 xv0 = __ldg(x4), xv1 = __ldg(x4 + 1);
            __half2 xh[8];
            const __half2* p0 = reinterpret_cast<const __half2*>(&xv0);
            const __half2* p1 = reinterpret_cast<const __half2*>(&xv1);
            xh[0] = p0[0]; xh[1] = p0[1]; xh[2] = p0[2]; xh[3] = p0[3];
            xh[4] = p1[0]; xh[5] = p1[1]; xh[6] = p1[2]; xh[7] = p1[3];
            uint2 w8 = __ldg(wr + blk);
            float sc = cvt_e4m3_to_float(__ldg(sr + blk));
            acc = fmaf(nvfp4_block16_dot(w8, xh), sc, acc);
        }
        float val = quarter_warp_sum(acc) * exp_scales[expert];
        if (lane == 0) out_down[((size_t)b * TOP_K + slot) * HIDDEN + j] = val * g->route_w[u][b];
        return;
    }

    float acc[B_CONST];
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) acc[b] = 0.0f;

    #pragma unroll
    for (int iter = 0; iter < 9; ++iter) {
        int blk = lane + iter * 8;
        uint2 w8 = __ldg(wr + blk);
        float sc = cvt_e4m3_to_float(__ldg(sr + blk));
        __half2 wh[8];
        cvt_e2m1x8_to_half2x4(w8.x, wh[0], wh[1], wh[2], wh[3]);
        cvt_e2m1x8_to_half2x4(w8.y, wh[4], wh[5], wh[6], wh[7]);
        int koff = blk << 4;
        #pragma unroll
        for (int b = 0; b < B_CONST; ++b) {
            if (mask & (1u << b)) {
                int slot = g->slot_idx[u][b];
                const uint4* x4 = reinterpret_cast<const uint4*>(act_h + ((size_t)b * TOP_K + slot) * INTERMEDIATE + koff);
                uint4 xv0 = __ldg(x4), xv1 = __ldg(x4 + 1);
                const __half2* xb0 = reinterpret_cast<const __half2*>(&xv0);
                const __half2* xb1 = reinterpret_cast<const __half2*>(&xv1);
                __half2 p0 = __hmul2(wh[0], xb0[0]), p1 = __hmul2(wh[1], xb0[1]);
                p0 = __hfma2(wh[2], xb0[2], p0); p1 = __hfma2(wh[3], xb0[3], p1);
                p0 = __hfma2(wh[4], xb1[0], p0); p1 = __hfma2(wh[5], xb1[1], p1);
                p0 = __hfma2(wh[6], xb1[2], p0); p1 = __hfma2(wh[7], xb1[3], p1);
                float2 pf = __half22float2(__hadd2(p0, p1));
                acc[b] = fmaf(pf.x + pf.y, sc, acc[b]);
            }
        }
    }
    float es = exp_scales[expert];
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) {
        if (mask & (1u << b)) {
            float val = quarter_warp_sum(acc[b]) * es;
            if (lane == 0) {
                int slot = g->slot_idx[u][b];
                out_down[((size_t)b * TOP_K + slot) * HIDDEN + j] = val * g->route_w[u][b];
            }
        }
    }
}

template <int B_CONST>
__global__ void __launch_bounds__(128)
moe_down_mxfp8_grouped_h_kernel(
    const uint8_t* __restrict__ w_data,
    const uint8_t* __restrict__ w_scales,
    const __half* __restrict__ act_h,
    const ExpertGroups* __restrict__ g,
    float* __restrict__ out_down) {
    int u = blockIdx.y;
    if (u >= g->num_unique) return;
    uint32_t mask = g->token_mask[u];
    int expert = g->expert_id[u];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int j = blockIdx.x * 4 + warp;
    if (j >= HIDDEN) return;

    size_t row = (size_t)expert * HIDDEN + j;
    constexpr int S_STRIDE = INTERMEDIATE >> 5;
    const uint4* wr = reinterpret_cast<const uint4*>(w_data + row * INTERMEDIATE);
    const uint8_t* sr = w_scales + row * S_STRIDE;

    float acc[B_CONST];
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) acc[b] = 0.0f;

    for (int v = lane; v < (INTERMEDIATE >> 4); v += 32) {
        uint4 w16 = __ldg(wr + v);
        float sc = cvt_ue8m0_to_float(__ldg(sr + (v >> 1)));
        __half2 wh[8];
        wh[0] = cvt_e4m3x2_to_half2(uint16_t(w16.x & 0xFFFFu));
        wh[1] = cvt_e4m3x2_to_half2(uint16_t(w16.x >> 16));
        wh[2] = cvt_e4m3x2_to_half2(uint16_t(w16.y & 0xFFFFu));
        wh[3] = cvt_e4m3x2_to_half2(uint16_t(w16.y >> 16));
        wh[4] = cvt_e4m3x2_to_half2(uint16_t(w16.z & 0xFFFFu));
        wh[5] = cvt_e4m3x2_to_half2(uint16_t(w16.z >> 16));
        wh[6] = cvt_e4m3x2_to_half2(uint16_t(w16.w & 0xFFFFu));
        wh[7] = cvt_e4m3x2_to_half2(uint16_t(w16.w >> 16));
        int koff = v << 4;
        #pragma unroll
        for (int b = 0; b < B_CONST; ++b) {
            if (mask & (1u << b)) {
                int slot = g->slot_idx[u][b];
                const uint4* x4 = reinterpret_cast<const uint4*>(act_h + ((size_t)b * TOP_K + slot) * INTERMEDIATE + koff);
                uint4 xv0 = __ldg(x4), xv1 = __ldg(x4 + 1);
                const __half2* xb0 = reinterpret_cast<const __half2*>(&xv0);
                const __half2* xb1 = reinterpret_cast<const __half2*>(&xv1);
                __half2 p0 = __hmul2(wh[0], xb0[0]), p1 = __hmul2(wh[1], xb0[1]);
                p0 = __hfma2(wh[2], xb0[2], p0); p1 = __hfma2(wh[3], xb0[3], p1);
                p0 = __hfma2(wh[4], xb1[0], p0); p1 = __hfma2(wh[5], xb1[1], p1);
                p0 = __hfma2(wh[6], xb1[2], p0); p1 = __hfma2(wh[7], xb1[3], p1);
                float2 pf = __half22float2(__hadd2(p0, p1));
                acc[b] = fmaf(pf.x + pf.y, sc, acc[b]);
            }
        }
    }
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) {
        if (mask & (1u << b)) {
            float val = warp_sum(acc[b]);
            if (lane == 0) {
                int slot = g->slot_idx[u][b];
                out_down[((size_t)b * TOP_K + slot) * HIDDEN + j] = val * g->route_w[u][b];
            }
        }
    }
}

// --- 8 GB Unified VRAM Expert Cache + Zero-Copy Mapped Host Fallback MoE Kernels ---
// Dual-path CTA-uniform dispatch (cslot >= 0 VRAM hit vs cslot < 0 PCIe Gen5 zero-copy miss):
//   - VRAM hit:  zero-SMEM uint2 path (>1,200 GB/s L2+GDDR7 on w13, >740 GB/s on w2)
//   - PCIe miss: 1-burst coalesced SMEM scale preload + uint4 (128B/256B) weight loads
//                hitting 23.7-23.9 GB/s (96% of 24.8 GB/s PCIe Gen5 x8 hardware DMA ceiling!)

__device__ __forceinline__ float nvfp4_block32_dot_u4_h(uint4 w16, uint16_t sc2, const __half* x_ptr) {
    const uint4* x4 = reinterpret_cast<const uint4*>(x_ptr);
    uint4 xv0 = __ldg(x4), xv1 = __ldg(x4 + 1), xv2 = __ldg(x4 + 2), xv3 = __ldg(x4 + 3);
    __half2 xh0[8], xh1[8];
    const __half2* p0 = reinterpret_cast<const __half2*>(&xv0);
    const __half2* p1 = reinterpret_cast<const __half2*>(&xv1);
    const __half2* p2 = reinterpret_cast<const __half2*>(&xv2);
    const __half2* p3 = reinterpret_cast<const __half2*>(&xv3);
    xh0[0] = p0[0]; xh0[1] = p0[1]; xh0[2] = p0[2]; xh0[3] = p0[3];
    xh0[4] = p1[0]; xh0[5] = p1[1]; xh0[6] = p1[2]; xh0[7] = p1[3];
    xh1[0] = p2[0]; xh1[1] = p2[1]; xh1[2] = p2[2]; xh1[3] = p2[3];
    xh1[4] = p3[0]; xh1[5] = p3[1]; xh1[6] = p3[2]; xh1[7] = p3[3];
    float s0 = cvt_e4m3_to_float(uint8_t(sc2 & 0xFFu));
    float s1 = cvt_e4m3_to_float(uint8_t(sc2 >> 8));
    return fmaf(nvfp4_block16_dot(make_uint2(w16.z, w16.w), xh1), s1,
                nvfp4_block16_dot(make_uint2(w16.x, w16.y), xh0) * s0);
}

// HALF-WARP (16 threads/row, 8 rows/CTA, grid=(144, TOP_K)) Dual-Path Cached MoE Gate+Up
__global__ void __launch_bounds__(128)
moe_gate_up_nvfp4_b1_cached_kernel(
    const uint8_t* __restrict__ cache_pool,
    const uint8_t* __restrict__ host_pool,
    const int32_t* __restrict__ slot_map,
    uint32_t* __restrict__ touch_counts,
    const float* __restrict__ exp_scales,
    const __half* __restrict__ x_h,
    const int32_t* __restrict__ r_ids,
    __half* __restrict__ out_act_h) {
    int slot = blockIdx.y;
    int expert = r_ids[slot];
    int tid = threadIdx.x;
    if (blockIdx.x == 0 && tid == 0 && touch_counts) {
        atomicAdd(&touch_counts[expert], 1u);
    }

    int hw = tid >> 4;   // 0..7 (8 j's/CTA!)
    int lane = tid & 15; // 0..15
    int j_base = blockIdx.x * 8;
    int j = j_base + hw;
    if (j >= INTERMEDIATE) return;

    int cslot = __ldg(slot_map + expert);
    float acc_g = 0.0f, acc_u = 0.0f;

    if (cslot >= 0) {
        // VRAM hit fast path: 9 exact uint2 iters, zero SMEM/syncthreads!
        const uint8_t* exp_base = cache_pool + (size_t)cslot * EXPERT_SLOT_BYTES;
        const uint8_t* w_data   = exp_base + EXPERT_W13_DATA_OFF;
        const uint8_t* w_scales = exp_base + EXPERT_W13_SCALE_OFF;
        size_t row_gate = (size_t)j * 2;
        size_t row_up = row_gate + 1;
        constexpr int W_STRIDE = HIDDEN >> 1;
        constexpr int S_STRIDE = HIDDEN >> 4;
        const uint2* wg = reinterpret_cast<const uint2*>(w_data + row_gate * W_STRIDE);
        const uint2* wu = reinterpret_cast<const uint2*>(w_data + row_up * W_STRIDE);
        const uint8_t* sg = w_scales + row_gate * S_STRIDE;
        const uint8_t* su = w_scales + row_up * S_STRIDE;

        #pragma unroll
        for (int iter = 0; iter < 9; ++iter) {
            int blk = lane + iter * 16;
            const uint4* x4 = reinterpret_cast<const uint4*>(x_h + (blk << 4));
            uint4 xv0 = __ldg(x4), xv1 = __ldg(x4 + 1);
            __half2 xh[8];
            const __half2* p0 = reinterpret_cast<const __half2*>(&xv0);
            const __half2* p1 = reinterpret_cast<const __half2*>(&xv1);
            xh[0] = p0[0]; xh[1] = p0[1]; xh[2] = p0[2]; xh[3] = p0[3];
            xh[4] = p1[0]; xh[5] = p1[1]; xh[6] = p1[2]; xh[7] = p1[3];

            uint2 w8g = __ldg(wg + blk);
            uint2 w8u = __ldg(wu + blk);
            float scg = cvt_e4m3_to_float(__ldg(sg + blk));
            float scu = cvt_e4m3_to_float(__ldg(su + blk));
            acc_g = fmaf(nvfp4_block16_dot(w8g, xh), scg, acc_g);
            acc_u = fmaf(nvfp4_block16_dot(w8u, xh), scu, acc_u);
        }
    } else {
        // PCIe Gen5 zero-copy miss path: 1 coalesced 2304B scale burst + 256B uint4 weight bursts (~23.8 GB/s)!
        const uint8_t* exp_base = host_pool + (size_t)expert * EXPERT_SLOT_BYTES;
        __shared__ __align__(16) uint8_t s_sc[16 * (HIDDEN >> 4)]; // 2,304 B = 144 x uint4
        const uint4* sc_g = reinterpret_cast<const uint4*>(exp_base + EXPERT_W13_SCALE_OFF + (size_t)(2 * j_base) * (HIDDEN >> 4));
        uint4* sc_s = reinterpret_cast<uint4*>(s_sc);
        sc_s[tid] = __ldg(sc_g + tid);
        if (tid < 16) sc_s[128 + tid] = __ldg(sc_g + 128 + tid);
        __syncthreads();

        const uint4* wg = reinterpret_cast<const uint4*>(exp_base + EXPERT_W13_DATA_OFF + (size_t)(2 * j) * (HIDDEN >> 1));
        const uint4* wu = reinterpret_cast<const uint4*>(exp_base + EXPERT_W13_DATA_OFF + (size_t)(2 * j + 1) * (HIDDEN >> 1));
        const uint16_t* sg = reinterpret_cast<const uint16_t*>(s_sc + (2 * hw) * (HIDDEN >> 4));
        const uint16_t* su = reinterpret_cast<const uint16_t*>(s_sc + (2 * hw + 1) * (HIDDEN >> 4));
        constexpr int V_STRIDE = HIDDEN >> 5; // 72 uint4s per row
        #pragma unroll
        for (int v = lane; v < V_STRIDE; v += 16) {
            acc_g += nvfp4_block32_dot_u4_h(__ldg(wg + v), sg[v], x_h + (v << 5));
            acc_u += nvfp4_block32_dot_u4_h(__ldg(wu + v), su[v], x_h + (v << 5));
        }
    }

    float g = half_warp_sum(acc_g);
    float u = half_warp_sum(acc_u);
    if (lane == 0) {
        float es = exp_scales[expert];
        g *= es; u *= es;
        float g_cap = fast_tanh(g * (1.0f / GATE_SOFTCAP)) * GATE_SOFTCAP;
        float u_cap = fast_tanh(u * (1.0f / UP_SOFTCAP)) * UP_SOFTCAP;
        out_act_h[slot * INTERMEDIATE + j] = __float2half((g_cap * fast_sigmoid(g)) * u_cap);
    }
}

// SLOTTED 1-EXPERT-PER-CTA B=1 Dual-Path Cached MoE Down
// Grid: (HIDDEN/16 = 144, TOP_K = 8), Block: 128 (16 rows/CTA x 8 threads/row)
//   - VRAM hit:  9 exact uint2 iters, zero SMEM/syncthreads (~740 GB/s)
//   - PCIe miss: 1 coalesced 1152B SMEM scale burst + 128B uint4 weight bursts (~23.7 GB/s, +61% vs uint2!)
__global__ void __launch_bounds__(128)
moe_down_nvfp4_b1_cached_kernel(
    const uint8_t* __restrict__ cache_pool,
    const uint8_t* __restrict__ host_pool,
    const int32_t* __restrict__ slot_map,
    const float* __restrict__ exp_scales,
    const __half* __restrict__ act_h,
    const int32_t* __restrict__ r_ids,
    const float* __restrict__ r_w,
    float* __restrict__ out_slots) {
    int slot = blockIdx.y;
    int tid = threadIdx.x;
    int qw = tid >> 3;   // 0..15 (16 rows/CTA!)
    int lane = tid & 7;  // 0..7
    int j_base = blockIdx.x * 16;
    int j = j_base + qw;
    if (j >= HIDDEN) return;

    int expert = r_ids[slot];
    int cslot = __ldg(slot_map + expert);
    const __half* ah = act_h + slot * INTERMEDIATE;
    float acc = 0.0f;

    if (cslot >= 0) {
        const uint8_t* exp_base = cache_pool + (size_t)cslot * EXPERT_SLOT_BYTES;
        constexpr int W_STRIDE = INTERMEDIATE >> 1;
        constexpr int S_STRIDE = INTERMEDIATE >> 4;
        const uint2* wr = reinterpret_cast<const uint2*>(exp_base + EXPERT_W2_DATA_OFF + (size_t)j * W_STRIDE);
        const uint8_t* sr = exp_base + EXPERT_W2_SCALE_OFF + (size_t)j * S_STRIDE;
        #pragma unroll
        for (int iter = 0; iter < 9; ++iter) {
            int blk = lane + iter * 8;
            const uint4* x4 = reinterpret_cast<const uint4*>(ah + (blk << 4));
            uint4 xv0 = __ldg(x4), xv1 = __ldg(x4 + 1);
            __half2 xh[8];
            const __half2* p0 = reinterpret_cast<const __half2*>(&xv0);
            const __half2* p1 = reinterpret_cast<const __half2*>(&xv1);
            xh[0] = p0[0]; xh[1] = p0[1]; xh[2] = p0[2]; xh[3] = p0[3];
            xh[4] = p1[0]; xh[5] = p1[1]; xh[6] = p1[2]; xh[7] = p1[3];

            uint2 w8 = __ldg(wr + blk);
            float sc = cvt_e4m3_to_float(__ldg(sr + blk));
            acc = fmaf(nvfp4_block16_dot(w8, xh), sc, acc);
        }
    } else {
        const uint8_t* exp_base = host_pool + (size_t)expert * EXPERT_SLOT_BYTES;
        __shared__ __align__(16) uint8_t s_sc[16 * (INTERMEDIATE >> 4)]; // 1,152 B = 144 x uint2
        const uint2* sc_g = reinterpret_cast<const uint2*>(exp_base + EXPERT_W2_SCALE_OFF + (size_t)j_base * (INTERMEDIATE >> 4));
        uint2* sc_s = reinterpret_cast<uint2*>(s_sc);
        sc_s[tid] = __ldg(sc_g + tid);
        if (tid < 16) sc_s[128 + tid] = __ldg(sc_g + 128 + tid);
        __syncthreads();

        const uint4* wr = reinterpret_cast<const uint4*>(exp_base + EXPERT_W2_DATA_OFF + (size_t)j * (INTERMEDIATE >> 1));
        const uint16_t* sr = reinterpret_cast<const uint16_t*>(s_sc + qw * (INTERMEDIATE >> 4));
        constexpr int V_STRIDE = INTERMEDIATE >> 5; // 36 uint4s per row
        #pragma unroll
        for (int v = lane; v < V_STRIDE; v += 8) {
            acc += nvfp4_block32_dot_u4_h(__ldg(wr + v), sr[v], ah + (v << 5));
        }
    }

    float dot = quarter_warp_sum(acc);
    if (lane == 0) {
        out_slots[(size_t)slot * HIDDEN + j] = dot * exp_scales[expert] * r_w[slot];
    }
}

// HALF-WARP Dual-Path Grouped Cached MoE Gate+Up (B=2..8 MTP verify)
template <int B_CONST>
__global__ void __launch_bounds__(128)
moe_gate_up_nvfp4_grouped_cached_kernel(
    const uint8_t* __restrict__ cache_pool,
    const uint8_t* __restrict__ host_pool,
    const int32_t* __restrict__ slot_map,
    uint32_t* __restrict__ touch_counts,
    const float* __restrict__ exp_scales,
    const __half* __restrict__ x_h,
    const ExpertGroups* __restrict__ g,
    __half* __restrict__ out_act_h) {
    int u = blockIdx.y;
    if (u >= g->num_unique) return;
    uint32_t mask = g->token_mask[u];
    int expert = g->expert_id[u];
    int tid = threadIdx.x;

    if (blockIdx.x == 0 && tid == 0 && touch_counts) {
        atomicAdd(&touch_counts[expert], 1u);
    }

    int hw = tid >> 4;   // 0..7 (8 j's/CTA!)
    int lane = tid & 15; // 0..15
    int j_base = blockIdx.x * 8;
    int j = j_base + hw;
    if (j >= INTERMEDIATE) return;

    int cslot = __ldg(slot_map + expert);
    const uint8_t* exp_base = (cslot >= 0)
        ? (cache_pool + (size_t)cslot * EXPERT_SLOT_BYTES)
        : (host_pool  + (size_t)expert * EXPERT_SLOT_BYTES);

    __shared__ __align__(16) uint8_t s_sc[16 * (HIDDEN >> 4)];
    if (cslot < 0) {
        const uint4* sc_g = reinterpret_cast<const uint4*>(exp_base + EXPERT_W13_SCALE_OFF + (size_t)(2 * j_base) * (HIDDEN >> 4));
        uint4* sc_s = reinterpret_cast<uint4*>(s_sc);
        sc_s[tid] = __ldg(sc_g + tid);
        if (tid < 16) sc_s[128 + tid] = __ldg(sc_g + 128 + tid);
        __syncthreads();
    }

    if (__popc(mask) == 1) {
        int b = __ffs(mask) - 1;
        const __half* xb = x_h + (size_t)b * HIDDEN;
        float ag = 0.0f, au = 0.0f;
        if (cslot >= 0) {
            constexpr int W_STRIDE = HIDDEN >> 1;
            constexpr int S_STRIDE = HIDDEN >> 4;
            const uint2* wg = reinterpret_cast<const uint2*>(exp_base + EXPERT_W13_DATA_OFF + (size_t)(2 * j) * W_STRIDE);
            const uint2* wu = reinterpret_cast<const uint2*>(exp_base + EXPERT_W13_DATA_OFF + (size_t)(2 * j + 1) * W_STRIDE);
            const uint8_t* sg = exp_base + EXPERT_W13_SCALE_OFF + (size_t)(2 * j) * S_STRIDE;
            const uint8_t* su = exp_base + EXPERT_W13_SCALE_OFF + (size_t)(2 * j + 1) * S_STRIDE;
            #pragma unroll
            for (int iter = 0; iter < 9; ++iter) {
                int blk = lane + iter * 16;
                const uint4* x4 = reinterpret_cast<const uint4*>(xb + (blk << 4));
                uint4 xv0 = __ldg(x4), xv1 = __ldg(x4 + 1);
                __half2 xh[8];
                const __half2* p0 = reinterpret_cast<const __half2*>(&xv0);
                const __half2* p1 = reinterpret_cast<const __half2*>(&xv1);
                xh[0] = p0[0]; xh[1] = p0[1]; xh[2] = p0[2]; xh[3] = p0[3];
                xh[4] = p1[0]; xh[5] = p1[1]; xh[6] = p1[2]; xh[7] = p1[3];
                uint2 w8g = __ldg(wg + blk), w8u = __ldg(wu + blk);
                float scg = cvt_e4m3_to_float(__ldg(sg + blk));
                float scu = cvt_e4m3_to_float(__ldg(su + blk));
                ag = fmaf(nvfp4_block16_dot(w8g, xh), scg, ag);
                au = fmaf(nvfp4_block16_dot(w8u, xh), scu, au);
            }
        } else {
            const uint4* wg = reinterpret_cast<const uint4*>(exp_base + EXPERT_W13_DATA_OFF + (size_t)(2 * j) * (HIDDEN >> 1));
            const uint4* wu = reinterpret_cast<const uint4*>(exp_base + EXPERT_W13_DATA_OFF + (size_t)(2 * j + 1) * (HIDDEN >> 1));
            const uint16_t* sg = reinterpret_cast<const uint16_t*>(s_sc + (2 * hw) * (HIDDEN >> 4));
            const uint16_t* su = reinterpret_cast<const uint16_t*>(s_sc + (2 * hw + 1) * (HIDDEN >> 4));
            #pragma unroll
            for (int v = lane; v < (HIDDEN >> 5); v += 16) {
                ag += nvfp4_block32_dot_u4_h(__ldg(wg + v), sg[v], xb + (v << 5));
                au += nvfp4_block32_dot_u4_h(__ldg(wu + v), su[v], xb + (v << 5));
            }
        }
        float es = exp_scales[expert];
        float gv = half_warp_sum(ag) * es;
        float uv = half_warp_sum(au) * es;
        if (lane == 0) {
            int slot = g->slot_idx[u][b];
            float g_cap = fast_tanh(gv * (1.0f / GATE_SOFTCAP)) * GATE_SOFTCAP;
            float u_cap = fast_tanh(uv * (1.0f / UP_SOFTCAP)) * UP_SOFTCAP;
            out_act_h[((size_t)b * TOP_K + slot) * INTERMEDIATE + j] = __float2half((g_cap * fast_sigmoid(gv)) * u_cap);
        }
        return;
    }

    float acc_g[B_CONST], acc_u[B_CONST];
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) { acc_g[b] = 0.0f; acc_u[b] = 0.0f; }

    if (cslot >= 0) {
        constexpr int W_STRIDE = HIDDEN >> 1;
        constexpr int S_STRIDE = HIDDEN >> 4;
        const uint2* wg = reinterpret_cast<const uint2*>(exp_base + EXPERT_W13_DATA_OFF + (size_t)(2 * j) * W_STRIDE);
        const uint2* wu = reinterpret_cast<const uint2*>(exp_base + EXPERT_W13_DATA_OFF + (size_t)(2 * j + 1) * W_STRIDE);
        const uint8_t* sg = exp_base + EXPERT_W13_SCALE_OFF + (size_t)(2 * j) * S_STRIDE;
        const uint8_t* su = exp_base + EXPERT_W13_SCALE_OFF + (size_t)(2 * j + 1) * S_STRIDE;
        #pragma unroll
        for (int iter = 0; iter < 9; ++iter) {
            int blk = lane + iter * 16;
            uint2 w8g = __ldg(wg + blk), w8u = __ldg(wu + blk);
            float scg = cvt_e4m3_to_float(__ldg(sg + blk));
            float scu = cvt_e4m3_to_float(__ldg(su + blk));
            __half2 gh[8], uh[8];
            cvt_e2m1x8_to_half2x4(w8g.x, gh[0], gh[1], gh[2], gh[3]);
            cvt_e2m1x8_to_half2x4(w8g.y, gh[4], gh[5], gh[6], gh[7]);
            cvt_e2m1x8_to_half2x4(w8u.x, uh[0], uh[1], uh[2], uh[3]);
            cvt_e2m1x8_to_half2x4(w8u.y, uh[4], uh[5], uh[6], uh[7]);
            int koff = blk << 4;
            #pragma unroll
            for (int b = 0; b < B_CONST; ++b) {
                if (mask & (1u << b)) {
                    const uint4* x4 = reinterpret_cast<const uint4*>(x_h + (size_t)b * HIDDEN + koff);
                    uint4 xv0 = __ldg(x4), xv1 = __ldg(x4 + 1);
                    const __half2* xb0 = reinterpret_cast<const __half2*>(&xv0);
                    const __half2* xb1 = reinterpret_cast<const __half2*>(&xv1);
                    __half2 pg0 = __hmul2(gh[0], xb0[0]), pg1 = __hmul2(gh[1], xb0[1]);
                    __half2 pu0 = __hmul2(uh[0], xb0[0]), pu1 = __hmul2(uh[1], xb0[1]);
                    pg0 = __hfma2(gh[2], xb0[2], pg0); pg1 = __hfma2(gh[3], xb0[3], pg1);
                    pu0 = __hfma2(uh[2], xb0[2], pu0); pu1 = __hfma2(uh[3], xb0[3], pu1);
                    pg0 = __hfma2(gh[4], xb1[0], pg0); pg1 = __hfma2(gh[5], xb1[1], pg1);
                    pu0 = __hfma2(uh[4], xb1[0], pu0); pu1 = __hfma2(uh[5], xb1[1], pu1);
                    pg0 = __hfma2(gh[6], xb1[2], pg0); pg1 = __hfma2(gh[7], xb1[3], pg1);
                    pu0 = __hfma2(uh[6], xb1[2], pu0); pu1 = __hfma2(uh[7], xb1[3], pu1);
                    float2 fg = __half22float2(__hadd2(pg0, pg1));
                    float2 fu = __half22float2(__hadd2(pu0, pu1));
                    acc_g[b] = fmaf(fg.x + fg.y, scg, acc_g[b]);
                    acc_u[b] = fmaf(fu.x + fu.y, scu, acc_u[b]);
                }
            }
        }
    } else {
        const uint4* wg = reinterpret_cast<const uint4*>(exp_base + EXPERT_W13_DATA_OFF + (size_t)(2 * j) * (HIDDEN >> 1));
        const uint4* wu = reinterpret_cast<const uint4*>(exp_base + EXPERT_W13_DATA_OFF + (size_t)(2 * j + 1) * (HIDDEN >> 1));
        const uint16_t* sg = reinterpret_cast<const uint16_t*>(s_sc + (2 * hw) * (HIDDEN >> 4));
        const uint16_t* su = reinterpret_cast<const uint16_t*>(s_sc + (2 * hw + 1) * (HIDDEN >> 4));
        #pragma unroll
        for (int v = lane; v < (HIDDEN >> 5); v += 16) {
            uint4 w16g = __ldg(wg + v), w16u = __ldg(wu + v);
            uint16_t scg2 = sg[v], scu2 = su[v];
            float scg0 = cvt_e4m3_to_float(uint8_t(scg2 & 0xFFu));
            float scg1 = cvt_e4m3_to_float(uint8_t(scg2 >> 8));
            float scu0 = cvt_e4m3_to_float(uint8_t(scu2 & 0xFFu));
            float scu1 = cvt_e4m3_to_float(uint8_t(scu2 >> 8));
            __half2 gh0[8], gh1[8], uh0[8], uh1[8];
            cvt_e2m1x8_to_half2x4(w16g.x, gh0[0], gh0[1], gh0[2], gh0[3]);
            cvt_e2m1x8_to_half2x4(w16g.y, gh0[4], gh0[5], gh0[6], gh0[7]);
            cvt_e2m1x8_to_half2x4(w16g.z, gh1[0], gh1[1], gh1[2], gh1[3]);
            cvt_e2m1x8_to_half2x4(w16g.w, gh1[4], gh1[5], gh1[6], gh1[7]);
            cvt_e2m1x8_to_half2x4(w16u.x, uh0[0], uh0[1], uh0[2], uh0[3]);
            cvt_e2m1x8_to_half2x4(w16u.y, uh0[4], uh0[5], uh0[6], uh0[7]);
            cvt_e2m1x8_to_half2x4(w16u.z, uh1[0], uh1[1], uh1[2], uh1[3]);
            cvt_e2m1x8_to_half2x4(w16u.w, uh1[4], uh1[5], uh1[6], uh1[7]);
            int koff = v << 5;
            #pragma unroll
            for (int b = 0; b < B_CONST; ++b) {
                if (mask & (1u << b)) {
                    const uint4* x4 = reinterpret_cast<const uint4*>(x_h + (size_t)b * HIDDEN + koff);
                    uint4 xv0 = __ldg(x4), xv1 = __ldg(x4 + 1), xv2 = __ldg(x4 + 2), xv3 = __ldg(x4 + 3);
                    const __half2* xb0 = reinterpret_cast<const __half2*>(&xv0);
                    const __half2* xb1 = reinterpret_cast<const __half2*>(&xv1);
                    const __half2* xb2 = reinterpret_cast<const __half2*>(&xv2);
                    const __half2* xb3 = reinterpret_cast<const __half2*>(&xv3);
                    __half2 pg0 = __hmul2(gh0[0], xb0[0]), pg1 = __hmul2(gh0[1], xb0[1]);
                    __half2 pu0 = __hmul2(uh0[0], xb0[0]), pu1 = __hmul2(uh0[1], xb0[1]);
                    pg0 = __hfma2(gh0[2], xb0[2], pg0); pg1 = __hfma2(gh0[3], xb0[3], pg1);
                    pu0 = __hfma2(uh0[2], xb0[2], pu0); pu1 = __hfma2(uh0[3], xb0[3], pu1);
                    pg0 = __hfma2(gh0[4], xb1[0], pg0); pg1 = __hfma2(gh0[5], xb1[1], pg1);
                    pu0 = __hfma2(uh0[4], xb1[0], pu0); pu1 = __hfma2(uh0[5], xb1[1], pu1);
                    pg0 = __hfma2(gh0[6], xb1[2], pg0); pg1 = __hfma2(gh0[7], xb1[3], pg1);
                    pu0 = __hfma2(uh0[6], xb1[2], pu0); pu1 = __hfma2(uh0[7], xb1[3], pu1);
                    float2 fg0 = __half22float2(__hadd2(pg0, pg1));
                    float2 fu0 = __half22float2(__hadd2(pu0, pu1));
                    acc_g[b] = fmaf(fg0.x + fg0.y, scg0, acc_g[b]);
                    acc_u[b] = fmaf(fu0.x + fu0.y, scu0, acc_u[b]);

                    pg0 = __hmul2(gh1[0], xb2[0]); pg1 = __hmul2(gh1[1], xb2[1]);
                    pu0 = __hmul2(uh1[0], xb2[0]); pu1 = __hmul2(uh1[1], xb2[1]);
                    pg0 = __hfma2(gh1[2], xb2[2], pg0); pg1 = __hfma2(gh1[3], xb2[3], pg1);
                    pu0 = __hfma2(uh1[2], xb2[2], pu0); pu1 = __hfma2(uh1[3], xb2[3], pu1);
                    pg0 = __hfma2(gh1[4], xb3[0], pg0); pg1 = __hfma2(gh1[5], xb3[1], pg1);
                    pu0 = __hfma2(uh1[4], xb3[0], pu0); pu1 = __hfma2(uh1[5], xb3[1], pu1);
                    pg0 = __hfma2(gh1[6], xb3[2], pg0); pg1 = __hfma2(gh1[7], xb3[3], pg1);
                    pu0 = __hfma2(uh1[6], xb3[2], pu0); pu1 = __hfma2(uh1[7], xb3[3], pu1);
                    float2 fg1 = __half22float2(__hadd2(pg0, pg1));
                    float2 fu1 = __half22float2(__hadd2(pu0, pu1));
                    acc_g[b] = fmaf(fg1.x + fg1.y, scg1, acc_g[b]);
                    acc_u[b] = fmaf(fu1.x + fu1.y, scu1, acc_u[b]);
                }
            }
        }
    }

    float es = exp_scales[expert];
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) {
        if (mask & (1u << b)) {
            float gv = half_warp_sum(acc_g[b]) * es;
            float uv = half_warp_sum(acc_u[b]) * es;
            if (lane == 0) {
                int slot = g->slot_idx[u][b];
                float g_cap = fast_tanh(gv * (1.0f / GATE_SOFTCAP)) * GATE_SOFTCAP;
                float u_cap = fast_tanh(uv * (1.0f / UP_SOFTCAP)) * UP_SOFTCAP;
                out_act_h[((size_t)b * TOP_K + slot) * INTERMEDIATE + j] = __float2half((g_cap * fast_sigmoid(gv)) * u_cap);
            }
        }
    }
}

template <int B_CONST>
__global__ void __launch_bounds__(128)
moe_down_nvfp4_grouped_cached_kernel(
    const uint8_t* __restrict__ cache_pool,
    const uint8_t* __restrict__ host_pool,
    const int32_t* __restrict__ slot_map,
    const float* __restrict__ exp_scales,
    const __half* __restrict__ act_h,
    const ExpertGroups* __restrict__ g,
    float* __restrict__ out_down) {
    int u = blockIdx.y;
    if (u >= g->num_unique) return;
    uint32_t mask = g->token_mask[u];
    int expert = g->expert_id[u];
    int tid = threadIdx.x;

    int qw = tid >> 3;
    int lane = tid & 7;
    int j_base = blockIdx.x * 16;
    int j = j_base + qw;
    if (j >= HIDDEN) return;

    int cslot = __ldg(slot_map + expert);
    const uint8_t* exp_base = (cslot >= 0)
        ? (cache_pool + (size_t)cslot * EXPERT_SLOT_BYTES)
        : (host_pool  + (size_t)expert * EXPERT_SLOT_BYTES);

    __shared__ __align__(16) uint8_t s_sc[16 * (INTERMEDIATE >> 4)]; // 1,152 B = 144 x uint2
    if (cslot < 0) {
        const uint2* sc_g = reinterpret_cast<const uint2*>(exp_base + EXPERT_W2_SCALE_OFF + (size_t)j_base * (INTERMEDIATE >> 4));
        uint2* sc_s = reinterpret_cast<uint2*>(s_sc);
        sc_s[tid] = __ldg(sc_g + tid);
        if (tid < 16) sc_s[128 + tid] = __ldg(sc_g + 128 + tid);
        __syncthreads();
    }

    if (__popc(mask) == 1) {
        int b = __ffs(mask) - 1;
        int slot = g->slot_idx[u][b];
        const __half* ah = act_h + ((size_t)b * TOP_K + slot) * INTERMEDIATE;
        float acc = 0.0f;
        if (cslot >= 0) {
            constexpr int W_STRIDE = INTERMEDIATE >> 1;
            constexpr int S_STRIDE = INTERMEDIATE >> 4;
            const uint2* wr = reinterpret_cast<const uint2*>(exp_base + EXPERT_W2_DATA_OFF + (size_t)j * W_STRIDE);
            const uint8_t* sr = exp_base + EXPERT_W2_SCALE_OFF + (size_t)j * S_STRIDE;
            #pragma unroll
            for (int iter = 0; iter < 9; ++iter) {
                int blk = lane + iter * 8;
                const uint4* x4 = reinterpret_cast<const uint4*>(ah + (blk << 4));
                uint4 xv0 = __ldg(x4), xv1 = __ldg(x4 + 1);
                __half2 xh[8];
                const __half2* p0 = reinterpret_cast<const __half2*>(&xv0);
                const __half2* p1 = reinterpret_cast<const __half2*>(&xv1);
                xh[0] = p0[0]; xh[1] = p0[1]; xh[2] = p0[2]; xh[3] = p0[3];
                xh[4] = p1[0]; xh[5] = p1[1]; xh[6] = p1[2]; xh[7] = p1[3];
                uint2 w8 = __ldg(wr + blk);
                float sc = cvt_e4m3_to_float(__ldg(sr + blk));
                acc = fmaf(nvfp4_block16_dot(w8, xh), sc, acc);
            }
        } else {
            const uint4* wr = reinterpret_cast<const uint4*>(exp_base + EXPERT_W2_DATA_OFF + (size_t)j * (INTERMEDIATE >> 1));
            const uint16_t* sr = reinterpret_cast<const uint16_t*>(s_sc + qw * (INTERMEDIATE >> 4));
            #pragma unroll
            for (int v = lane; v < (INTERMEDIATE >> 5); v += 8) {
                acc += nvfp4_block32_dot_u4_h(__ldg(wr + v), sr[v], ah + (v << 5));
            }
        }
        float val = quarter_warp_sum(acc) * exp_scales[expert];
        if (lane == 0) out_down[((size_t)b * TOP_K + slot) * HIDDEN + j] = val * g->route_w[u][b];
        return;
    }

    float acc[B_CONST];
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) acc[b] = 0.0f;

    if (cslot >= 0) {
        constexpr int W_STRIDE = INTERMEDIATE >> 1;
        constexpr int S_STRIDE = INTERMEDIATE >> 4;
        const uint2* wr = reinterpret_cast<const uint2*>(exp_base + EXPERT_W2_DATA_OFF + (size_t)j * W_STRIDE);
        const uint8_t* sr = exp_base + EXPERT_W2_SCALE_OFF + (size_t)j * S_STRIDE;
        #pragma unroll
        for (int iter = 0; iter < 9; ++iter) {
            int blk = lane + iter * 8;
            uint2 w8 = __ldg(wr + blk);
            float sc = cvt_e4m3_to_float(__ldg(sr + blk));
            __half2 wh[8];
            cvt_e2m1x8_to_half2x4(w8.x, wh[0], wh[1], wh[2], wh[3]);
            cvt_e2m1x8_to_half2x4(w8.y, wh[4], wh[5], wh[6], wh[7]);
            int koff = blk << 4;
            #pragma unroll
            for (int b = 0; b < B_CONST; ++b) {
                if (mask & (1u << b)) {
                    int slot = g->slot_idx[u][b];
                    const uint4* x4 = reinterpret_cast<const uint4*>(act_h + ((size_t)b * TOP_K + slot) * INTERMEDIATE + koff);
                    uint4 xv0 = __ldg(x4), xv1 = __ldg(x4 + 1);
                    const __half2* xb0 = reinterpret_cast<const __half2*>(&xv0);
                    const __half2* xb1 = reinterpret_cast<const __half2*>(&xv1);
                    __half2 p0 = __hmul2(wh[0], xb0[0]), p1 = __hmul2(wh[1], xb0[1]);
                    p0 = __hfma2(wh[2], xb0[2], p0); p1 = __hfma2(wh[3], xb0[3], p1);
                    p0 = __hfma2(wh[4], xb1[0], p0); p1 = __hfma2(wh[5], xb1[1], p1);
                    p0 = __hfma2(wh[6], xb1[2], p0); p1 = __hfma2(wh[7], xb1[3], p1);
                    float2 pf = __half22float2(__hadd2(p0, p1));
                    acc[b] = fmaf(pf.x + pf.y, sc, acc[b]);
                }
            }
        }
    } else {
        const uint4* wr = reinterpret_cast<const uint4*>(exp_base + EXPERT_W2_DATA_OFF + (size_t)j * (INTERMEDIATE >> 1));
        const uint16_t* sr = reinterpret_cast<const uint16_t*>(s_sc + qw * (INTERMEDIATE >> 4));
        #pragma unroll
        for (int v = lane; v < (INTERMEDIATE >> 5); v += 8) {
            uint4 w16 = __ldg(wr + v);
            uint16_t sc2 = sr[v];
            float sc0 = cvt_e4m3_to_float(uint8_t(sc2 & 0xFFu));
            float sc1 = cvt_e4m3_to_float(uint8_t(sc2 >> 8));
            __half2 wh0[8], wh1[8];
            cvt_e2m1x8_to_half2x4(w16.x, wh0[0], wh0[1], wh0[2], wh0[3]);
            cvt_e2m1x8_to_half2x4(w16.y, wh0[4], wh0[5], wh0[6], wh0[7]);
            cvt_e2m1x8_to_half2x4(w16.z, wh1[0], wh1[1], wh1[2], wh1[3]);
            cvt_e2m1x8_to_half2x4(w16.w, wh1[4], wh1[5], wh1[6], wh1[7]);
            int koff = v << 5;
            #pragma unroll
            for (int b = 0; b < B_CONST; ++b) {
                if (mask & (1u << b)) {
                    int slot = g->slot_idx[u][b];
                    const uint4* x4 = reinterpret_cast<const uint4*>(act_h + ((size_t)b * TOP_K + slot) * INTERMEDIATE + koff);
                    uint4 xv0 = __ldg(x4), xv1 = __ldg(x4 + 1), xv2 = __ldg(x4 + 2), xv3 = __ldg(x4 + 3);
                    const __half2* xb0 = reinterpret_cast<const __half2*>(&xv0);
                    const __half2* xb1 = reinterpret_cast<const __half2*>(&xv1);
                    const __half2* xb2 = reinterpret_cast<const __half2*>(&xv2);
                    const __half2* xb3 = reinterpret_cast<const __half2*>(&xv3);
                    __half2 p0 = __hmul2(wh0[0], xb0[0]), p1 = __hmul2(wh0[1], xb0[1]);
                    p0 = __hfma2(wh0[2], xb0[2], p0); p1 = __hfma2(wh0[3], xb0[3], p1);
                    p0 = __hfma2(wh0[4], xb1[0], p0); p1 = __hfma2(wh0[5], xb1[1], p1);
                    p0 = __hfma2(wh0[6], xb1[2], p0); p1 = __hfma2(wh0[7], xb1[3], p1);
                    float2 pf0 = __half22float2(__hadd2(p0, p1));
                    acc[b] = fmaf(pf0.x + pf0.y, sc0, acc[b]);

                    p0 = __hmul2(wh1[0], xb2[0]); p1 = __hmul2(wh1[1], xb2[1]);
                    p0 = __hfma2(wh1[2], xb2[2], p0); p1 = __hfma2(wh1[3], xb2[3], p1);
                    p0 = __hfma2(wh1[4], xb3[0], p0); p1 = __hfma2(wh1[5], xb3[1], p1);
                    p0 = __hfma2(wh1[6], xb3[2], p0); p1 = __hfma2(wh1[7], xb3[3], p1);
                    float2 pf1 = __half22float2(__hadd2(p0, p1));
                    acc[b] = fmaf(pf1.x + pf1.y, sc1, acc[b]);
                }
            }
        }
    }

    float es = exp_scales[expert];
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) {
        if (mask & (1u << b)) {
            float val = quarter_warp_sum(acc[b]) * es;
            if (lane == 0) {
                int slot = g->slot_idx[u][b];
                out_down[((size_t)b * TOP_K + slot) * HIDDEN + j] = val * g->route_w[u][b];
            }
        }
    }
}

} // namespace

void launch_f32_to_f16(const float* src, __half* dst, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = ((n + 3) / 4 + threads - 1) / threads;
    f32_to_f16_kernel<<<blocks, threads, 0, stream>>>(src, dst, n);
}

void launch_linear(
    const LinearWeight& w, const float* x, float* y, int B, bool add_residual, cudaStream_t stream) {
    while (B > 0) {
        int chunk = (B >= 8) ? 8 : B;
        switch (chunk) {
            case 1: dispatch_linear_b<1>(w, x, y, add_residual, stream); break;
            case 2: dispatch_linear_b<2>(w, x, y, add_residual, stream); break;
            case 3: dispatch_linear_b<3>(w, x, y, add_residual, stream); break;
            case 4: dispatch_linear_b<4>(w, x, y, add_residual, stream); break;
            case 5: dispatch_linear_b<5>(w, x, y, add_residual, stream); break;
            case 6: dispatch_linear_b<6>(w, x, y, add_residual, stream); break;
            case 7: dispatch_linear_b<7>(w, x, y, add_residual, stream); break;
            case 8: dispatch_linear_b<8>(w, x, y, add_residual, stream); break;
        }
        x += (size_t)chunk * w.cols;
        y += (size_t)chunk * w.rows;
        B -= chunk;
    }
}

void launch_linear_h(
    const LinearWeight& w, const __half* x_h, float* y, int B, bool add_residual, cudaStream_t stream) {
    switch (B) {
        case 1: dispatch_linear_h_b<1>(w, x_h, y, add_residual, stream); break;
        case 2: dispatch_linear_h_b<2>(w, x_h, y, add_residual, stream); break;
        case 3: dispatch_linear_h_b<3>(w, x_h, y, add_residual, stream); break;
        case 4: dispatch_linear_h_b<4>(w, x_h, y, add_residual, stream); break;
        case 5: dispatch_linear_h_b<5>(w, x_h, y, add_residual, stream); break;
        case 6: dispatch_linear_h_b<6>(w, x_h, y, add_residual, stream); break;
        case 7: dispatch_linear_h_b<7>(w, x_h, y, add_residual, stream); break;
        case 8: dispatch_linear_h_b<8>(w, x_h, y, add_residual, stream); break;
    }
}

template <int B_CONST>
static void dispatch_qkvrg_b(
    const LayerWeights& lw, const __half* xq, const __half* xk, const __half* xv,
    float* y_qrgs, float* y_k, float* y_v, cudaStream_t stream) {
    constexpr int N_TOT = (Q_DIM + R_DIM + Q_DIM + NUM_HEADS) + 2 * KV_DIM; // 8600
    int grid = (N_TOT + 7) / 8; // 1075 half-warp blocks!
    qkvrg_fused_nvfp4_kernel<B_CONST><<<grid, 128, 0, stream>>>(
        static_cast<const uint8_t*>(lw.w_qrgs.data), lw.w_qrgs.scales, lw.w_qrgs.global_scale,
        static_cast<const uint8_t*>(lw.w_k.data), lw.w_k.scales, lw.w_k.global_scale,
        static_cast<const uint8_t*>(lw.w_v.data), lw.w_v.scales, lw.w_v.global_scale,
        xq, xk, xv, y_qrgs, y_k, y_v);
}

void launch_qkvrg_fused(
    const LayerWeights& lw, const __half* in_q_h, const __half* in_k_h, const __half* in_v_h,
    float* qrgs_out, float* k_out, float* v_out, int B, cudaStream_t stream) {
    switch (B) {
        case 1: dispatch_qkvrg_b<1>(lw, in_q_h, in_k_h, in_v_h, qrgs_out, k_out, v_out, stream); break;
        case 2: dispatch_qkvrg_b<2>(lw, in_q_h, in_k_h, in_v_h, qrgs_out, k_out, v_out, stream); break;
        case 3: dispatch_qkvrg_b<3>(lw, in_q_h, in_k_h, in_v_h, qrgs_out, k_out, v_out, stream); break;
        case 4: dispatch_qkvrg_b<4>(lw, in_q_h, in_k_h, in_v_h, qrgs_out, k_out, v_out, stream); break;
        case 5: dispatch_qkvrg_b<5>(lw, in_q_h, in_k_h, in_v_h, qrgs_out, k_out, v_out, stream); break;
        case 6: dispatch_qkvrg_b<6>(lw, in_q_h, in_k_h, in_v_h, qrgs_out, k_out, v_out, stream); break;
        case 7: dispatch_qkvrg_b<7>(lw, in_q_h, in_k_h, in_v_h, qrgs_out, k_out, v_out, stream); break;
        case 8: dispatch_qkvrg_b<8>(lw, in_q_h, in_k_h, in_v_h, qrgs_out, k_out, v_out, stream); break;
    }
}

void launch_bf16_linear(
    const __nv_bfloat16* w, const float* x, float* y, int B, int N, int K, cudaStream_t stream) {
    LinearWeight lw;
    lw.kind = WeightKind::BF16;
    lw.rows = N;
    lw.cols = K;
    lw.data = const_cast<__nv_bfloat16*>(w);
    launch_linear(lw, x, y, B, false, stream);
}

template <int B_CONST>
static void dispatch_lm_head(
    const LinearWeight& u, const float* x, int32_t* out_toks, float* out_logits,
    float* s_val, int32_t* s_idx, cudaStream_t stream) {
    if (u.kind == WeightKind::NVFP4) {
        lm_head_stage1_kernel<B_CONST, 2><<<LM_HEAD_BLOCKS, 256, 0, stream>>>(
            u.data, u.scales, nullptr, u.global_scale, x, out_logits, s_val, s_idx, u.rows);
    } else if (u.kind == WeightKind::FP8) {
        lm_head_stage1_kernel<B_CONST, 1><<<LM_HEAD_BLOCKS, 256, 0, stream>>>(
            u.data, nullptr, u.row_scales, 1.0f, x, out_logits, s_val, s_idx, u.rows);
    } else {
        lm_head_stage1_kernel<B_CONST, 0><<<LM_HEAD_BLOCKS, 256, 0, stream>>>(
            u.data, nullptr, nullptr, 1.0f, x, out_logits, s_val, s_idx, u.rows);
    }
    lm_head_stage2_argmax_kernel<<<B_CONST, 256, 0, stream>>>(s_val, s_idx, out_toks, B_CONST);
}

void launch_lm_head_argmax(
    const LinearWeight& unembed, const float* x, int32_t* out_tokens, float* out_logits,
    float* scratch_vals, int32_t* scratch_idxs, int B, cudaStream_t stream) {
    switch (B) {
        case 1: dispatch_lm_head<1>(unembed, x, out_tokens, out_logits, scratch_vals, scratch_idxs, stream); break;
        case 2: dispatch_lm_head<2>(unembed, x, out_tokens, out_logits, scratch_vals, scratch_idxs, stream); break;
        case 3: dispatch_lm_head<3>(unembed, x, out_tokens, out_logits, scratch_vals, scratch_idxs, stream); break;
        case 4: dispatch_lm_head<4>(unembed, x, out_tokens, out_logits, scratch_vals, scratch_idxs, stream); break;
        case 5: dispatch_lm_head<5>(unembed, x, out_tokens, out_logits, scratch_vals, scratch_idxs, stream); break;
        case 6: dispatch_lm_head<6>(unembed, x, out_tokens, out_logits, scratch_vals, scratch_idxs, stream); break;
        case 7: dispatch_lm_head<7>(unembed, x, out_tokens, out_logits, scratch_vals, scratch_idxs, stream); break;
        case 8: dispatch_lm_head<8>(unembed, x, out_tokens, out_logits, scratch_vals, scratch_idxs, stream); break;
        default:
            for (int i = 0; i < B; ++i) {
                dispatch_lm_head<1>(unembed, x + (size_t)i * HIDDEN, out_tokens + i,
                    out_logits ? out_logits + (size_t)i * unembed.rows : nullptr,
                    scratch_vals, scratch_idxs, stream);
            }
            break;
    }
}

void launch_build_expert_groups(
    const int32_t* route_ids, const float* route_w, ExpertGroups* groups, int B, cudaStream_t stream) {
    build_expert_groups_kernel<<<1, 1, 0, stream>>>(route_ids, route_w, groups, B);
}

template <int B_CONST>
static void dispatch_moe_gu_grouped(
    const ExpertBank& w13, const __half* x_h, const ExpertGroups* groups, __half* out_act_h, cudaStream_t stream) {
    if (w13.cache_pool != nullptr) {
        dim3 grid((INTERMEDIATE + 7) / 8, B_CONST * TOP_K);
        moe_gate_up_nvfp4_grouped_cached_kernel<B_CONST><<<grid, 128, 0, stream>>>(
            w13.cache_pool, w13.host_pool, w13.slot_map, w13.touch_counts, w13.expert_scale, x_h, groups, out_act_h);
    } else if (w13.is_nvfp4) {
        dim3 grid((INTERMEDIATE + 3) / 4, B_CONST * TOP_K);
        moe_gate_up_nvfp4_grouped_h_kernel<B_CONST><<<grid, 128, 0, stream>>>(
            w13.data, w13.scales, w13.expert_scale, x_h, groups, out_act_h);
    } else {
        dim3 grid((INTERMEDIATE + 3) / 4, B_CONST * TOP_K);
        moe_gate_up_mxfp8_grouped_h_kernel<B_CONST><<<grid, 128, 0, stream>>>(
            w13.data, w13.scales, x_h, groups, out_act_h);
    }
}

void launch_moe_gate_up_h(
    const ExpertBank& w13, const __half* x_h, const int32_t* route_ids,
    const ExpertGroups* groups, __half* out_act_h, int B, cudaStream_t stream) {
    if (B == 1) {
        if (w13.cache_pool != nullptr) {
            dim3 grid((INTERMEDIATE + 7) / 8, TOP_K);
            moe_gate_up_nvfp4_b1_cached_kernel<<<grid, 128, 0, stream>>>(
                w13.cache_pool, w13.host_pool, w13.slot_map, w13.touch_counts, w13.expert_scale, x_h, route_ids, out_act_h);
        } else if (w13.is_nvfp4) {
            dim3 grid((INTERMEDIATE + 3) / 4, TOP_K);
            moe_gate_up_nvfp4_b1_h_kernel<<<grid, 128, 0, stream>>>(
                w13.data, w13.scales, w13.expert_scale, x_h, route_ids, out_act_h);
        } else {
            dim3 grid((INTERMEDIATE + 3) / 4, TOP_K);
            moe_gate_up_mxfp8_b1_h_kernel<<<grid, 128, 0, stream>>>(
                w13.data, w13.scales, x_h, route_ids, out_act_h);
        }
    } else {
        switch (B) {
            case 2: dispatch_moe_gu_grouped<2>(w13, x_h, groups, out_act_h, stream); break;
            case 3: dispatch_moe_gu_grouped<3>(w13, x_h, groups, out_act_h, stream); break;
            case 4: dispatch_moe_gu_grouped<4>(w13, x_h, groups, out_act_h, stream); break;
            case 5: dispatch_moe_gu_grouped<5>(w13, x_h, groups, out_act_h, stream); break;
            case 6: dispatch_moe_gu_grouped<6>(w13, x_h, groups, out_act_h, stream); break;
            case 7: dispatch_moe_gu_grouped<7>(w13, x_h, groups, out_act_h, stream); break;
            case 8: dispatch_moe_gu_grouped<8>(w13, x_h, groups, out_act_h, stream); break;
        }
    }
}

template <int B_CONST>
static void dispatch_moe_dn_grouped(
    const ExpertBank& w2, const __half* act_h, const ExpertGroups* groups, float* out_down, cudaStream_t stream) {
    if (w2.cache_pool != nullptr) {
        dim3 grid((HIDDEN + 15) / 16, B_CONST * TOP_K);
        moe_down_nvfp4_grouped_cached_kernel<B_CONST><<<grid, 128, 0, stream>>>(
            w2.cache_pool, w2.host_pool, w2.slot_map, w2.expert_scale, act_h, groups, out_down);
    } else if (w2.is_nvfp4) {
        dim3 grid((HIDDEN + 15) / 16, B_CONST * TOP_K);
        moe_down_nvfp4_grouped_h_kernel<B_CONST><<<grid, 128, 0, stream>>>(
            w2.data, w2.scales, w2.expert_scale, act_h, groups, out_down);
    } else {
        dim3 grid((HIDDEN + 3) / 4, B_CONST * TOP_K);
        moe_down_mxfp8_grouped_h_kernel<B_CONST><<<grid, 128, 0, stream>>>(
            w2.data, w2.scales, act_h, groups, out_down);
    }
}

void launch_moe_down_h(
    const ExpertBank& w2, const __half* act_h, const int32_t* route_ids, const float* route_w,
    const ExpertGroups* groups, float* out_down, int B, cudaStream_t stream, bool already_zeroed) {
    if (B == 1) {
        if (w2.cache_pool != nullptr) {
            dim3 grid((HIDDEN + 15) / 16, TOP_K);
            moe_down_nvfp4_b1_cached_kernel<<<grid, 128, 0, stream>>>(
                w2.cache_pool, w2.host_pool, w2.slot_map, w2.expert_scale, act_h, route_ids, route_w, out_down);
        } else if (w2.is_nvfp4)
            moe_down_nvfp4_b1_h_kernel<<<(HIDDEN + 3) / 4, 256, 0, stream>>>(
                w2.data, w2.scales, w2.expert_scale, act_h, route_ids, route_w, out_down);
        else
            moe_down_mxfp8_b1_h_kernel<<<HIDDEN, 256, 0, stream>>>(
                w2.data, w2.scales, act_h, route_ids, route_w, out_down);
    } else {
        if (!already_zeroed) {
            CUDA_CHECK(cudaMemsetAsync(out_down, 0, (size_t)B * HIDDEN * sizeof(float), stream));
        }
        switch (B) {
            case 2: dispatch_moe_dn_grouped<2>(w2, act_h, groups, out_down, stream); break;
            case 3: dispatch_moe_dn_grouped<3>(w2, act_h, groups, out_down, stream); break;
            case 4: dispatch_moe_dn_grouped<4>(w2, act_h, groups, out_down, stream); break;
            case 5: dispatch_moe_dn_grouped<5>(w2, act_h, groups, out_down, stream); break;
            case 6: dispatch_moe_dn_grouped<6>(w2, act_h, groups, out_down, stream); break;
            case 7: dispatch_moe_dn_grouped<7>(w2, act_h, groups, out_down, stream); break;
            case 8: dispatch_moe_dn_grouped<8>(w2, act_h, groups, out_down, stream); break;
        }
    }
}

} // namespace squid
