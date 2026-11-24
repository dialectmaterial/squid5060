#include "kernels.hpp"
#include "kernels_common.cuh"

namespace squid {

namespace {

// Mix 4 target taps [4, B, HIDDEN] using fuse_alpha[m, 0..3], RMSNorm + dflare_norm -> out_mix[m, b, HIDDEN]
// m = 0,1,2 -> draft layers 0..2; m = 3 -> markov f stored into f_ring[(pos+b) & 15]
__global__ void dspark_mix_kernel(
    const float* __restrict__ taps,       // [4, B, HIDDEN]
    int B,
    const float* __restrict__ fuse_alpha, // [4, 4]
    const __nv_bfloat16* __restrict__ n0,
    const __nv_bfloat16* __restrict__ n1,
    const __nv_bfloat16* __restrict__ n2,
    const __nv_bfloat16* __restrict__ n3,
    float* __restrict__ out_mix,          // [3, B, HIDDEN]
    float* __restrict__ f_ring,           // [16, HIDDEN]
    const int32_t* __restrict__ d_pos) {
    int b = blockIdx.x;
    int m = blockIdx.y; // 0..3
    int tid = threadIdx.x;
    const float a0 = fuse_alpha[m * 4 + 0];
    const float a1 = fuse_alpha[m * 4 + 1];
    const float a2 = fuse_alpha[m * 4 + 2];
    const float a3 = fuse_alpha[m * 4 + 3];

    const float* t0 = taps + ((size_t)0 * B + b) * HIDDEN;
    const float* t1 = taps + ((size_t)1 * B + b) * HIDDEN;
    const float* t2 = taps + ((size_t)2 * B + b) * HIDDEN;
    const float* t3 = taps + ((size_t)3 * B + b) * HIDDEN;

    __shared__ float buf[HIDDEN];
    float sum_sq = 0.0f;
    for (int d = tid; d < HIDDEN; d += 256) {
        float v = a0 * t0[d] + a1 * t1[d] + a2 * t2[d] + a3 * t3[d];
        buf[d] = v;
        sum_sq = fmaf(v, v, sum_sq);
    }
    sum_sq = warp_sum(sum_sq);
    __shared__ float sm[8];
    if ((tid & 31) == 0) sm[tid >> 5] = sum_sq;
    __syncthreads();
    __shared__ float inv_rms;
    if (tid == 0) {
        float tot = 0.0f;
        for (int i = 0; i < 8; ++i) tot += sm[i];
        inv_rms = rsqrtf(tot / float(HIDDEN) + RMS_EPS);
    }
    __syncthreads();

    const __nv_bfloat16* nw = (m == 0) ? n0 : (m == 1 ? n1 : (m == 2 ? n2 : n3));
    float* dst = (m < 3)
        ? (out_mix + ((size_t)m * B + b) * HIDDEN)
        : (f_ring + (size_t)(((*d_pos) + b) & 15) * HIDDEN);
    float s = inv_rms;
    for (int d = tid; d < HIDDEN; d += 256) {
        dst[d] = buf[d] * s * __bfloat162float(nw[d]);
    }
}

// NeoX RoPE (theta = 1e6) + per-head RMSNorm -> store ctx K & V into draft KV cache (BF16/FP8/Q4!)
// Input kv [B, 2*KV_DIM] (K at 0..1023, V at 1024..2047)
template <int KV_FMT>
__global__ void dspark_store_ctx_kv_kernel(
    const float* __restrict__ kv,
    const __nv_bfloat16* __restrict__ k_norm, // [128]
    void* __restrict__ k_cache,
    void* __restrict__ v_cache,
    __half* __restrict__ k_scale,
    __half* __restrict__ v_scale,
    int cap,
    const int32_t* __restrict__ d_pos) {
    int b = blockIdx.x;
    int h = blockIdx.y; // 0..7
    int tid = threadIdx.x; // 0..127 (full head_dim!)
    int pos = (*d_pos) + b;
    int slot = pos % cap;

    const float* kp = kv + (size_t)b * (2 * KV_DIM) + h * HEAD_DIM;
    const float* vp = kv + (size_t)b * (2 * KV_DIM) + KV_DIM + h * HEAD_DIM;

    __shared__ float sm[4];
    // Store V
    store_kv_head_128<KV_FMT>(vp[tid], slot, h, tid, v_cache, v_scale, sm);
    __syncthreads();

    // RMSNorm K per head
    float kv_val = kp[tid];
    float sq = warp_sum(kv_val * kv_val);
    if ((tid & 31) == 0) sm[tid >> 5] = sq;
    __syncthreads();
    __shared__ float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrtf((sm[0] + sm[1] + sm[2] + sm[3]) / float(HEAD_DIM) + RMS_EPS);
    }
    __syncthreads();

    __shared__ float kn[HEAD_DIM];
    kn[tid] = kv_val * inv_rms * __bfloat162float(k_norm[tid]);
    __syncthreads();

    // NeoX RoPE: pair (i, i + 64) for i in 0..63
    if (tid < 64) {
        float freq = 1.0f / powf(MTP_ROPE_THETA, float(2 * tid) / float(HEAD_DIM));
        float ang = float(pos) * freq;
        float c = cosf(ang), s = sinf(ang);
        float x0 = kn[tid], x1 = kn[tid + 64];
        kn[tid]      = x0 * c - x1 * s;
        kn[tid + 64] = x0 * s + x1 * c;
    }
    __syncthreads();
    store_kv_head_128<KV_FMT>(kn[tid], slot, h, tid, k_cache, k_scale, sm);
}

// Initialize DSpark 8-row block: row 0 = embed_norm(anchor), rows 1..7 = mask_embed
__global__ void dspark_init_block_kernel(
    const __nv_bfloat16* __restrict__ embed,
    const __nv_bfloat16* __restrict__ embed_norm,
    const __nv_bfloat16* __restrict__ mask_embed,
    const int32_t* __restrict__ d_anchor,
    float* __restrict__ block_x) { // [8, HIDDEN]
    int r = blockIdx.x; // 0..7
    int tid = threadIdx.x; // 256
    float* dst = block_x + (size_t)r * HIDDEN;
    if (r == 0) {
        int tok = *d_anchor;
        const __nv_bfloat16* row = embed + (size_t)tok * HIDDEN;
        float sum_sq = 0.0f;
        for (int d = tid; d < HIDDEN; d += 256) {
            float v = __bfloat162float(row[d]);
            sum_sq = fmaf(v, v, sum_sq);
        }
        sum_sq = warp_sum(sum_sq);
        __shared__ float sm[8];
        if ((tid & 31) == 0) sm[tid >> 5] = sum_sq;
        __syncthreads();
        __shared__ float inv_rms;
        if (tid == 0) {
            float tot = 0.0f;
            for (int i = 0; i < 8; ++i) tot += sm[i];
            inv_rms = rsqrtf(tot / float(HIDDEN) + RMS_EPS);
        }
        __syncthreads();
        float s = inv_rms;
        for (int d = tid; d < HIDDEN; d += 256) {
            dst[d] = __bfloat162float(row[d]) * s * __bfloat162float(embed_norm[d]);
        }
    } else {
        for (int d = tid; d < HIDDEN; d += 256) {
            dst[d] = __bfloat162float(mask_embed[d]);
        }
    }
}

// DSpark Q/K RMSNorm + NeoX RoPE on the 8-row block (positions p..p+7)
// qkv layout: [8, 5120] (Q: 0..3071, K: 3072..4095, V: 4096..5119)
__global__ void dspark_block_qk_rope_kernel(
    float* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ q_norm,
    const __nv_bfloat16* __restrict__ k_norm,
    const int32_t* __restrict__ d_pos) {
    int r = blockIdx.x; // 0..7
    int h = blockIdx.y; // 0..23: Q, 24..31: K
    int tid = threadIdx.x; // 0..127
    int pos = (*d_pos) + r;
    constexpr int QKV_STRIDE = Q_DIM + 2 * KV_DIM;

    float* vec = (h < 24)
        ? (qkv + (size_t)r * QKV_STRIDE + h * HEAD_DIM)
        : (qkv + (size_t)r * QKV_STRIDE + Q_DIM + (h - 24) * HEAD_DIM);
    const __nv_bfloat16* nw = (h < 24) ? q_norm : k_norm;

    float val = vec[tid];
    float sq = warp_sum(val * val);
    __shared__ float sm[4];
    if ((tid & 31) == 0) sm[tid >> 5] = sq;
    __syncthreads();
    __shared__ float inv_rms;
    if (tid == 0) inv_rms = rsqrtf((sm[0] + sm[1] + sm[2] + sm[3]) / float(HEAD_DIM) + RMS_EPS);
    __syncthreads();

    __shared__ float vn[HEAD_DIM];
    vn[tid] = val * inv_rms * __bfloat162float(nw[tid]);
    __syncthreads();

    if (tid < 64) {
        float freq = 1.0f / powf(MTP_ROPE_THETA, float(2 * tid) / float(HEAD_DIM));
        float ang = float(pos) * freq;
        float c = cosf(ang), s = sinf(ang);
        float x0 = vn[tid], x1 = vn[tid + 64];
        vec[tid]      = x0 * c - x1 * s;
        vec[tid + 64] = x0 * s + x1 * c;
    }
}

// Fast RMSNorm producing __half directly for DSpark's NVFP4 linear_h layers!
__global__ void dspark_rmsnorm_h_kernel(
    const float* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    float* __restrict__ out_f,
    __half* __restrict__ out_h) {
    int r = blockIdx.x;
    int tid = threadIdx.x;
    const float* xb = x + (size_t)r * HIDDEN;
    float sq = 0.0f;
    for (int d = tid; d < HIDDEN; d += 256) { float v = xb[d]; sq = fmaf(v, v, sq); }
    sq = warp_sum(sq);
    __shared__ float sm[8];
    if ((tid & 31) == 0) sm[tid >> 5] = sq;
    __syncthreads();
    __shared__ float inv_rms;
    if (tid == 0) {
        float tot = 0;
        for (int i = 0; i < 8; ++i) tot += sm[i];
        inv_rms = rsqrtf(tot / float(HIDDEN) + RMS_EPS);
    }
    __syncthreads();
    float ir = inv_rms;
    float* of = out_f ? (out_f + (size_t)r * HIDDEN) : nullptr;
    __half* oh = out_h + (size_t)r * HIDDEN;
    for (int d = tid; d < HIDDEN; d += 256) {
        float v = xb[d] * ir * __bfloat162float(w[d]);
        oh[d] = __float2half(v);
        if (of) of[d] = v;
    }
}

// 256-CTA Blackwell Tensor Core (m16n8k16) Split-K FlashDecoding for DSpark Global Layer 1!
// Reads each global context KV token ONCE across all 24 DSpark queries (24x less BW, ~350 TFLOPS TC!).
template <int KV_FMT>
__global__ void __launch_bounds__(128)
dspark_global_attn_bulk_splitk_kernel(
    const float* __restrict__ qkv, // [8, 5120]
    const void* __restrict__ ck,
    const void* __restrict__ cv,
    const __half* __restrict__ cks,
    const __half* __restrict__ cvs,
    int cap,
    const int32_t* __restrict__ d_pos,
    float* __restrict__ split_buf) {
    int p = *d_pos;
    if (p <= 0) return;

    int first = (p > cap) ? (p - cap) : 0;
    int total_len = p - first;
    int split = blockIdx.y;
    int chunk = (total_len + KV_SPLIT_K - 1) / KV_SPLIT_K;
    int t_start = first + split * chunk;
    int t_end = min(p, t_start + chunk);

    constexpr int QKV_STRIDE = Q_DIM + 2 * KV_DIM;
    run_tc_bulk_splitk<KV_FMT, 24, QKV_STRIDE, 128>(
        qkv, ck, cv, cks, cvs, cap, blockIdx.x, split, t_start, t_end, split_buf);
}

// DSpark Block Attention writing __half directly for w_o linear_h!
// Local layers (window > 0): BF16 SWA context + 8 bidirectional block rows.
// Global layer (window == 0): 8 bidirectional block rows + exact merge of 24 bulk Split-K partials!
__global__ void __launch_bounds__(256)
dspark_block_attn_kernel(
    const float* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ ck,
    const __nv_bfloat16* __restrict__ cv,
    const float* __restrict__ split_buf,
    int cap, int window,
    const int32_t* __restrict__ d_pos,
    float* __restrict__ out,
    __half* __restrict__ out_h) {
    int h = blockIdx.x;
    int r = blockIdx.y;
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int kvh = h / GQA_RATIO;
    constexpr int QKV_STRIDE = Q_DIM + 2 * KV_DIM;

    int p = *d_pos;
    bool has_bulk = (window == 0 && split_buf != nullptr && p > 0);
    int first = (window > 0 && p > window) ? (p - window) : 0;

    int d_off = lane * 4;
    const float* qp = qkv + (size_t)r * QKV_STRIDE + h * HEAD_DIM;
    float4 qv = *reinterpret_cast<const float4*>(qp + d_off);
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float m_val = -1e30f, d_val = 0.0f;

    // 1. Context KV [first .. p-1] (only for local BF16 layers; global layer uses 192-CTA Split-K!)
    if (!has_bulk && window > 0) {
        for (int t = first + warp; t < p; t += 8) {
            int slot = t % cap;
            uint2 k4 = __ldg(reinterpret_cast<const uint2*>(ck + (size_t)slot * KV_DIM + kvh * HEAD_DIM + d_off));
            uint2 v4 = __ldg(reinterpret_cast<const uint2*>(cv + (size_t)slot * KV_DIM + kvh * HEAD_DIM + d_off));
            const __nv_bfloat162* kb = reinterpret_cast<const __nv_bfloat162*>(&k4);
            const __nv_bfloat162* vb = reinterpret_cast<const __nv_bfloat162*>(&v4);
            float2 kf0 = __bfloat1622float2(kb[0]), kf1 = __bfloat1622float2(kb[1]);
            float2 vf0 = __bfloat1622float2(vb[0]), vf1 = __bfloat1622float2(vb[1]);
            float dot = fmaf(qv.x, kf0.x, fmaf(qv.y, kf0.y, fmaf(qv.z, kf1.x, qv.w * kf1.y)));
            float score = warp_sum(dot) * INV_SQRT_HEAD_DIM;
            float m_new = fmaxf(m_val, score);
            float alpha = __expf(m_val - m_new);
            float pw = __expf(score - m_new);
            d_val = d_val * alpha + pw;
            acc0 = fmaf(pw, vf0.x, acc0 * alpha);
            acc1 = fmaf(pw, vf0.y, acc1 * alpha);
            acc2 = fmaf(pw, vf1.x, acc2 * alpha);
            acc3 = fmaf(pw, vf1.y, acc3 * alpha);
            m_val = m_new;
        }
    }

    // 2. Bidirectional block KV (all 8 rows: warp `w` handles block row `w`!)
    {
        float4 bkv = *reinterpret_cast<const float4*>(qkv + (size_t)warp * QKV_STRIDE + Q_DIM + kvh * HEAD_DIM + d_off);
        float4 bvv = *reinterpret_cast<const float4*>(qkv + (size_t)warp * QKV_STRIDE + Q_DIM + KV_DIM + kvh * HEAD_DIM + d_off);
        float dot = fmaf(qv.x, bkv.x, fmaf(qv.y, bkv.y, fmaf(qv.z, bkv.z, qv.w * bkv.w)));
        float score = warp_sum(dot) * INV_SQRT_HEAD_DIM;
        float m_new = fmaxf(m_val, score);
        float alpha = __expf(m_val - m_new);
        float pw = __expf(score - m_new);
        d_val = d_val * alpha + pw;
        acc0 = fmaf(pw, bvv.x, acc0 * alpha);
        acc1 = fmaf(pw, bvv.y, acc1 * alpha);
        acc2 = fmaf(pw, bvv.z, acc2 * alpha);
        acc3 = fmaf(pw, bvv.w, acc3 * alpha);
        m_val = m_new;
    }

    __shared__ float sm_m[8], sm_d[8], sm_acc[8][HEAD_DIM];
    if (lane == 0) { sm_m[warp] = m_val; sm_d[warp] = d_val; }
    sm_acc[warp][d_off + 0] = acc0;
    sm_acc[warp][d_off + 1] = acc1;
    sm_acc[warp][d_off + 2] = acc2;
    sm_acc[warp][d_off + 3] = acc3;
    __syncthreads();

    if (tid < HEAD_DIM) {
        float g_m = sm_m[0];
        #pragma unroll
        for (int w = 1; w < 8; ++w) g_m = fmaxf(g_m, sm_m[w]);

        int base_hr = (h * MAX_VERIFY_B + r) * KV_TOTAL_SPLITS;
        const float* s_m_in = split_buf + SPLIT_M_OFF;
        const float* s_d_in = split_buf + SPLIT_D_OFF;
        const float* s_o_in = split_buf + SPLIT_O_OFF;
        if (has_bulk) {
            #pragma unroll
            for (int s = 0; s < KV_SPLIT_K; ++s) {
                g_m = fmaxf(g_m, s_m_in[base_hr + s]);
            }
        }

        float g_d = 0.0f, g_o = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            if (sm_d[w] > 0.0f) {
                float sc = __expf(sm_m[w] - g_m);
                g_d += sm_d[w] * sc;
                g_o += sm_acc[w][tid] * sc;
            }
        }
        if (has_bulk) {
            #pragma unroll
            for (int s = 0; s < KV_SPLIT_K; ++s) {
                float sd = s_d_in[base_hr + s];
                if (sd > 0.0f) {
                    float sc = __expf(s_m_in[base_hr + s] - g_m);
                    g_d += sd * sc;
                    g_o += s_o_in[(size_t)(base_hr + s) * HEAD_DIM + tid] * sc;
                }
            }
        }
        float val = g_o / g_d;
        size_t oidx = (size_t)r * Q_DIM + h * HEAD_DIM + tid;
        if (out) out[oidx] = val;
        if (out_h) out_h[oidx] = __float2half(val);
    }
}

// Standard SwiGLU for DSpark MLP writing __half directly for w_down linear_h!
__global__ void dspark_swiglu_kernel(const float* __restrict__ gu, float* __restrict__ out, __half* __restrict__ out_h, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * MTP_INTERMEDIATE;
    if (idx >= total) return;
    int b = idx / MTP_INTERMEDIATE;
    int j = idx % MTP_INTERMEDIATE;
    float g = gu[(size_t)b * (2 * MTP_INTERMEDIATE) + j];
    float u = gu[(size_t)b * (2 * MTP_INTERMEDIATE) + MTP_INTERMEDIATE + j];
    float v = (g * fast_sigmoid(g)) * u;
    if (out) out[idx] = v;
    if (out_h) out_h[idx] = __float2half(v);
}

// Coalesced 24-block Markov RNN projection:
// raw[768] = rnn_bias + c_h[i] + c_f + rnn_w_sp @ [state; w1[prev_tok]]
// Grid: 24 blocks, Block: 128 threads (4 warps/block -> 8 rows/warp = 768 rows, 100% coalesced uint4!)
__global__ void __launch_bounds__(128)
markov_rnn_proj_kernel(
    const __nv_bfloat16* __restrict__ rnn_w_sp,  // [768, 512]
    const __nv_bfloat16* __restrict__ rnn_bias,  // [768]
    const __nv_bfloat16* __restrict__ markov_w1, // [201024, 256]
    const float* __restrict__ c_h_i,             // [768]
    const float* __restrict__ c_f,               // [768]
    const int32_t* __restrict__ prev_tok_ptr,    // [1]
    const float* __restrict__ state,             // [256]
    float* __restrict__ out_raw) {               // [768]
    __shared__ float sp[512];
    int tid = threadIdx.x;
    int prev_tok = *prev_tok_ptr;
    const __nv_bfloat16* w1_row = markov_w1 + (size_t)prev_tok * MTP_MARKOV_RANK;
    sp[tid]       = state[tid];
    sp[128 + tid] = state[128 + tid];
    sp[256 + tid] = __bfloat162float(w1_row[tid]);
    sp[384 + tid] = __bfloat162float(w1_row[128 + tid]);
    __syncthreads();

    int warp = tid >> 5;
    int lane = tid & 31;
    int base_r = blockIdx.x * 32 + warp * 8;
    #pragma unroll
    for (int rr = 0; rr < 8; ++rr) {
        int r = base_r + rr;
        const uint4* wr = reinterpret_cast<const uint4*>(rnn_w_sp + (size_t)r * 512);
        float acc = 0.0f;
        #pragma unroll
        for (int iter = 0; iter < 2; ++iter) {
            int v = lane + iter * 32; // 0..63 (64 * 8 = 512)
            uint4 w8 = __ldg(wr + v);
            const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&w8);
            float2 wf0 = __bfloat1622float2(wb[0]), wf1 = __bfloat1622float2(wb[1]);
            float2 wf2 = __bfloat1622float2(wb[2]), wf3 = __bfloat1622float2(wb[3]);
            const float4* xf = reinterpret_cast<const float4*>(sp + (v << 3));
            float4 x0 = xf[0], x1 = xf[1];
            acc = fmaf(wf0.x, x0.x, acc); acc = fmaf(wf0.y, x0.y, acc);
            acc = fmaf(wf1.x, x0.z, acc); acc = fmaf(wf1.y, x0.w, acc);
            acc = fmaf(wf2.x, x1.x, acc); acc = fmaf(wf2.y, x1.y, acc);
            acc = fmaf(wf3.x, x1.z, acc); acc = fmaf(wf3.y, x1.w, acc);
        }
        float val = warp_sum(acc);
        if (lane == 0) {
            out_raw[r] = val + __bfloat162float(rnn_bias[r]) + c_h_i[r] + c_f[r];
        }
    }
}

// Persistent 208-block SMEM NVFP4 base_logits[K, Vd] = unembed_draft @ normed_h + fold_b (8*26 SMs -> 0% wave tail!)
constexpr int DSPARK_UNEMBED_BLOCKS = 208;

template <int K_CONST>
__global__ void __launch_bounds__(256)
dspark_base_logits_nvfp4_kernel(
    const uint8_t* __restrict__ w_data,
    const uint8_t* __restrict__ w_scales,
    float gscale,
    const __half* __restrict__ normed_h, // [K_CONST, HIDDEN]
    const float* __restrict__ fold_b,    // [V]
    float* __restrict__ out_base,        // [K_CONST, V]
    int V) {
    __shared__ __half s_xh[K_CONST * HIDDEN];
    int tid = threadIdx.x;
    int total_x = K_CONST * HIDDEN;
    for (int i = tid; i < total_x; i += 256) s_xh[i] = normed_h[i];
    __syncthreads();

    int hw = tid >> 4;   // 0..15 (16 rows/CTA!)
    int lane = tid & 15; // 0..15 (9 EXACT iters!)
    int global_r = blockIdx.x * 16 + hw;
    int total_r = gridDim.x * 16;

    for (int row = global_r; row < V; row += total_r) {
        const uint2* wr = reinterpret_cast<const uint2*>(w_data + (size_t)row * (HIDDEN >> 1));
        const uint8_t* sr = w_scales + (size_t)row * (HIDDEN >> 4);
        float acc[K_CONST];
        #pragma unroll
        for (int k = 0; k < K_CONST; ++k) acc[k] = 0.0f;

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
            for (int k = 0; k < K_CONST; ++k) {
                const __half2* xb = reinterpret_cast<const __half2*>(s_xh + k * HIDDEN + koff);
                __half2 p0 = __hmul2(wh[0], xb[0]), p1 = __hmul2(wh[1], xb[1]);
                p0 = __hfma2(wh[2], xb[2], p0); p1 = __hfma2(wh[3], xb[3], p1);
                p0 = __hfma2(wh[4], xb[4], p0); p1 = __hfma2(wh[5], xb[5], p1);
                p0 = __hfma2(wh[6], xb[6], p0); p1 = __hfma2(wh[7], xb[7], p1);
                float2 pf = __half22float2(__hadd2(p0, p1));
                acc[k] = fmaf(pf.x + pf.y, sc, acc[k]);
            }
        }
        float fb = (lane == 0 && fold_b) ? fold_b[row] : 0.0f;
        #pragma unroll
        for (int k = 0; k < K_CONST; ++k) {
            float val = half_warp_sum(acc[k]) * gscale;
            if (lane == 0) out_base[(size_t)k * V + row] = val + fb;
        }
    }
}

constexpr int MARKOV_BLOCKS = 104; // 4 blocks/SM on RTX 5060 (26 SMs -> 0% wave tail!)

template <int KIND> // 0=BF16, 2=NVFP4
__global__ void __launch_bounds__(256)
markov_fold_argmax_stage1_kernel(
    const void* __restrict__ fold_data,
    const uint8_t* __restrict__ fold_scales,
    float fold_gscale,
    const float* __restrict__ base_logits, // [V] (unembed @ h_{i+1} [+ fold_b if fold_b==nullptr])
    const float* __restrict__ fold_b,      // optional [V]
    const float* __restrict__ raw_rnn,     // [768]: gate[0..255], cand[256..511], lat[512..767]
    float* __restrict__ state,             // [256] updated by block 0!
    float* __restrict__ blk_val,
    int32_t* __restrict__ blk_idx,
    int V) {
    __shared__ __half s_lh[256];
    __shared__ float s_lf[256];
    int tid = threadIdx.x;
    float lat_v = fast_tanh(raw_rnn[512 + tid]);
    s_lf[tid] = lat_v;
    s_lh[tid] = __float2half(lat_v);
    if (blockIdx.x == 0) {
        float gate = fast_sigmoid(raw_rnn[tid]);
        float cand = fast_tanh(raw_rnn[256 + tid]);
        state[tid] = gate * state[tid] + (1.0f - gate) * cand;
    }
    __syncthreads();

    constexpr int TPR = (KIND == 2) ? 16 : 32;
    constexpr int RPB = 256 / TPR; // 16 rows/block for NVFP4 (0% idle!), 8 for BF16
    int r_in_blk = (TPR == 16) ? (tid >> 4) : (tid >> 5);
    int lane = tid & (TPR - 1);
    int gr = blockIdx.x * RPB + r_in_blk;
    int tr = gridDim.x * RPB;

    float bv = -1e30f;
    int32_t bi = 0;

    for (int row = gr; row < V; row += tr) {
        float acc = 0.0f;
        if constexpr (KIND == 2) {
            // HALF-WARP: K = 256 -> 16 blocks of 16 -> 1 EXACT block per lane, 100% active!!
            const uint2* wr = reinterpret_cast<const uint2*>(
                static_cast<const uint8_t*>(fold_data) + (size_t)row * 128);
            const uint8_t* sr = fold_scales + (size_t)row * 16;
            uint2 w8 = __ldg(wr + lane);
            float sc = cvt_e4m3_to_float(__ldg(sr + lane));
            const __half2* xh = reinterpret_cast<const __half2*>(s_lh + (lane << 4));
            acc = half_warp_sum(nvfp4_block16_dot(w8, xh) * sc) * fold_gscale;
        } else {
            // BF16: K = 256 -> 32 uint4 vectors (1 per lane!)
            const uint4* wr = reinterpret_cast<const uint4*>(
                static_cast<const __nv_bfloat16*>(fold_data) + (size_t)row * 256);
            uint4 w8 = __ldg(wr + lane);
            const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&w8);
            float2 wf0 = __bfloat1622float2(wb[0]);
            float2 wf1 = __bfloat1622float2(wb[1]);
            float2 wf2 = __bfloat1622float2(wb[2]);
            float2 wf3 = __bfloat1622float2(wb[3]);
            const float4* xf = reinterpret_cast<const float4*>(s_lf + (lane << 3));
            float4 x0 = xf[0], x1 = xf[1];
            acc = fmaf(wf0.x, x0.x, acc); acc = fmaf(wf0.y, x0.y, acc);
            acc = fmaf(wf1.x, x0.z, acc); acc = fmaf(wf1.y, x0.w, acc);
            acc = fmaf(wf2.x, x1.x, acc); acc = fmaf(wf2.y, x1.y, acc);
            acc = fmaf(wf3.x, x1.z, acc); acc = fmaf(wf3.y, x1.w, acc);
            acc = warp_sum(acc);
        }
        if (lane == 0) {
            float total = base_logits[row] + (fold_b ? fold_b[row] : 0.0f) + acc;
            if (total > bv) { bv = total; bi = row; }
        }
    }

    __shared__ float sv[RPB];
    __shared__ int32_t si[RPB];
    if (lane == 0) { sv[r_in_blk] = bv; si[r_in_blk] = bi; }
    __syncthreads();
    if (tid == 0) {
        bv = sv[0]; bi = si[0];
        for (int w = 1; w < RPB; ++w) {
            if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) { bv = sv[w]; bi = si[w]; }
        }
        blk_val[blockIdx.x] = bv;
        blk_idx[blockIdx.x] = bi;
    }
}

__global__ void markov_fold_argmax_stage2_kernel(
    const float* __restrict__ blk_val,
    const int32_t* __restrict__ blk_idx,
    const int32_t* __restrict__ draft_vocab_ids, // optional remap to real vocab ID
    int32_t* __restrict__ out_tok) {
    int tid = threadIdx.x;
    float bv = -1e30f;
    int32_t bi = 0;
    for (int i = tid; i < MARKOV_BLOCKS; i += 256) {
        float v = blk_val[i];
        int32_t idx = blk_idx[i];
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
        *out_tok = draft_vocab_ids ? draft_vocab_ids[bi] : bi;
    }
}

__global__ void copy_f_prev_zero_state_kernel(
    const float* __restrict__ f_ring, const int32_t* __restrict__ d_pos,
    float* __restrict__ out_f, float* __restrict__ state) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < HIDDEN) {
        int prev_p = (*d_pos) - 1;
        out_f[i] = (prev_p >= 0) ? f_ring[(size_t)(prev_p & 15) * HIDDEN + i] : 0.0f;
    }
    if (i < MTP_MARKOV_RANK) state[i] = 0.0f;
}

__global__ void spec_verify_kernel(
    const int32_t* __restrict__ target_preds,
    const int32_t* __restrict__ draft_tokens,
    int B,
    int32_t* __restrict__ out_accepted,
    int32_t* __restrict__ out_num_accepted,
    int32_t* __restrict__ d_pos,
    int32_t* __restrict__ d_next_anchor) {
    if (threadIdx.x != 0) return;
    int n_acc = 0;
    for (int i = 0; i < B - 1; ++i) {
        int32_t tp = target_preds[i];
        out_accepted[n_acc++] = tp;
        if (tp != draft_tokens[i]) {
            *out_num_accepted = n_acc;
            *d_pos += n_acc;
            *d_next_anchor = tp;
            return;
        }
    }
    // All B-1 drafts matched! Emit bonus token target_preds[B-1]!
    int32_t bonus = target_preds[B - 1];
    out_accepted[n_acc++] = bonus;
    *out_num_accepted = n_acc;
    *d_pos += n_acc;
    *d_next_anchor = bonus;
}

} // namespace

void launch_dspark_commit(
    const DsparkWeights& dw, const float* taps,
    void* ctx_k[MTP_LAYERS], void* ctx_v[MTP_LAYERS],
    __half* ctx_ks[MTP_LAYERS], __half* ctx_vs[MTP_LAYERS],
    int ctx_cap[MTP_LAYERS], KVFormat global_kv_fmt,
    float* f_ring, float* scratch_mix, float* scratch_kv,
    int B, const int32_t* d_pos, cudaStream_t stream) {
    dspark_mix_kernel<<<dim3(B, 4), 256, 0, stream>>>(
        taps, B, dw.fuse_alpha,
        dw.layers[0].dflare_norm, dw.layers[1].dflare_norm, dw.layers[2].dflare_norm,
        dw.markov_dflare_norm, scratch_mix, f_ring, d_pos);
    for (int l = 0; l < MTP_LAYERS; ++l) {
        const float* mix_l = scratch_mix + (size_t)l * B * HIDDEN;
        launch_linear(dw.layers[l].w_kv_ctx, mix_l, scratch_kv, B, false, stream);
        KVFormat lfmt = dw.layers[l].is_local ? KVFormat::BF16 : global_kv_fmt;
        if (lfmt == KVFormat::FP8) {
            dspark_store_ctx_kv_kernel<1><<<dim3(B, NUM_KV_HEADS), HEAD_DIM, 0, stream>>>(
                scratch_kv, dw.layers[l].k_norm_ctx, ctx_k[l], ctx_v[l], ctx_ks[l], ctx_vs[l], ctx_cap[l], d_pos);
        } else if (lfmt == KVFormat::Q4) {
            dspark_store_ctx_kv_kernel<2><<<dim3(B, NUM_KV_HEADS), HEAD_DIM, 0, stream>>>(
                scratch_kv, dw.layers[l].k_norm_ctx, ctx_k[l], ctx_v[l], ctx_ks[l], ctx_vs[l], ctx_cap[l], d_pos);
        } else {
            dspark_store_ctx_kv_kernel<0><<<dim3(B, NUM_KV_HEADS), HEAD_DIM, 0, stream>>>(
                scratch_kv, dw.layers[l].k_norm_ctx, ctx_k[l], ctx_v[l], ctx_ks[l], ctx_vs[l], ctx_cap[l], d_pos);
        }
    }
}

void launch_dspark_draft(
    const ModelWeights& mw, const int32_t* d_anchor_token, const int32_t* d_pos, int num_drafts,
    void* ctx_k[MTP_LAYERS], void* ctx_v[MTP_LAYERS],
    const __half* ctx_ks[MTP_LAYERS], const __half* ctx_vs[MTP_LAYERS],
    int ctx_cap[MTP_LAYERS], KVFormat global_kv_fmt, float* split_buf,
    const float* f_ring, float* ws, int32_t* out_draft_tokens, cudaStream_t stream) {
    const DsparkWeights& dw = mw.dspark;

    // Partition preallocated DSpark workspace (in floats):
    // block_x:     8 * 2304 = 18,432
    // normed:      8 * 2304 = 18,432
    // qkv:         8 * 5120 = 40,960
    // attn_out:    8 * 3072 = 24,576
    // gu:          8 * 18432 = 147,456
    // swiglu:      8 * 9216 = 73,728
    // f_prev:      2304
    // c_f:         768
    // c_h:         7 * 768 = 5,376
    // state:       256
    // latent:      256
    // blk_val:     680
    // blk_idx:     680 (as int32)
    // base_logits: 7 * VOCAB_UNPADDED = 1,400,406 floats (~5.6 MB)
    float* block_x     = ws;
    float* normed      = block_x + MTP_BLOCK_SIZE * HIDDEN;
    __half* normed_h   = reinterpret_cast<__half*>(normed + MTP_BLOCK_SIZE * HIDDEN);
    float* qkv         = reinterpret_cast<float*>(normed_h + MTP_BLOCK_SIZE * HIDDEN);
    float* attn_out    = qkv + MTP_BLOCK_SIZE * (Q_DIM + 2 * KV_DIM);
    __half* attn_out_h = reinterpret_cast<__half*>(attn_out + MTP_BLOCK_SIZE * Q_DIM);
    float* gu          = reinterpret_cast<float*>(attn_out_h + MTP_BLOCK_SIZE * Q_DIM);
    float* swiglu      = gu + MTP_BLOCK_SIZE * (2 * MTP_INTERMEDIATE);
    __half* swiglu_h   = reinterpret_cast<__half*>(swiglu + MTP_BLOCK_SIZE * MTP_INTERMEDIATE);
    float* f_prev      = reinterpret_cast<float*>(swiglu_h + MTP_BLOCK_SIZE * MTP_INTERMEDIATE);
    float* c_f         = f_prev + HIDDEN;
    float* c_h         = c_f + 3 * MTP_MARKOV_RANK;
    float* state       = c_h + MTP_MAX_DRAFTS * (3 * MTP_MARKOV_RANK);
    float* raw_rnn     = state + MTP_MARKOV_RANK;
    float* blk_val     = raw_rnn + 3 * MTP_MARKOV_RANK;
    int32_t* blk_idx   = reinterpret_cast<int32_t*>(blk_val + MARKOV_BLOCKS);
    float* base_logits = reinterpret_cast<float*>(blk_idx + MARKOV_BLOCKS);

    bool use_half = (dw.layers[0].w_qkv.kind != WeightKind::BF16);

    // 1. Build 8-row block [anchor, mask x7]
    dspark_init_block_kernel<<<MTP_BLOCK_SIZE, 256, 0, stream>>>(
        mw.embed, mw.embed_norm, dw.mask_embed, d_anchor_token, block_x);

    // 2. Run 3 dense DSpark layers over all 8 rows (ZERO inner-loop F32->F16 conversions!)
    for (int l = 0; l < MTP_LAYERS; ++l) {
        const DsparkLayerWeights& dl = dw.layers[l];
        if (use_half) {
            dspark_rmsnorm_h_kernel<<<MTP_BLOCK_SIZE, 256, 0, stream>>>(block_x, dl.input_norm, nullptr, normed_h);
            launch_linear_h(dl.w_qkv, normed_h, qkv, MTP_BLOCK_SIZE, false, stream);
        } else {
            launch_rmsnorm(block_x, dl.input_norm, normed, MTP_BLOCK_SIZE, HIDDEN, HIDDEN, RMS_EPS, stream);
            launch_linear(dl.w_qkv, normed, qkv, MTP_BLOCK_SIZE, false, stream);
        }
        dspark_block_qk_rope_kernel<<<dim3(MTP_BLOCK_SIZE, NUM_HEADS + NUM_KV_HEADS), HEAD_DIM, 0, stream>>>(
            qkv, dl.q_norm, dl.k_norm, d_pos);
        if (!dl.is_local) {
            dim3 sgrid(NUM_KV_HEADS, KV_SPLIT_K);
            if (global_kv_fmt == KVFormat::FP8) {
                dspark_global_attn_bulk_splitk_kernel<1><<<sgrid, 128, 0, stream>>>(
                    qkv, ctx_k[l], ctx_v[l], ctx_ks[l], ctx_vs[l], ctx_cap[l], d_pos, split_buf);
            } else if (global_kv_fmt == KVFormat::Q4) {
                dspark_global_attn_bulk_splitk_kernel<2><<<sgrid, 128, 0, stream>>>(
                    qkv, ctx_k[l], ctx_v[l], ctx_ks[l], ctx_vs[l], ctx_cap[l], d_pos, split_buf);
            } else {
                dspark_global_attn_bulk_splitk_kernel<0><<<sgrid, 128, 0, stream>>>(
                    qkv, ctx_k[l], ctx_v[l], ctx_ks[l], ctx_vs[l], ctx_cap[l], d_pos, split_buf);
            }
        }
        dspark_block_attn_kernel<<<dim3(NUM_HEADS, MTP_BLOCK_SIZE), 256, 0, stream>>>(
            qkv, static_cast<const __nv_bfloat16*>(ctx_k[l]), static_cast<const __nv_bfloat16*>(ctx_v[l]),
            dl.is_local ? nullptr : split_buf, ctx_cap[l], dl.is_local ? SWA_WINDOW : 0, d_pos,
            use_half ? nullptr : attn_out, use_half ? attn_out_h : nullptr);
        if (use_half) launch_linear_h(dl.w_o, attn_out_h, block_x, MTP_BLOCK_SIZE, true, stream);
        else          launch_linear(dl.w_o, attn_out, block_x, MTP_BLOCK_SIZE, true, stream);

        if (use_half) {
            dspark_rmsnorm_h_kernel<<<MTP_BLOCK_SIZE, 256, 0, stream>>>(block_x, dl.post_norm, nullptr, normed_h);
            launch_linear_h(dl.w_gu, normed_h, gu, MTP_BLOCK_SIZE, false, stream);
            dspark_swiglu_kernel<<<(MTP_BLOCK_SIZE * MTP_INTERMEDIATE + 255) / 256, 256, 0, stream>>>(
                gu, nullptr, swiglu_h, MTP_BLOCK_SIZE);
            launch_linear_h(dl.w_down, swiglu_h, block_x, MTP_BLOCK_SIZE, true, stream);
        } else {
            launch_rmsnorm(block_x, dl.post_norm, normed, MTP_BLOCK_SIZE, HIDDEN, HIDDEN, RMS_EPS, stream);
            launch_linear(dl.w_gu, normed, gu, MTP_BLOCK_SIZE, false, stream);
            dspark_swiglu_kernel<<<(MTP_BLOCK_SIZE * MTP_INTERMEDIATE + 255) / 256, 256, 0, stream>>>(
                gu, swiglu, nullptr, MTP_BLOCK_SIZE);
            launch_linear(dl.w_down, swiglu, block_x, MTP_BLOCK_SIZE, true, stream);
        }
    }

    // 3. Final norm on rows 1..num_drafts -> normed (F32 for rnn_w_h) + normed_h (F16 for unembed_draft!)
    dspark_rmsnorm_h_kernel<<<num_drafts, 256, 0, stream>>>(block_x + HIDDEN, dw.norm_w, normed, normed_h);

    // 4. Precompute Markov inputs + zero RNN state in ONE kernel (no cudaMemsetAsync!)
    copy_f_prev_zero_state_kernel<<<(HIDDEN + 255) / 256, 256, 0, stream>>>(f_ring, d_pos, f_prev, state);
    launch_bf16_linear(dw.rnn_w_f, f_prev, c_f, 1, 3 * MTP_MARKOV_RANK, HIDDEN, stream);
    launch_bf16_linear(dw.rnn_w_h, normed, c_h, num_drafts, 3 * MTP_MARKOV_RANK, HIDDEN, stream);

    // 5. Precompute base_logits[0..num_drafts-1, 0..Vd-1] (+ fold_b inline for NVFP4!)
    int Vd = dw.draft_vocab_size;
    bool fold_b_in_base = false;
    if (dw.unembed_draft.kind == WeightKind::NVFP4) {
        fold_b_in_base = true;
        const auto* wd = static_cast<const uint8_t*>(dw.unembed_draft.data);
        const auto* ws_sc = dw.unembed_draft.scales;
        float gs = dw.unembed_draft.global_scale;
        switch (num_drafts) {
            case 1: dspark_base_logits_nvfp4_kernel<1><<<DSPARK_UNEMBED_BLOCKS, 256, 0, stream>>>(wd, ws_sc, gs, normed_h, dw.fold_b, base_logits, Vd); break;
            case 2: dspark_base_logits_nvfp4_kernel<2><<<DSPARK_UNEMBED_BLOCKS, 256, 0, stream>>>(wd, ws_sc, gs, normed_h, dw.fold_b, base_logits, Vd); break;
            case 3: dspark_base_logits_nvfp4_kernel<3><<<DSPARK_UNEMBED_BLOCKS, 256, 0, stream>>>(wd, ws_sc, gs, normed_h, dw.fold_b, base_logits, Vd); break;
            case 4: dspark_base_logits_nvfp4_kernel<4><<<DSPARK_UNEMBED_BLOCKS, 256, 0, stream>>>(wd, ws_sc, gs, normed_h, dw.fold_b, base_logits, Vd); break;
            case 5: dspark_base_logits_nvfp4_kernel<5><<<DSPARK_UNEMBED_BLOCKS, 256, 0, stream>>>(wd, ws_sc, gs, normed_h, dw.fold_b, base_logits, Vd); break;
            case 6: dspark_base_logits_nvfp4_kernel<6><<<DSPARK_UNEMBED_BLOCKS, 256, 0, stream>>>(wd, ws_sc, gs, normed_h, dw.fold_b, base_logits, Vd); break;
            case 7: dspark_base_logits_nvfp4_kernel<7><<<DSPARK_UNEMBED_BLOCKS, 256, 0, stream>>>(wd, ws_sc, gs, normed_h, dw.fold_b, base_logits, Vd); break;
        }
    } else if (dw.unembed_draft.kind != WeightKind::BF16) {
        launch_linear_h(dw.unembed_draft, normed_h, base_logits, num_drafts, false, stream);
    } else {
        launch_linear(dw.unembed_draft, normed, base_logits, num_drafts, false, stream);
    }

    // 6. Run num_drafts sequential Markov steps (24-block coalesced RNN + half-warp fused argmax!)
    const float* fb_arg = fold_b_in_base ? nullptr : dw.fold_b;
    for (int i = 0; i < num_drafts; ++i) {
        const int32_t* prev_ptr = (i == 0) ? d_anchor_token : (out_draft_tokens + i - 1);
        markov_rnn_proj_kernel<<<24, 128, 0, stream>>>(
            dw.rnn_w_sp, dw.rnn_bias, dw.markov_w1,
            c_h + i * (3 * MTP_MARKOV_RANK), c_f, prev_ptr, state, raw_rnn);

        const float* bl_i = base_logits + (size_t)i * Vd;
        if (dw.fold_w.kind == WeightKind::NVFP4) {
            markov_fold_argmax_stage1_kernel<2><<<MARKOV_BLOCKS, 256, 0, stream>>>(
                dw.fold_w.data, dw.fold_w.scales, dw.fold_w.global_scale,
                bl_i, fb_arg, raw_rnn, state, blk_val, blk_idx, Vd);
        } else {
            markov_fold_argmax_stage1_kernel<0><<<MARKOV_BLOCKS, 256, 0, stream>>>(
                dw.fold_w.data, nullptr, 1.0f,
                bl_i, fb_arg, raw_rnn, state, blk_val, blk_idx, Vd);
        }
        markov_fold_argmax_stage2_kernel<<<1, 256, 0, stream>>>(blk_val, blk_idx, dw.draft_vocab_ids, out_draft_tokens + i);
    }
}

void launch_spec_verify(
    const int32_t* target_preds, const int32_t* draft_tokens, int B,
    int32_t* out_accepted_tokens, int32_t* out_num_accepted,
    int32_t* d_pos, int32_t* d_next_anchor, cudaStream_t stream) {
    spec_verify_kernel<<<1, 1, 0, stream>>>(
        target_preds, draft_tokens, B, out_accepted_tokens, out_num_accepted, d_pos, d_next_anchor);
}

} // namespace squid
