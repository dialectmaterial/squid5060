// DFlash2 block-diffusion drafter for Inkling 2 Mini (incoai/Inkling-2-Mini-DFlash2).
//
// Reference semantics: z-lab/dflash `DFlash2DraftModel` (dflash/model.py) plus the sglang
// Inkling compatibility patch shipped with the checkpoint (selector softcap "after_correction",
// candidate projection limited to the unpadded vocabulary).
//
// The drafter reads the same four target features that DSpark taps: post-residual visits 5, 23, 41
// and the post-final-norm state (taps slot 3). It keeps its own context KV ring (sliding window 2048,
// non-causal), drafts a block of 8 (anchor + 7 mask tokens) in one pass and selects one path through
// the top-16 candidates of every position with a low-rank transition score. Verification is the
// engine's existing path: dense q rows [7, VOCAB_UNPADDED] with zeros outside the 16 candidates.
//
// Everything is self-contained: weights, ring and scratch live in one context object with its own
// VRAM, allocated before the expert cache takes the remaining free memory.

#include "kernels.hpp"
#include "kernels_common.cuh"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

extern "C" float squid_pack_bf16_to_nvfp4(
    const void* src_bf16_dev, void* dst_data, uint8_t* dst_scales, float* d_amax_scratch, int32_t rows, int32_t cols);
extern "C" void squid_pack_bf16_to_fp8(
    const void* src_bf16_dev, void* dst_data, float* dst_row_scales, int32_t rows, int32_t cols);

namespace squid {
namespace dflash {

constexpr int DH = 2304;           // hidden size (same as target)
constexpr int NL = 5;              // draft layers
constexpr int NQ = 32;             // query heads
constexpr int NKV = 8;             // kv heads
constexpr int HD = 128;            // head dim
constexpr int GQA = NQ / NKV;      // 4
constexpr int QD = NQ * HD;        // 4096
constexpr int KVD = NKV * HD;      // 1024
constexpr int FF = 6912;           // MLP intermediate
constexpr int CG = 16;             // conv group size
constexpr int NG = DH / CG;        // 144 groups
constexpr int CK = 2;              // conv kernel size
constexpr int CDYN = 2 * CK * NG;  // 576 dynamic kernel outputs
constexpr int BLK = 8;             // block size
constexpr int ND = BLK - 1;        // 7 drafts
constexpr int TOPK = 16;
constexpr int RANK = 256;
constexpr int WIN = 2048;          // sliding window
constexpr int RING = WIN + BLK;    // ring slots
constexpr int NTAP = 4;
constexpr int FC_IN = NTAP * DH;   // 9216
constexpr int KV_ALL = NL * 2 * KVD; // 10240 stacked ctx k/v rows
constexpr int V = VOCAB_UNPADDED;  // 200058 candidate rows
constexpr int MASK_TOKEN = 200056;
constexpr float SOFTCAP = 30.0f;
constexpr double ROPE_THETA = 10000.0;
constexpr float EPS = 1e-6f;
constexpr int ATT_TILE = 32;       // keys per smem tile
constexpr int ATT_CHUNK = 64;      // keys per split block
constexpr int ATT_SPLITS = (WIN - 1 + BLK + ATT_CHUNK - 1) / ATT_CHUNK; // 33
constexpr int ATT_Q = BLK * GQA;   // 32 queries per kv head
constexpr int TOPK_PARTS = 64;
constexpr int PREFILL_SUB = 256;   // prefill commit sub-batch
constexpr uint64_t SEED_SALT = 0xD1F1A5A2C0FFEE11ULL;

struct Layer {
    const __nv_bfloat16* in_norm = nullptr;
    const __nv_bfloat16* post_norm = nullptr;
    const __nv_bfloat16* q_norm = nullptr;
    const __nv_bfloat16* k_norm = nullptr;
    const __nv_bfloat16* attn_base = nullptr; // [2][2][DH]
    const __nv_bfloat16* mlp_base = nullptr;  // [2][2][DH]
    LinearWeight w_attn_conv;                  // [576, DH]
    LinearWeight w_mlp_conv;                   // [576, DH]
    LinearWeight w_q;                          // [4096, DH]
    LinearWeight w_kv;                         // [2048, DH] view into ctx.w_kv_all (bf16)
    LinearWeight w_o;                          // [DH, 4096]
    LinearWeight w_gu;                         // [2*FF, DH] gate rows then up rows
    LinearWeight w_down;                       // [DH, FF]
};

struct KNormSet {
    const __nv_bfloat16* w[NL];
};

struct Ctx {
    int quant = 0; // 0 bf16, 1 fp8, 2 nvfp4 (draft layers and decode-commit matrices)
    bool use_embed_norm = false;
    Layer layers[NL];
    LinearWeight w_fc;      // [DH, 9216] bf16
    LinearWeight w_kv_all;  // [10240, DH] bf16 (prefill commit, cuBLAS)
    LinearWeight w_fc_q;    // decode-commit copies in the chosen format
    LinearWeight w_kv_all_q;
    LinearWeight w_hproj;   // [256, DH] bf16
    const __nv_bfloat16* hidden_norm = nullptr;
    const __nv_bfloat16* final_norm = nullptr;
    const __nv_bfloat16* pred_cb = nullptr; // [VOCAB_PADDED, 256] (device pointer, may be host-mapped)
    const __nv_bfloat16* succ_cb = nullptr;
    void* pred_host = nullptr;
    void* succ_host = nullptr;
    KNormSet knorms;

    // ring: K [NL][RING][KVD] then V [NL][RING][KVD], bf16, one contiguous buffer
    __nv_bfloat16* ring = nullptr;
    size_t ring_bytes = 0;

    // draft scratch (f32)
    float* x = nullptr;      // [8, DH] residual stream
    float* nrm = nullptr;    // [8, DH]
    float* cv = nullptr;     // [8, DH]
    float* o = nullptr;      // [8, DH]
    float* dyn = nullptr;    // [8, 576]
    float* q = nullptr;      // [8, 4096]
    float* kvb = nullptr;    // [8, 2048]
    float* qn = nullptr;     // [8, 32, 128]
    float* kn = nullptr;     // [8, 8, 128]
    float* vn = nullptr;     // [8, 8, 128]
    float* part = nullptr;   // [SPLITS][NKV][ATT_Q][HD + 2]
    float* attn = nullptr;   // [8, 4096]
    float* gu = nullptr;     // [8, 2*FF]
    float* act = nullptr;    // [8, FF]
    float* logits = nullptr; // [7, V]
    float* top_v1 = nullptr; // [7, PARTS, 16]
    int32_t* top_i1 = nullptr;
    float* cand_v = nullptr; // [7, 16]
    int32_t* cand_i = nullptr;
    int32_t* prev_cand = nullptr; // [7, 16] last written dense q entries
    float* hp = nullptr;     // [7, 256]
    float* probs = nullptr;  // [7, V] dense q

    // commit scratch
    float* c_cat = nullptr;  // [8, 9216]
    float* c_ctx = nullptr;  // [8, DH]
    float* c_ctxn = nullptr; // [8, DH]
    float* c_kv = nullptr;   // [8, 10240]
    __nv_bfloat16* p_cat_h = nullptr; // [SUB, 9216]
    float* p_ctx = nullptr;           // [SUB, DH]
    __nv_bfloat16* p_ctxn_h = nullptr;// [SUB, DH]
    float* p_kv = nullptr;            // [SUB, 10240]

    cublasHandle_t blas = nullptr;
    void* vram = nullptr;
    size_t vram_bytes = 0;
    size_t vram_used = 0;
    size_t weight_bytes = 0;
};

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

__device__ __forceinline__ float block_sum_256(float v, float* red) {
    v = warp_sum(v);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = (threadIdx.x < (blockDim.x >> 5)) ? red[threadIdx.x] : 0.0f;
    if (warp == 0) t = warp_sum(t);
    if (threadIdx.x == 0) red[0] = t;
    __syncthreads();
    float r = red[0];
    __syncthreads();
    return r;
}

// Block rows: row 0 = embed[anchor], rows 1..7 = embed[MASK]. Optional target embed_norm.
__global__ void __launch_bounds__(256) dfl_embed_block_kernel(
    const __nv_bfloat16* __restrict__ embed, const __nv_bfloat16* __restrict__ embed_norm,
    const int32_t* __restrict__ d_anchor, float* __restrict__ out) {
    __shared__ float red[32];
    const int t = blockIdx.x;
    const int32_t tok = (t == 0) ? *d_anchor : MASK_TOKEN;
    const __nv_bfloat16* row = embed + (size_t)tok * DH;
    float* dst = out + (size_t)t * DH;
    float ss = 0.0f;
    for (int c = threadIdx.x; c < DH; c += 256) {
        float v = __bfloat162float(row[c]);
        dst[c] = v;
        ss += v * v;
    }
    if (embed_norm == nullptr) return;
    float tot = block_sum_256(ss, red);
    float inv = rsqrtf(tot / DH + EPS);
    for (int c = threadIdx.x; c < DH; c += 256) dst[c] = dst[c] * inv * __bfloat162float(embed_norm[c]);
}

// out[r] = rmsnorm(in[r]) * w (w may be null). in and out may alias.
__global__ void __launch_bounds__(256) dfl_rmsnorm_kernel(
    const float* __restrict__ in, const __nv_bfloat16* __restrict__ w, float* out, int width) {
    __shared__ float red[32];
    const float* src = in + (size_t)blockIdx.x * width;
    float* dst = out + (size_t)blockIdx.x * width;
    float ss = 0.0f;
    for (int c = threadIdx.x; c < width; c += 256) { float v = src[c]; ss += v * v; }
    float inv = rsqrtf(block_sum_256(ss, red) / width + EPS);
    for (int c = threadIdx.x; c < width; c += 256) {
        float g = w ? __bfloat162float(w[c]) : 1.0f;
        dst[c] = src[c] * inv * g;
    }
}

// Same, writing bf16 (prefill commit GEMM input).
__global__ void __launch_bounds__(256) dfl_rmsnorm_to_bf16_kernel(
    const float* __restrict__ in, const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ out, int width) {
    __shared__ float red[32];
    const float* src = in + (size_t)blockIdx.x * width;
    __nv_bfloat16* dst = out + (size_t)blockIdx.x * width;
    float ss = 0.0f;
    for (int c = threadIdx.x; c < width; c += 256) { float v = src[c]; ss += v * v; }
    float inv = rsqrtf(block_sum_256(ss, red) / width + EPS);
    for (int c = threadIdx.x; c < width; c += 256) dst[c] = __float2bfloat16(src[c] * inv * __bfloat162float(w[c]));
}

// Grouped dynamic causal conv over the 8 block rows (kernel size 2):
// out[t,c] = res[t,c] + sum_o (base[which,o,c] + dyn[t, which,o, c/16]) * x[t-o, c],  x[-1] = 0.
__global__ void __launch_bounds__(256) dfl_conv_kernel(
    const float* __restrict__ x, const float* __restrict__ dyn, const __nv_bfloat16* __restrict__ base,
    int which, const float* res, float* out) {
    const int t = blockIdx.y;
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= DH) return;
    const int g = c / CG;
    const float* dr = dyn + (size_t)t * CDYN + which * (CK * NG);
    const __nv_bfloat16* bw = base + (size_t)which * CK * DH;
    float v = (__bfloat162float(bw[c]) + dr[g]) * x[(size_t)t * DH + c];
    if (t > 0) v += (__bfloat162float(bw[DH + c]) + dr[NG + g]) * x[(size_t)(t - 1) * DH + c];
    if (res) v += res[(size_t)t * DH + c];
    out[(size_t)t * DH + c] = v;
}

__device__ __forceinline__ void rope_pair(float& lo, float& hi, int i, int32_t pos) {
    // Qwen3 default RoPE (rotate_half), inv_freq = theta^(-2i/128); double precision angle.
    double inv_freq = pow(ROPE_THETA, -2.0 * (double)i / (double)HD);
    double s, c;
    sincos((double)pos * inv_freq, &s, &c);
    float cf = (float)c, sf = (float)s;
    float a = lo, b = hi;
    lo = a * cf - b * sf;
    hi = b * cf + a * sf;
}

// Per-head RMSNorm (+weight) and RoPE for one 128-wide head held by 64 threads (2 elems each).
__device__ __forceinline__ void norm_rope_head(
    const float* __restrict__ src, const __nv_bfloat16* __restrict__ w, float* __restrict__ dst,
    int32_t pos, float* red, bool do_norm, bool do_rope) {
    const int i = threadIdx.x; // 0..63
    float lo = src[i], hi = src[i + 64];
    if (do_norm) {
        float ss = warp_sum(lo * lo + hi * hi);
        if ((i & 31) == 0) red[i >> 5] = ss;
        __syncthreads();
        float inv = rsqrtf((red[0] + red[1]) / HD + EPS);
        __syncthreads();
        lo = lo * inv * __bfloat162float(w[i]);
        hi = hi * inv * __bfloat162float(w[i + 64]);
    }
    if (do_rope) rope_pair(lo, hi, i, pos);
    dst[i] = lo;
    dst[i + 64] = hi;
}

// Block q/k/v: grid (8 rows, 32 q + 8 k + 8 v heads), block 64.
__global__ void __launch_bounds__(64) dfl_block_qkv_kernel(
    const float* __restrict__ q, const float* __restrict__ kv,
    const __nv_bfloat16* __restrict__ q_norm, const __nv_bfloat16* __restrict__ k_norm,
    const int32_t* __restrict__ d_pos, float* __restrict__ qn, float* __restrict__ kn, float* __restrict__ vn) {
    __shared__ float red[2];
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int32_t pos = *d_pos + t;
    if (h < NQ) {
        norm_rope_head(q + (size_t)t * QD + h * HD, q_norm, qn + ((size_t)t * NQ + h) * HD, pos, red, true, true);
    } else if (h < NQ + NKV) {
        const int kh = h - NQ;
        norm_rope_head(kv + (size_t)t * 2 * KVD + kh * HD, k_norm, kn + ((size_t)t * NKV + kh) * HD, pos, red, true, true);
    } else {
        const int vh = h - NQ - NKV;
        norm_rope_head(kv + (size_t)t * 2 * KVD + KVD + vh * HD, nullptr, vn + ((size_t)t * NKV + vh) * HD, pos, red, false, false);
    }
}

// Context k/v store: kv [B, 10240] (layer l: k at l*2048, v at l*2048+1024).
// grid (B, NL * 2 * NKV), block 64. pos = *d_pos + pos_off + b.
__global__ void __launch_bounds__(64) dfl_ctx_store_kernel(
    const float* __restrict__ kv, KNormSet kn, const int32_t* __restrict__ d_pos, int pos_off,
    __nv_bfloat16* __restrict__ ring) {
    __shared__ float red[2];
    __shared__ float tmp[HD];
    const int b = blockIdx.x;
    const int l = blockIdx.y / (2 * NKV);
    const int r = blockIdx.y % (2 * NKV);
    const bool is_v = r >= NKV;
    const int h = is_v ? r - NKV : r;
    const int32_t pos = *d_pos + pos_off + b;
    const float* src = kv + (size_t)b * KV_ALL + (size_t)l * 2 * KVD + (is_v ? KVD : 0) + h * HD;
    norm_rope_head(src, kn.w[l], tmp, pos, red, !is_v, !is_v);
    __syncthreads();
    const size_t slot = (size_t)(pos % RING);
    __nv_bfloat16* dst = ring + (is_v ? (size_t)NL * RING * KVD : 0) + ((size_t)l * RING + slot) * KVD + h * HD;
    const int i = threadIdx.x;
    dst[i] = __float2bfloat16(tmp[i]);
    dst[i + 64] = __float2bfloat16(tmp[i + 64]);
}

// Split-K attention for 8 block rows x 4 q heads of one kv head over ring context + block keys.
// grid (ATT_SPLITS, NKV), block 256, dynamic smem.
__global__ void __launch_bounds__(256) dfl_attn_kernel(
    const float* __restrict__ qn, const float* __restrict__ kn, const float* __restrict__ vn,
    const __nv_bfloat16* __restrict__ ring_k, const __nv_bfloat16* __restrict__ ring_v,
    const int32_t* __restrict__ d_pos, float* __restrict__ part) {
    extern __shared__ float sm[];
    float* q_s = sm;                       // [32][128]
    float* k_s = q_s + ATT_Q * HD;         // [32][129] padded
    float* v_s = k_s + ATT_TILE * (HD + 1);// [32][128]
    float* sc = v_s + ATT_TILE * HD;       // [32 q][32 k]

    const int split = blockIdx.x;
    const int kvh = blockIdx.y;
    const int tid = threadIdx.x;
    const int32_t dpos = *d_pos;
    const int32_t lo0 = max(0, dpos - (WIN - 1));
    const int n_ctx = dpos - lo0;
    const int nk = n_ctx + BLK;
    const int j0 = split * ATT_CHUNK;
    const float scale = rsqrtf((float)HD);

    // queries: qi = t*4 + g, q head = kvh*4 + g
    for (int e = tid; e < ATT_Q * HD; e += 256) {
        int qi = e / HD, d = e % HD;
        int t = qi / GQA, g = qi % GQA;
        q_s[e] = qn[((size_t)t * NQ + kvh * GQA + g) * HD + d] * scale;
    }

    const int my_q = tid >> 3;       // 0..31
    const int dsub = (tid & 7) * 16; // 16 dims per thread
    float m = -INFINITY, l = 0.0f;
    float acc[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;

    for (int tile = 0; tile < ATT_CHUNK / ATT_TILE; ++tile) {
        const int jt = j0 + tile * ATT_TILE;
        __syncthreads();
        if (jt < nk) {
            for (int e = tid; e < ATT_TILE * HD; e += 256) {
                int kk = e / HD, d = e % HD;
                int j = jt + kk;
                float kvk = 0.0f, kvv = 0.0f;
                if (j < n_ctx) {
                    size_t slot = (size_t)((lo0 + j) % RING);
                    kvk = __bfloat162float(ring_k[slot * KVD + kvh * HD + d]);
                    kvv = __bfloat162float(ring_v[slot * KVD + kvh * HD + d]);
                } else if (j < nk) {
                    int bt = j - n_ctx;
                    kvk = kn[((size_t)bt * NKV + kvh) * HD + d];
                    kvv = vn[((size_t)bt * NKV + kvh) * HD + d];
                }
                k_s[kk * (HD + 1) + d] = kvk;
                v_s[kk * HD + d] = kvv;
            }
        }
        __syncthreads();
        if (jt >= nk) break;
        // scores: 1024 (q,k) pairs, 4 per thread
        for (int p = tid; p < ATT_Q * ATT_TILE; p += 256) {
            int qi = p / ATT_TILE, kk = p % ATT_TILE;
            int j = jt + kk;
            int t = qi / GQA;
            bool vis = (j < nk) && (j >= n_ctx || (lo0 + j) >= dpos + t - (WIN - 1));
            float s = -INFINITY;
            if (vis) {
                const float* qr = q_s + qi * HD;
                const float* kr = k_s + kk * (HD + 1);
                float a = 0.0f;
                #pragma unroll 8
                for (int d = 0; d < HD; ++d) a = fmaf(qr[d], kr[d], a);
                s = a;
            }
            sc[qi * ATT_TILE + kk] = s;
        }
        __syncthreads();
        const float* srow = sc + my_q * ATT_TILE;
        float tmax = -INFINITY;
        for (int kk = 0; kk < ATT_TILE; ++kk) tmax = fmaxf(tmax, srow[kk]);
        if (tmax == -INFINITY) continue;
        float m_new = fmaxf(m, tmax);
        float corr = (m == -INFINITY) ? 0.0f : __expf(m - m_new);
        l *= corr;
        #pragma unroll
        for (int i = 0; i < 16; ++i) acc[i] *= corr;
        for (int kk = 0; kk < ATT_TILE; ++kk) {
            float s = srow[kk];
            if (s == -INFINITY) continue;
            float p = __expf(s - m_new);
            l += p;
            const float* vr = v_s + kk * HD + dsub;
            #pragma unroll
            for (int i = 0; i < 16; ++i) acc[i] = fmaf(p, vr[i], acc[i]);
        }
        m = m_new;
    }
    float* dst = part + (((size_t)split * NKV + kvh) * ATT_Q + my_q) * (HD + 2);
    #pragma unroll
    for (int i = 0; i < 16; ++i) dst[dsub + i] = acc[i];
    if ((tid & 7) == 0) { dst[HD] = m; dst[HD + 1] = l; }
}

// Merge split partials -> attn [8, 4096]. grid (8 rows * 32 heads), block 128.
__global__ void __launch_bounds__(128) dfl_attn_merge_kernel(const float* __restrict__ part, float* __restrict__ attn) {
    const int t = blockIdx.x / NQ;
    const int h = blockIdx.x % NQ;
    const int kvh = h / GQA, g = h % GQA;
    const int qi = t * GQA + g;
    const int d = threadIdx.x;
    float M = -INFINITY;
    for (int s = 0; s < ATT_SPLITS; ++s) {
        const float* p = part + (((size_t)s * NKV + kvh) * ATT_Q + qi) * (HD + 2);
        M = fmaxf(M, p[HD]);
    }
    float num = 0.0f, den = 0.0f;
    for (int s = 0; s < ATT_SPLITS; ++s) {
        const float* p = part + (((size_t)s * NKV + kvh) * ATT_Q + qi) * (HD + 2);
        float ms = p[HD];
        if (ms == -INFINITY) continue;
        float w = __expf(ms - M);
        num += p[d] * w;
        den += p[HD + 1] * w;
    }
    attn[(size_t)t * QD + h * HD + d] = (den > 0.0f) ? num / den : 0.0f;
}

__global__ void dfl_swiglu_kernel(const float* __restrict__ gu, float* __restrict__ act) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= FF) return;
    float gv = gu[(size_t)t * 2 * FF + i];
    float uv = gu[(size_t)t * 2 * FF + FF + i];
    act[(size_t)t * FF + i] = gv / (1.0f + __expf(-gv)) * uv;
}

// cat[b, s*DH + c] = taps[s, b, c] (taps stride = B rows); optional bf16 output.
__global__ void dfl_gather_taps_kernel(const float* __restrict__ taps, int B, int b_off, int nb, float* out_f, __nv_bfloat16* out_h) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)nb * FC_IN;
    if (idx >= total) return;
    int b = (int)(idx / FC_IN);
    int k = (int)(idx % FC_IN);
    int s = k / DH, c = k % DH;
    float v = taps[((size_t)s * B + b_off + b) * DH + c];
    if (out_f) out_f[idx] = v;
    else out_h[idx] = __float2bfloat16(v);
}

// Block-wide argmax helper over (val, idx) pairs, 256 threads.
__device__ __forceinline__ void block_argmax_256(float& v, int& i, float* rv, int* ri) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, v, off);
        int oi = __shfl_xor_sync(0xffffffffu, i, off);
        if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
    }
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { rv[warp] = v; ri[warp] = i; }
    __syncthreads();
    if (warp == 0) {
        v = (lane < 8) ? rv[lane] : -INFINITY;
        i = (lane < 8) ? ri[lane] : 0x7fffffff;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, v, off);
            int oi = __shfl_xor_sync(0xffffffffu, i, off);
            if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
        }
        if (lane == 0) { rv[0] = v; ri[0] = i; }
    }
    __syncthreads();
    v = rv[0];
    i = ri[0];
    __syncthreads();
}

constexpr int TOPK_PER = (V + TOPK_PARTS - 1) / TOPK_PARTS; // 3126

// Stage 1: grid (7, PARTS), block 256: top-16 of each partition.
__global__ void __launch_bounds__(256) dfl_topk_stage1_kernel(const float* __restrict__ logits, float* __restrict__ ov, int32_t* __restrict__ oi) {
    __shared__ float vals[TOPK_PER];
    __shared__ float rv[8];
    __shared__ int ri[8];
    const int row = blockIdx.x, p = blockIdx.y;
    const int start = p * TOPK_PER;
    const int n = min(TOPK_PER, V - start);
    for (int e = threadIdx.x; e < TOPK_PER; e += 256) {
        float x = (e < n) ? logits[(size_t)row * V + start + e] : -INFINITY;
        vals[e] = isnan(x) ? -INFINITY : x; // a NaN row must not yield out-of-range winners
    }
    __syncthreads();
    for (int r = 0; r < TOPK; ++r) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int e = threadIdx.x; e < TOPK_PER; e += 256) {
            float x = vals[e];
            if (x > bv) { bv = x; bi = e; }
        }
        block_argmax_256(bv, bi, rv, ri);
        if (threadIdx.x == 0) {
            size_t o = ((size_t)row * TOPK_PARTS + p) * TOPK + r;
            ov[o] = bv;
            oi[o] = (bi < TOPK_PER) ? start + bi : 0;
            if (bi < TOPK_PER) vals[bi] = -INFINITY;
        }
        __syncthreads();
    }
}

// Stage 2: grid 7, block 256: top-16 of the 1024 partition winners.
__global__ void __launch_bounds__(256) dfl_topk_stage2_kernel(const float* __restrict__ iv, const int32_t* __restrict__ ii, float* __restrict__ cv, int32_t* __restrict__ ci) {
    constexpr int N = TOPK_PARTS * TOPK;
    __shared__ float vals[N];
    __shared__ float rv[8];
    __shared__ int ri[8];
    const int row = blockIdx.x;
    for (int e = threadIdx.x; e < N; e += 256) vals[e] = iv[(size_t)row * N + e];
    __syncthreads();
    for (int r = 0; r < TOPK; ++r) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int e = threadIdx.x; e < N; e += 256) {
            float x = vals[e];
            if (x > bv) { bv = x; bi = e; }
        }
        block_argmax_256(bv, bi, rv, ri);
        if (threadIdx.x == 0) {
            if (bi < N) {
                cv[row * TOPK + r] = bv;
                ci[row * TOPK + r] = ii[(size_t)row * N + bi];
                vals[bi] = -INFINITY;
            } else {
                cv[row * TOPK + r] = -INFINITY;
                ci[row * TOPK + r] = 0;
            }
        }
        __syncthreads();
    }
}

// Candidate path selector, one block of 256 threads, 7 sequential positions.
__global__ void __launch_bounds__(256) dfl_select_kernel(
    const float* __restrict__ cand_v, const int32_t* __restrict__ cand_i, const float* __restrict__ hp,
    const __nv_bfloat16* __restrict__ pred_cb, const __nv_bfloat16* __restrict__ succ_cb,
    const int32_t* __restrict__ d_anchor, const int32_t* __restrict__ d_pos, const SampleConfig* __restrict__ cfg,
    int32_t* __restrict__ prev_cand, float* __restrict__ probs,
    int32_t* __restrict__ out_drafts, float* __restrict__ out_conf, float* __restrict__ out_q_sel, int32_t* __restrict__ out_is_delta) {
    __shared__ float a[RANK];
    __shared__ float score[TOPK];
    __shared__ int32_t s_prev;
    const int tid = threadIdx.x;
    const bool sample = (cfg != nullptr && cfg->inv_temp > 0.0f && cfg->top_k != 1);
    const float inv_temp = sample ? cfg->inv_temp : 1.0f;
    const uint64_t seed = (cfg ? cfg->seed : 0ULL) ^ SEED_SALT;
    const int32_t pos0 = *d_pos;
    if (tid == 0) {
        s_prev = *d_anchor;
        if (out_is_delta) *out_is_delta = 0;
    }
    __syncthreads();
    for (int i = 0; i < ND; ++i) {
        const int32_t prev = s_prev;
        a[tid] = __bfloat162float(pred_cb[(size_t)prev * RANK + tid]) * hp[i * RANK + tid];
        __syncthreads();
        const int k = tid >> 4, lane = tid & 15;
        const int32_t ck = cand_i[i * TOPK + k];
        const __nv_bfloat16* srow = succ_cb + (size_t)ck * RANK + lane * 16;
        float c = 0.0f;
        #pragma unroll
        for (int r = 0; r < 16; ++r) c = fmaf(a[lane * 16 + r], __bfloat162float(srow[r]), c);
        #pragma unroll
        for (int off = 8; off > 0; off >>= 1) c += __shfl_xor_sync(0xffffffffu, c, off);
        if (lane == 0) {
            float u = cand_v[i * TOPK + k];
            score[k] = SOFTCAP * tanhf((u + c) / SOFTCAP);
        }
        __syncthreads();
        if (tid == 0) {
            float mx = -INFINITY;
            for (int j = 0; j < TOPK; ++j) mx = fmaxf(mx, score[j] * inv_temp);
            float z = 0.0f;
            float e[TOPK];
            for (int j = 0; j < TOPK; ++j) { e[j] = __expf(score[j] * inv_temp - mx); z += e[j]; }
            int best = 0;
            if (sample) {
                float bs = -INFINITY;
                for (int j = 0; j < TOPK; ++j) {
                    float g = score[j] * inv_temp + hash_gumbel(seed, pos0 + 1 + i, cand_i[i * TOPK + j]);
                    if (g > bs) { bs = g; best = j; }
                }
            } else {
                float bs = -INFINITY;
                for (int j = 0; j < TOPK; ++j) if (score[j] > bs) { bs = score[j]; best = j; }
            }
            const float qb = e[best] / z;
            float* prow = probs + (size_t)i * V;
            for (int j = 0; j < TOPK; ++j) {
                int32_t old = prev_cand[i * TOPK + j];
                if (old >= 0 && old < V) prow[old] = 0.0f;
            }
            for (int j = 0; j < TOPK; ++j) {
                int32_t cj = cand_i[i * TOPK + j];
                prev_cand[i * TOPK + j] = cj;
                if (cj >= 0 && cj < V) prow[cj] = sample ? e[j] / z : ((j == best) ? 1.0f : 0.0f);
            }
            const int32_t tok = cand_i[i * TOPK + best];
            out_drafts[i] = tok;
            if (out_conf) out_conf[i] = qb;
            if (out_q_sel) out_q_sel[i] = sample ? qb : 1.0f;
            s_prev = tok;
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Host side: safetensors reader, upload, launches
// ---------------------------------------------------------------------------

#define DFL_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "[dflash2] CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return false; } } while (0)

struct StFile {
    int fd = -1;
    size_t size = 0;
    const uint8_t* map = nullptr;
    char* header = nullptr; // NUL-terminated copy of the JSON header
    size_t data_off = 0;

    bool open(const char* path) {
        fd = ::open(path, O_RDONLY);
        if (fd < 0) return false;
        struct stat st;
        if (fstat(fd, &st) != 0) return false;
        size = (size_t)st.st_size;
        void* p = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (p == MAP_FAILED) return false;
        map = static_cast<const uint8_t*>(p);
        uint64_t n = 0;
        memcpy(&n, map, 8);
        if (n + 8 > size) return false;
        header = static_cast<char*>(malloc(n + 1));
        if (!header) return false;
        memcpy(header, map + 8, n);
        header[n] = 0;
        data_off = 8 + n;
        return true;
    }
    void close() {
        if (map) munmap(const_cast<uint8_t*>(map), size);
        free(header);
        header = nullptr;
        if (fd >= 0) ::close(fd);
        map = nullptr;
        fd = -1;
    }
    // Returns pointer and byte length of a BF16 tensor; nullptr if missing.
    const void* find(const char* name, size_t& bytes) const {
        char key[256];
        snprintf(key, sizeof(key), "\"%s\":", name);
        const char* k = strstr(header, key);
        if (!k) return nullptr;
        const char* d = strstr(k, "\"dtype\"");
        const char* o = strstr(k, "\"data_offsets\"");
        if (!d || !o) return nullptr;
        const char* dv = strchr(d + 7, '"');
        if (!dv || strncmp(dv + 1, "BF16", 4) != 0) {
            fprintf(stderr, "[dflash2] tensor %s is not BF16\n", name);
            return nullptr;
        }
        const char* lb = strchr(o, '[');
        unsigned long long a = 0, b = 0;
        if (!lb || (sscanf(lb, "[%llu,%llu]", &a, &b) != 2 && sscanf(lb, "[%llu, %llu]", &a, &b) != 2)) return nullptr;
        bytes = (size_t)(b - a);
        return map + data_off + a;
    }
};

struct Loader {
    Ctx* c;
    StFile* f;
    __nv_bfloat16* stage = nullptr; // device staging for FP8/NVFP4 packing
    float* amax = nullptr;
    bool ok = true;

    void* carve(size_t bytes) {
        size_t off = (c->vram_used + 255) & ~(size_t)255;
        if (off + bytes > c->vram_bytes) { ok = false; fprintf(stderr, "[dflash2] VRAM arena overflow\n"); return nullptr; }
        c->vram_used = off + bytes;
        return static_cast<uint8_t*>(c->vram) + off;
    }
    const void* src(const char* name, size_t expect) {
        size_t bytes = 0;
        const void* p = f->find(name, bytes);
        if (!p || bytes != expect) {
            fprintf(stderr, "[dflash2] missing or wrong-size tensor %s (%zu vs %zu)\n", name, bytes, expect);
            ok = false;
            return nullptr;
        }
        return p;
    }
    const __nv_bfloat16* vec(const char* name, size_t n) {
        const void* s = src(name, n * 2);
        if (!s) return nullptr;
        void* d = carve(n * 2);
        if (!d) return nullptr;
        cudaMemcpy(d, s, n * 2, cudaMemcpyHostToDevice);
        c->weight_bytes += n * 2;
        return static_cast<const __nv_bfloat16*>(d);
    }
    // Packs a device bf16 matrix [rows, cols] into FP8 (q=1) or NVFP4 (q=2).
    LinearWeight pack_dev(const void* dev_bf16, int rows, int cols, int q) {
        LinearWeight w;
        w.rows = rows;
        w.cols = cols;
        if (q == 1) {
            void* d = carve((size_t)rows * cols);
            float* rs = static_cast<float*>(carve((size_t)rows * 4));
            if (!d || !rs) return w;
            squid_pack_bf16_to_fp8(dev_bf16, d, rs, rows, cols);
            w.kind = WeightKind::FP8;
            w.data = d;
            w.row_scales = rs;
            c->weight_bytes += (size_t)rows * cols + (size_t)rows * 4;
        } else {
            void* d = carve((size_t)rows * cols / 2);
            uint8_t* sc = static_cast<uint8_t*>(carve((size_t)rows * cols / 16));
            if (!d || !sc) return w;
            w.global_scale = squid_pack_bf16_to_nvfp4(dev_bf16, d, sc, amax, rows, cols);
            w.kind = WeightKind::NVFP4;
            w.data = d;
            w.scales = sc;
            c->weight_bytes += (size_t)rows * cols / 2 + (size_t)rows * cols / 16;
        }
        cudaDeviceSynchronize();
        return w;
    }
    // Uploads one or two row-stacked bf16 matrices as a LinearWeight in format q.
    LinearWeight mat(const char* a, int rows_a, const char* b, int rows_b, int cols, int q) {
        LinearWeight w;
        const int rows = rows_a + rows_b;
        w.rows = rows;
        w.cols = cols;
        const size_t ba = (size_t)rows_a * cols * 2, bb = (size_t)rows_b * cols * 2;
        const void* sa = src(a, ba);
        const void* sb = rows_b ? src(b, bb) : nullptr;
        if (!ok) return w;
        if (q == 0) {
            uint8_t* d = static_cast<uint8_t*>(carve(ba + bb));
            if (!d) return w;
            cudaMemcpy(d, sa, ba, cudaMemcpyHostToDevice);
            if (rows_b) cudaMemcpy(d + ba, sb, bb, cudaMemcpyHostToDevice);
            w.kind = WeightKind::BF16;
            w.data = d;
            c->weight_bytes += ba + bb;
            return w;
        }
        uint8_t* st = reinterpret_cast<uint8_t*>(stage);
        cudaMemcpy(st, sa, ba, cudaMemcpyHostToDevice);
        if (rows_b) cudaMemcpy(st + ba, sb, bb, cudaMemcpyHostToDevice);
        return pack_dev(stage, rows, cols, q);
    }
};


// fp16 activation scratch for the tensor-core linear kernels (see launch_linear_rows).
static __half* g_xh = nullptr;
constexpr int XH_MAX_COLS = FC_IN;

static size_t qbytes(size_t rows, size_t cols, int q) {
    if (q == 0) return rows * cols * 2;
    if (q == 1) return rows * cols + rows * 4 + 512;
    return rows * cols / 2 + rows * cols / 16 + 512;
}

static size_t ctx_vram_bytes(int q, bool codebook_vram) {
    size_t layer = qbytes(CDYN, DH, q) * 2 + qbytes(QD, DH, q) + qbytes(DH, QD, q) + qbytes(2 * FF, DH, q) + qbytes(DH, FF, q);
    size_t t = 0;
    t += NL * (layer + 16 * 256 + (size_t)(4 * DH + 2 * HD + 8 * DH) * 2);
    t += (size_t)DH * FC_IN * 2 + (size_t)KV_ALL * DH * 2 + (size_t)RANK * DH * 2 + 4 * DH * 2;
    if (q != 0) t += qbytes(DH, FC_IN, q) + qbytes(KV_ALL, DH, q) + 4;
    if (codebook_vram) t += 2 * (size_t)VOCAB_PADDED * RANK * 2;
    t += 2 * (size_t)NL * RING * KVD * 2; // ring
    t += (size_t)BLK * (4 * DH + CDYN + QD + 2 * KVD + QD + 2 * KVD + QD + 2 * FF + FF) * 4;
    t += (size_t)ATT_SPLITS * NKV * ATT_Q * (HD + 2) * 4;
    t += 2 * (size_t)ND * V * 4; // logits, probs
    t += (size_t)ND * TOPK_PARTS * TOPK * 8 + (size_t)ND * TOPK * 12 + (size_t)ND * RANK * 4;
    t += (size_t)BLK * (FC_IN + 2 * DH + KV_ALL) * 4;
    t += (size_t)PREFILL_SUB * (FC_IN * 2 + DH * 4 + DH * 2 + KV_ALL * 4);
    t += 128 * 256; // alignment slack
    return t;
}

static bool ctx_create(Ctx* c, const char* dir, int q, bool embed_norm, bool codebook_vram) {
    c->quant = q;
    c->use_embed_norm = embed_norm;

    char path[1200];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    FILE* cf = fopen(path, "rb");
    if (!cf) { fprintf(stderr, "[dflash2] cannot open %s\n", path); return false; }
    static char flat[65536];
    size_t fl = 0;
    int ch;
    while ((ch = fgetc(cf)) != EOF && fl + 1 < sizeof(flat)) {
        if (ch != ' ' && ch != '\n' && ch != '\r' && ch != '\t') flat[fl++] = (char)ch;
    }
    flat[fl] = 0;
    fclose(cf);
    auto fhas = [&](const char* s) { return strstr(flat, s) != nullptr; };
    if (!fhas("DFlash2DraftModel") || !fhas("\"block_size\":8") || !fhas("\"selector_top_k\":16") ||
        !fhas("\"selector_rank\":256") || !fhas("\"conv_group_size\":16") || !fhas("\"conv_kernel_size\":2") ||
        !fhas("\"mask_token_id\":200056") || !fhas("\"target_layer_ids\":[5,23,41,48]") ||
        !fhas("\"hidden_size\":2304") || !fhas("\"num_hidden_layers\":5") || !fhas("\"num_attention_heads\":32") ||
        !fhas("\"num_key_value_heads\":8") || !fhas("\"intermediate_size\":6912") || !fhas("\"sliding_window\":2048") ||
        !fhas("\"selector_softcap_mode\":\"after_correction\"") || !fhas("\"final_logit_softcapping\":30.0") ||
        !fhas("\"rope_theta\":10000.0")) {
        fprintf(stderr, "[dflash2] config.json does not match the supported Inkling-2-Mini-DFlash2 layout\n");
        return false;
    }

    StFile f;
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    if (!f.open(path)) { fprintf(stderr, "[dflash2] cannot open %s\n", path); return false; }

    c->vram_bytes = ctx_vram_bytes(q, codebook_vram);
    DFL_CHECK(cudaMalloc(&c->vram, c->vram_bytes));
    DFL_CHECK(cudaMemset(c->vram, 0, c->vram_bytes));

    if (!g_xh) DFL_CHECK(cudaMalloc(&g_xh, (size_t)8 * XH_MAX_COLS * sizeof(__half)));
    Loader L{c, &f};
    if (q != 0) {
        DFL_CHECK(cudaMalloc(&L.stage, (size_t)2 * FF * DH * 2));
        L.amax = static_cast<float*>(L.carve(4));
    }

    for (int l = 0; l < NL; ++l) {
        char n[16][128];
        const char* tails[16] = {"input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_norm.weight",
            "self_attn.k_norm.weight", "attention_conv.base_kernel", "mlp_conv.base_kernel",
            "attention_conv.kernel_projection.weight", "mlp_conv.kernel_projection.weight", "self_attn.q_proj.weight",
            "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
            "self_attn.k_proj.weight", "self_attn.v_proj.weight", ""};
        for (int i = 0; i < 16; ++i) snprintf(n[i], sizeof(n[i]), "layers.%d.%s", l, tails[i]);
        Layer& ly = c->layers[l];
        ly.in_norm = L.vec(n[0], DH);
        ly.post_norm = L.vec(n[1], DH);
        ly.q_norm = L.vec(n[2], HD);
        ly.k_norm = L.vec(n[3], HD);
        ly.attn_base = L.vec(n[4], 2 * CK * DH);
        ly.mlp_base = L.vec(n[5], 2 * CK * DH);
        ly.w_attn_conv = L.mat(n[6], CDYN, "", 0, DH, q);
        ly.w_mlp_conv = L.mat(n[7], CDYN, "", 0, DH, q);
        ly.w_q = L.mat(n[8], QD, "", 0, DH, q);
        ly.w_o = L.mat(n[9], DH, "", 0, QD, q);
        ly.w_gu = L.mat(n[10], FF, n[11], FF, DH, q);
        ly.w_down = L.mat(n[12], DH, "", 0, FF, q);
        c->knorms.w[l] = ly.k_norm;
        if (!L.ok) break;
    }
    if (L.ok) {
        // Stacked context k/v for all layers (bf16, shared by the block path through per-layer views).
        uint8_t* kv = static_cast<uint8_t*>(L.carve((size_t)KV_ALL * DH * 2));
        for (int l = 0; l < NL && L.ok; ++l) {
            char nk[128], nv[128];
            snprintf(nk, sizeof(nk), "layers.%d.self_attn.k_proj.weight", l);
            snprintf(nv, sizeof(nv), "layers.%d.self_attn.v_proj.weight", l);
            const size_t mb = (size_t)KVD * DH * 2;
            const void* k = L.src(nk, mb);
            const void* v = L.src(nv, mb);
            if (!L.ok || !kv) break;
            cudaMemcpy(kv + (size_t)(2 * l) * mb, k, mb, cudaMemcpyHostToDevice);
            cudaMemcpy(kv + (size_t)(2 * l + 1) * mb, v, mb, cudaMemcpyHostToDevice);
            LinearWeight& w = c->layers[l].w_kv;
            w.kind = WeightKind::BF16;
            w.rows = 2 * KVD;
            w.cols = DH;
            w.data = kv + (size_t)(2 * l) * mb;
        }
        c->w_kv_all.kind = WeightKind::BF16;
        c->w_kv_all.rows = KV_ALL;
        c->w_kv_all.cols = DH;
        c->w_kv_all.data = kv;
        c->weight_bytes += (size_t)KV_ALL * DH * 2;
        c->w_kv_all_q = c->w_kv_all;
        if (q != 0 && L.ok) {
            // Quantized copy for decode commit and the block k/v path (per-layer row views share one scale).
            c->w_kv_all_q = L.pack_dev(kv, KV_ALL, DH, q);
            for (int l = 0; l < NL; ++l) {
                LinearWeight w = c->w_kv_all_q;
                const size_t r0 = (size_t)(2 * l) * KVD;
                w.rows = 2 * KVD;
                if (q == 1) {
                    w.data = static_cast<uint8_t*>(c->w_kv_all_q.data) + r0 * DH;
                    w.row_scales = c->w_kv_all_q.row_scales + r0;
                } else {
                    w.data = static_cast<uint8_t*>(c->w_kv_all_q.data) + r0 * DH / 2;
                    w.scales = c->w_kv_all_q.scales + r0 * DH / 16;
                }
                c->layers[l].w_kv = w;
            }
        }
    }
    if (L.ok) {
        c->w_fc = L.mat("fc.weight", DH, "", 0, FC_IN, 0);
        c->w_fc_q = (q != 0) ? L.pack_dev(c->w_fc.data, DH, FC_IN, q) : c->w_fc;
        c->w_hproj = L.mat("candidate_selector.hidden_projection.weight", RANK, "", 0, DH, 0);
        c->hidden_norm = L.vec("hidden_norm.weight", DH);
        c->final_norm = L.vec("norm.weight", DH);
    }
    if (L.ok) {
        const size_t cb = (size_t)VOCAB_PADDED * RANK * 2;
        const void* sp = L.src("candidate_selector.predecessor_codebook", cb);
        const void* ss = L.src("candidate_selector.successor_codebook", cb);
        if (L.ok && codebook_vram) {
            void* dp = L.carve(cb);
            void* ds = L.carve(cb);
            if (dp && ds) {
                cudaMemcpy(dp, sp, cb, cudaMemcpyHostToDevice);
                cudaMemcpy(ds, ss, cb, cudaMemcpyHostToDevice);
                c->pred_cb = static_cast<const __nv_bfloat16*>(dp);
                c->succ_cb = static_cast<const __nv_bfloat16*>(ds);
                c->weight_bytes += 2 * cb;
            }
        } else if (L.ok) {
            DFL_CHECK(cudaHostAlloc(&c->pred_host, cb, cudaHostAllocMapped));
            DFL_CHECK(cudaHostAlloc(&c->succ_host, cb, cudaHostAllocMapped));
            memcpy(c->pred_host, sp, cb);
            memcpy(c->succ_host, ss, cb);
            void* dp = nullptr;
            void* ds = nullptr;
            DFL_CHECK(cudaHostGetDevicePointer(&dp, c->pred_host, 0));
            DFL_CHECK(cudaHostGetDevicePointer(&ds, c->succ_host, 0));
            c->pred_cb = static_cast<const __nv_bfloat16*>(dp);
            c->succ_cb = static_cast<const __nv_bfloat16*>(ds);
        }
    }
    if (L.stage) cudaFree(L.stage);
    f.close();
    if (!L.ok) return false;

    c->ring_bytes = 2 * (size_t)NL * RING * KVD * 2;
    c->ring = static_cast<__nv_bfloat16*>(L.carve(c->ring_bytes));
    auto F = [&](size_t n) { return static_cast<float*>(L.carve(n * 4)); };
    c->x = F((size_t)BLK * DH);
    c->nrm = F((size_t)BLK * DH);
    c->cv = F((size_t)BLK * DH);
    c->o = F((size_t)BLK * DH);
    c->dyn = F((size_t)BLK * CDYN);
    c->q = F((size_t)BLK * QD);
    c->kvb = F((size_t)BLK * 2 * KVD);
    c->qn = F((size_t)BLK * QD);
    c->kn = F((size_t)BLK * KVD);
    c->vn = F((size_t)BLK * KVD);
    c->part = F((size_t)ATT_SPLITS * NKV * ATT_Q * (HD + 2));
    c->attn = F((size_t)BLK * QD);
    c->gu = F((size_t)BLK * 2 * FF);
    c->act = F((size_t)BLK * FF);
    c->logits = F((size_t)ND * V);
    c->probs = F((size_t)ND * V);
    c->top_v1 = F((size_t)ND * TOPK_PARTS * TOPK);
    c->top_i1 = reinterpret_cast<int32_t*>(F((size_t)ND * TOPK_PARTS * TOPK));
    c->cand_v = F((size_t)ND * TOPK);
    c->cand_i = reinterpret_cast<int32_t*>(F((size_t)ND * TOPK));
    c->prev_cand = reinterpret_cast<int32_t*>(F((size_t)ND * TOPK));
    c->hp = F((size_t)ND * RANK);
    c->c_cat = F((size_t)BLK * FC_IN);
    c->c_ctx = F((size_t)BLK * DH);
    c->c_ctxn = F((size_t)BLK * DH);
    c->c_kv = F((size_t)BLK * KV_ALL);
    c->p_cat_h = static_cast<__nv_bfloat16*>(L.carve((size_t)PREFILL_SUB * FC_IN * 2));
    c->p_ctx = F((size_t)PREFILL_SUB * DH);
    c->p_ctxn_h = static_cast<__nv_bfloat16*>(L.carve((size_t)PREFILL_SUB * DH * 2));
    c->p_kv = F((size_t)PREFILL_SUB * KV_ALL);
    if (!L.ok) return false;
    DFL_CHECK(cudaMemset(c->prev_cand, 0xff, (size_t)ND * TOPK * 4)); // -1: nothing to clear yet

    if (cublasCreate(&c->blas) != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "[dflash2] cublasCreate failed\n"); return false; }
    const int att_smem = (ATT_Q * HD + ATT_TILE * (HD + 1) + ATT_TILE * HD + ATT_Q * ATT_TILE) * 4;
    DFL_CHECK(cudaFuncSetAttribute(dfl_attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, att_smem));
    DFL_CHECK(cudaDeviceSynchronize());
    return true;
}

// fp16 copy of the activations for the tensor-core NVFP4/FP8 kernels the target uses.

__global__ void dfl_f32_to_f16_kernel(const float* __restrict__ x, __half* __restrict__ y, int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half_rn(x[i] * scale);
}

__global__ void dfl_scale_kernel(float* __restrict__ y, int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] *= scale;
}

// in_scale (a power of two) keeps un-normalized inputs, such as the raw target taps, inside the
// fp16 range; the output is scaled back so the result is unchanged.
static void launch_linear_rows(const LinearWeight& w, const float* x, float* y, int B, cudaStream_t s,
                               float in_scale = 1.0f) {
    if (w.kind != WeightKind::BF16 && g_xh && B >= 1 && B <= 8 && w.cols <= XH_MAX_COLS) {
        const int n = B * w.cols;
        dfl_f32_to_f16_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, g_xh, n, in_scale);
        launch_linear_h(w, g_xh, y, B, false, s);
        if (in_scale != 1.0f) {
            const int m = B * w.rows;
            dfl_scale_kernel<<<(m + 255) / 256, 256, 0, s>>>(y, m, 1.0f / in_scale);
        }
        return;
    }
    launch_linear(w, x, y, B, false, s);
}

static void draft(Ctx* c, const void* embed, const void* embed_norm, const LinearWeight& unembed,
                  const int32_t* d_anchor, const int32_t* d_pos, const SampleConfig* cfg,
                  int32_t* out_drafts, float* out_conf, float* out_q_sel, int32_t* out_is_delta, cudaStream_t s) {
    const __nv_bfloat16* en = c->use_embed_norm ? static_cast<const __nv_bfloat16*>(embed_norm) : nullptr;
    dfl_embed_block_kernel<<<BLK, 256, 0, s>>>(static_cast<const __nv_bfloat16*>(embed), en, d_anchor, c->x);
    const __nv_bfloat16* ring_k = c->ring;
    const __nv_bfloat16* ring_v = c->ring + (size_t)NL * RING * KVD;
    const int att_smem = (ATT_Q * HD + ATT_TILE * (HD + 1) + ATT_TILE * HD + ATT_Q * ATT_TILE) * 4;
    const dim3 conv_grid((DH + 255) / 256, BLK);
    for (int l = 0; l < NL; ++l) {
        const Layer& ly = c->layers[l];
        // attention
        dfl_rmsnorm_kernel<<<BLK, 256, 0, s>>>(c->x, ly.in_norm, c->nrm, DH);
        launch_linear_rows(ly.w_attn_conv, c->nrm, c->dyn, BLK, s);
        dfl_conv_kernel<<<conv_grid, 256, 0, s>>>(c->nrm, c->dyn, ly.attn_base, 0, nullptr, c->cv);
        launch_linear_rows(ly.w_q, c->cv, c->q, BLK, s);
        launch_linear_rows(ly.w_kv, c->cv, c->kvb, BLK, s);
        dfl_block_qkv_kernel<<<dim3(BLK, NQ + 2 * NKV), 64, 0, s>>>(c->q, c->kvb, ly.q_norm, ly.k_norm, d_pos, c->qn, c->kn, c->vn);
        dfl_attn_kernel<<<dim3(ATT_SPLITS, NKV), 256, att_smem, s>>>(
            c->qn, c->kn, c->vn, ring_k + (size_t)l * RING * KVD, ring_v + (size_t)l * RING * KVD, d_pos, c->part);
        dfl_attn_merge_kernel<<<BLK * NQ, HD, 0, s>>>(c->part, c->attn);
        launch_linear_rows(ly.w_o, c->attn, c->o, BLK, s);
        dfl_conv_kernel<<<conv_grid, 256, 0, s>>>(c->o, c->dyn, ly.attn_base, 1, c->x, c->x);
        // MLP
        dfl_rmsnorm_kernel<<<BLK, 256, 0, s>>>(c->x, ly.post_norm, c->nrm, DH);
        launch_linear_rows(ly.w_mlp_conv, c->nrm, c->dyn, BLK, s);
        dfl_conv_kernel<<<conv_grid, 256, 0, s>>>(c->nrm, c->dyn, ly.mlp_base, 0, nullptr, c->cv);
        launch_linear_rows(ly.w_gu, c->cv, c->gu, BLK, s);
        dfl_swiglu_kernel<<<dim3((FF + 255) / 256, BLK), 256, 0, s>>>(c->gu, c->act);
        launch_linear_rows(ly.w_down, c->act, c->o, BLK, s);
        dfl_conv_kernel<<<conv_grid, 256, 0, s>>>(c->o, c->dyn, ly.mlp_base, 1, c->x, c->x);
    }
    dfl_rmsnorm_kernel<<<BLK, 256, 0, s>>>(c->x, c->final_norm, c->nrm, DH);
    const float* h7 = c->nrm + DH; // rows 1..7
    launch_linear_rows(unembed, h7, c->logits, ND, s);
    dfl_topk_stage1_kernel<<<dim3(ND, TOPK_PARTS), 256, 0, s>>>(c->logits, c->top_v1, c->top_i1);
    dfl_topk_stage2_kernel<<<ND, 256, 0, s>>>(c->top_v1, c->top_i1, c->cand_v, c->cand_i);
    launch_linear_rows(c->w_hproj, h7, c->hp, ND, s);
    dfl_select_kernel<<<1, 256, 0, s>>>(c->cand_v, c->cand_i, c->hp, c->pred_cb, c->succ_cb, d_anchor, d_pos, cfg,
                                        c->prev_cand, c->probs, out_drafts, out_conf, out_q_sel, out_is_delta);
}

static void commit(Ctx* c, const float* taps, int B, const int32_t* d_pos, cudaStream_t s) {
    const size_t total = (size_t)B * FC_IN;
    dfl_gather_taps_kernel<<<(unsigned)((total + 255) / 256), 256, 0, s>>>(taps, B, 0, B, c->c_cat, nullptr);
    launch_linear_rows(c->w_fc_q, c->c_cat, c->c_ctx, B, s, 1.0f / 64.0f); // raw residual taps
    dfl_rmsnorm_kernel<<<B, 256, 0, s>>>(c->c_ctx, c->hidden_norm, c->c_ctxn, DH);
    launch_linear_rows(c->w_kv_all_q, c->c_ctxn, c->c_kv, B, s);
    dfl_ctx_store_kernel<<<dim3(B, NL * 2 * NKV), 64, 0, s>>>(c->c_kv, c->knorms, d_pos, 0, c->ring);
}

static void gemm_bf16(cublasHandle_t h, const LinearWeight& w, const __nv_bfloat16* x, float* y, int B) {
    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, w.rows, B, w.cols, &alpha,
                 w.data, CUDA_R_16BF, w.cols, x, CUDA_R_16BF, w.cols, &beta,
                 y, CUDA_R_32F, w.rows, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    static bool reported = false;
    if (st != CUBLAS_STATUS_SUCCESS && !reported) {
        reported = true;
        fprintf(stderr, "[dflash2] cublasGemmEx failed (status %d); prefill context not committed\n", (int)st);
    }
}

static void commit_prefill(Ctx* c, const float* taps, int B, const int32_t* d_pos, cudaStream_t s) {
    cublasSetStream(c->blas, s);
    for (int off = 0; off < B; off += PREFILL_SUB) {
        const int nb = (B - off < PREFILL_SUB) ? B - off : PREFILL_SUB;
        const size_t total = (size_t)nb * FC_IN;
        dfl_gather_taps_kernel<<<(unsigned)((total + 255) / 256), 256, 0, s>>>(taps, B, off, nb, nullptr, c->p_cat_h);
        gemm_bf16(c->blas, c->w_fc, c->p_cat_h, c->p_ctx, nb);
        dfl_rmsnorm_to_bf16_kernel<<<nb, 256, 0, s>>>(c->p_ctx, c->hidden_norm, c->p_ctxn_h, DH);
        gemm_bf16(c->blas, c->w_kv_all, c->p_ctxn_h, c->p_kv, nb);
        dfl_ctx_store_kernel<<<dim3(nb, NL * 2 * NKV), 64, 0, s>>>(c->p_kv, c->knorms, d_pos, off, c->ring);
    }
}

} // namespace dflash
} // namespace squid

using squid::dflash::Ctx;

static int g_dump_seq = 0;
static bool stream_capturing(void* stream) {
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(static_cast<cudaStream_t>(stream), &st);
    return st != cudaStreamCaptureStatusNone;
}

static void dump_taps(const float* taps, int32_t B, const int32_t* d_pos, void* stream) {
    const char* dump = getenv("SQUID_DFLASH_DUMP");
    if (!dump || stream_capturing(stream)) return; // debug hooks sync and copy; never inside a graph
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    int32_t pos0 = 0;
    cudaMemcpy(&pos0, d_pos, 4, cudaMemcpyDeviceToHost);
    size_t n = (size_t)4 * B * squid::dflash::DH;
    float* hbuf = (float*)malloc(n * 4);
    cudaMemcpy(hbuf, taps, n * 4, cudaMemcpyDeviceToHost);
    char path[1024];
    snprintf(path, sizeof(path), "%s/%05d_taps_%06d_%d.bin", dump, g_dump_seq++, pos0, B);
    FILE* f = fopen(path, "wb");
    if (f) { fwrite(hbuf, 4, n, f); fclose(f); }
    free(hbuf);
}

extern "C" {

void* squid_dflash_create(const char* dir, uint8_t quant, uint8_t embed_norm, uint8_t codebook_vram) {
    Ctx* c = new Ctx();
    if (!squid::dflash::ctx_create(c, dir, quant > 2 ? 0 : quant, embed_norm != 0, codebook_vram != 0)) {
        if (c->vram) cudaFree(c->vram);
        if (c->pred_host) cudaFreeHost(c->pred_host);
        if (c->succ_host) cudaFreeHost(c->succ_host);
        delete c;
        return nullptr;
    }
    fprintf(stdout, "[dflash2] Loaded %s | weights=%s %.1f MiB | VRAM arena %.1f MiB | codebooks in %s | embed_norm=%d\n",
            dir, c->quant == 2 ? "nvfp4" : (c->quant == 1 ? "fp8" : "bf16"), c->weight_bytes / 1048576.0, c->vram_bytes / 1048576.0,
            codebook_vram ? "VRAM" : "pinned host RAM", embed_norm ? 1 : 0);
    fflush(stdout);
    return c;
}

void squid_dflash_destroy(void* h) {
    Ctx* c = static_cast<Ctx*>(h);
    if (!c) return;
    if (c->blas) cublasDestroy(c->blas);
    if (c->vram) cudaFree(c->vram);
    if (c->pred_host) cudaFreeHost(c->pred_host);
    if (c->succ_host) cudaFreeHost(c->succ_host);
    delete c;
}

float* squid_dflash_probs(void* h) { return static_cast<Ctx*>(h)->probs; }
int32_t squid_dflash_q_vocab() { return squid::dflash::V; }
void* squid_dflash_ring(void* h) { return static_cast<Ctx*>(h)->ring; }
size_t squid_dflash_ring_bytes(void* h) { return static_cast<Ctx*>(h)->ring_bytes; }

void squid_dflash_draft(void* h, const void* embed, const void* embed_norm, const squid::LinearWeight* unembed,
                        const int32_t* d_anchor, const int32_t* d_pos, const squid::SampleConfig* cfg,
                        int32_t* out_drafts, float* out_conf, float* out_q_sel, int32_t* out_is_delta, void* stream) {
    squid::dflash::draft(static_cast<Ctx*>(h), embed, embed_norm, *unembed, d_anchor, d_pos, cfg,
                         out_drafts, out_conf, out_q_sel, out_is_delta, static_cast<cudaStream_t>(stream));
    static int dbg = getenv("SQUID_DFLASH_DEBUG") ? atoi(getenv("SQUID_DFLASH_DEBUG")) : 0;
    if (dbg > 0 && !stream_capturing(stream)) {
        --dbg;
        Ctx* c = static_cast<Ctx*>(h);
        cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
        int32_t anc = 0, pos = 0, dr[7], ci[7 * 16];
        float cvv[7 * 16], hx[8];
        cudaMemcpy(&anc, d_anchor, 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(&pos, d_pos, 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(dr, out_drafts, 28, cudaMemcpyDeviceToHost);
        cudaMemcpy(ci, c->cand_i, sizeof(ci), cudaMemcpyDeviceToHost);
        cudaMemcpy(cvv, c->cand_v, sizeof(cvv), cudaMemcpyDeviceToHost);
        cudaMemcpy(hx, c->nrm + 2304, 32, cudaMemcpyDeviceToHost);
        fprintf(stdout, "[dflash2-dbg] pos=%d anchor=%d drafts=%d %d %d %d %d %d %d | top1=%d(%.2f) %d(%.2f) %d(%.2f) | h1[0..3]=%.3f %.3f %.3f %.3f\n",
                pos, anc, dr[0], dr[1], dr[2], dr[3], dr[4], dr[5], dr[6], ci[0], cvv[0], ci[16], cvv[16], ci[32], cvv[32], hx[0], hx[1], hx[2], hx[3]);
        fflush(stdout);
        const char* dump = getenv("SQUID_DFLASH_DUMP");
        if (dump) {
            static float hn[8 * 2304];
            cudaMemcpy(hn, c->nrm, sizeof(hn), cudaMemcpyDeviceToHost);
            char path[1024];
            snprintf(path, sizeof(path), "%s/%05d_draft_%06d_%d.bin", dump, g_dump_seq++, pos, anc);
            FILE* f = fopen(path, "wb");
            if (f) { fwrite(hn, 4, 8 * 2304, f); fwrite(dr, 4, 7, f); fclose(f); }
        }
    }
}

void squid_dflash_commit(void* h, const float* taps, int32_t B, const int32_t* d_pos, void* stream) {
    squid::dflash::commit(static_cast<Ctx*>(h), taps, B, d_pos, static_cast<cudaStream_t>(stream));
    dump_taps(taps, B, d_pos, stream);
}

void squid_dflash_commit_prefill(void* h, const float* taps, int32_t B, const int32_t* d_pos, void* stream) {
    squid::dflash::commit_prefill(static_cast<Ctx*>(h), taps, B, d_pos, static_cast<cudaStream_t>(stream));
    dump_taps(taps, B, d_pos, stream);
}

// Debug: copy one draft step's internals to host (final hidden rows, candidates, hp).
void squid_dflash_debug_copy(void* h, float* h_hidden8, float* h_cand_v, int32_t* h_cand_i, float* h_hp) {
    Ctx* c = static_cast<Ctx*>(h);
    cudaDeviceSynchronize();
    if (h_hidden8) cudaMemcpy(h_hidden8, c->nrm, (size_t)squid::dflash::BLK * squid::dflash::DH * 4, cudaMemcpyDeviceToHost);
    if (h_cand_v) cudaMemcpy(h_cand_v, c->cand_v, (size_t)squid::dflash::ND * squid::dflash::TOPK * 4, cudaMemcpyDeviceToHost);
    if (h_cand_i) cudaMemcpy(h_cand_i, c->cand_i, (size_t)squid::dflash::ND * squid::dflash::TOPK * 4, cudaMemcpyDeviceToHost);
    if (h_hp) cudaMemcpy(h_hp, c->hp, (size_t)squid::dflash::ND * squid::dflash::RANK * 4, cudaMemcpyDeviceToHost);
}

} // extern "C"
