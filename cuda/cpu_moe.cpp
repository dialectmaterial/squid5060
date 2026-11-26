// Persistent AVX2+FMA+F16C+AVX-VNNI+WAITPKG NVFP4 CPU MoE Worker Pool for Hybrid CPU+GPU Miss Execution.
// Runs in parallel with GPU VRAM-hit & PCIe-miss kernels via a zero-copy mapped doorbell.
// Phase 1 optimizations:
//  - WAITPKG (_umonitor/_umwait C0.2/C0.1) idle wait to cut CPU spin watts & preserve GPU Dynamic Boost clocks.
//  - Frequency-ranked P-core + E-core affinity pinning (excludes slow off-ring LP-E cores).
//  - Atomic row-tile work-stealing across heterogeneous P/E cores (eliminates straggler barrier stalls).
//  - AVX-VNNI (vpdpbusd) Q8-activation NVFP4 dot product (32 MACs/inst) with exact FP32 fallback.
//  - Vectorized AVX2 SwiGLU + software prefetching.
// Pure POSIX pthreads (zero libstdc++ ABI dependency; zero heap allocations on hot path).

#include "cpu_moe.hpp"

#include <immintrin.h>
#include <x86intrin.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

namespace squid {

namespace {

static float g_e4m3_half_lut[256];
static bool g_e4m3_init = false;

static void init_e4m3_lut() {
    if (g_e4m3_init) return;
    for (int v = 0; v < 256; ++v) {
        int sign = (v >> 7) & 1;
        int exp  = (v >> 3) & 0xF;
        int man  = v & 0x7;
        float val = 0.0f;
        if (exp == 0xF && man == 0x7) {
            val = 0.0f; // NaN -> safe 0
        } else if (exp == 0) {
            val = std::ldexp(static_cast<float>(man) / 8.0f, -6);
        } else {
            val = std::ldexp(1.0f + static_cast<float>(man) / 8.0f, exp - 7);
        }
        // Fold * 0.5f into the scale LUT so the 2x integer FP4 table {0,1,2,3,4,6,8,12} is exact!
        g_e4m3_half_lut[v] = (sign ? -val : val) * 0.5f;
    }
    g_e4m3_init = true;
}

enum class WaitMode : int {
    UmwaitC02 = 0, // _umonitor + _umwait(0, ...) deepest user-space power save (~150-300 ns wake)
    UmwaitC01 = 1, // _umonitor + _umwait(1, ...) shallow user-space power save (~50-100 ns wake)
    TpauseC02 = 2, // _tpause(0, ...) timed low-power pause
    Pause     = 3, // legacy _mm_pause()
};

static inline void wait_on_u32(const volatile uint32_t* addr, uint32_t expected, WaitMode mode, uint64_t ticks) {
#if defined(__WAITPKG__)
    if (mode == WaitMode::UmwaitC02 || mode == WaitMode::UmwaitC01) {
        _umonitor(const_cast<uint32_t*>(addr));
        if (*addr == expected) {
            const unsigned int ctrl = (mode == WaitMode::UmwaitC01) ? 1u : 0u;
            _umwait(ctrl, __rdtsc() + ticks);
        }
        return;
    }
    if (mode == WaitMode::TpauseC02) {
        _tpause(0u, __rdtsc() + ticks);
        return;
    }
#endif
    (void)addr;
    (void)expected;
    (void)mode;
    (void)ticks;
    _mm_pause();
}

static inline void wait_on_atomic(const std::atomic<uint32_t>* addr, uint32_t expected, WaitMode mode, uint64_t ticks) {
#if defined(__WAITPKG__)
    if (mode == WaitMode::UmwaitC02 || mode == WaitMode::UmwaitC01) {
        _umonitor(const_cast<std::atomic<uint32_t>*>(addr));
        if (addr->load(std::memory_order_acquire) == expected) {
            const unsigned int ctrl = (mode == WaitMode::UmwaitC01) ? 1u : 0u;
            _umwait(ctrl, __rdtsc() + ticks);
        }
        return;
    }
    if (mode == WaitMode::TpauseC02) {
        _tpause(0u, __rdtsc() + ticks);
        return;
    }
#endif
    (void)addr;
    (void)expected;
    (void)mode;
    (void)ticks;
    _mm_pause();
}

// Fast accurate 8-wide AVX2 expf (~3e-7 relative error across [-80, 80]).
static inline __m256 fast_exp256_ps(__m256 x) {
    const __m256 lo = _mm256_set1_ps(-80.0f);
    const __m256 hi = _mm256_set1_ps(80.0f);
    x = _mm256_max_ps(lo, _mm256_min_ps(hi, x));
    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    const __m256 ln2_hi = _mm256_set1_ps(0.693359375f);
    const __m256 ln2_lo = _mm256_set1_ps(-2.12194440e-4f);
    __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(fx, ln2_hi, x);
    r = _mm256_fnmadd_ps(fx, ln2_lo, r);

    __m256 p = _mm256_set1_ps(1.9875691500e-4f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1f));
    p = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), _mm256_add_ps(r, _mm256_set1_ps(1.0f)));

    __m256i emm0 = _mm256_add_epi32(_mm256_cvttps_epi32(fx), _mm256_set1_epi32(127));
    __m256 pow2n = _mm256_castsi256_ps(_mm256_slli_epi32(emm0, 23));
    return _mm256_mul_ps(p, pow2n);
}

static inline __m256 fast_tanh256_ps(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 two = _mm256_set1_ps(2.0f);
    // tanh(x) = sign(x) * (1 - 2 / (exp(2*|x|) + 1)) -> numerically stable for all x
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    __m256 sign = _mm256_and_ps(x, sign_mask);
    __m256 ax = _mm256_andnot_ps(sign_mask, x);
    __m256 e = fast_exp256_ps(_mm256_mul_ps(two, _mm256_min_ps(ax, _mm256_set1_ps(10.0f))));
    __m256 res = _mm256_sub_ps(one, _mm256_div_ps(two, _mm256_add_ps(e, one)));
    return _mm256_or_ps(res, sign);
}

static inline __m256 fast_sigmoid256_ps(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);
    __m256 em = fast_exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), x));
    return _mm256_div_ps(one, _mm256_add_ps(one, em));
}

static inline __m256 eval_swiglu_softcap_avx2(__m256 g, __m256 u) {
    const __m256 inv_gc = _mm256_set1_ps(1.0f / GATE_SOFTCAP);
    const __m256 gc = _mm256_set1_ps(GATE_SOFTCAP);
    const __m256 inv_uc = _mm256_set1_ps(1.0f / UP_SOFTCAP);
    const __m256 uc = _mm256_set1_ps(UP_SOFTCAP);
    __m256 g_cap = _mm256_mul_ps(fast_tanh256_ps(_mm256_mul_ps(g, inv_gc)), gc);
    __m256 u_cap = _mm256_mul_ps(fast_tanh256_ps(_mm256_mul_ps(u, inv_uc)), uc);
    __m256 sig = fast_sigmoid256_ps(g);
    return _mm256_mul_ps(_mm256_mul_ps(g_cap, sig), u_cap);
}

// Convert fp16 vector x_h[K] to FP32 in per-block-16 [even0..7, odd0..7] order.
static void unpack_fp16_to_f32_eo(const uint16_t* __restrict__ x_h, float* __restrict__ x_eo, int K) {
    const int n_blk = K >> 4;
    const __m256i perm = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    for (int b = 0; b < n_blk; ++b) {
        __m256 f0 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(x_h + (b << 4))));
        __m256 f1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(x_h + (b << 4) + 8)));
        __m256 ev_mix = _mm256_shuffle_ps(f0, f1, _MM_SHUFFLE(2, 0, 2, 0));
        __m256 od_mix = _mm256_shuffle_ps(f0, f1, _MM_SHUFFLE(3, 1, 3, 1));
        __m256 ev = _mm256_permutevar8x32_ps(ev_mix, perm);
        __m256 od = _mm256_permutevar8x32_ps(od_mix, perm);
        _mm256_storeu_ps(x_eo + (b << 4), ev);
        _mm256_storeu_ps(x_eo + (b << 4) + 8, od);
    }
}

// Quantize per-block-16 FP32 [even0..7, odd0..7] vector to Q8 int8 + per-block scale + -12*sum(q) bias
// for AVX-VNNI vpdpbusd with unsigned biased NVFP4 weights (w_i8 + 12 in [0..24]).
static inline void quantize_f32_eo_to_q8(
    const float* __restrict__ x_eo,
    int8_t* __restrict__ qx_eo,
    float* __restrict__ q_scale,
    int32_t* __restrict__ q_bias,
    int K) {
    const int n_blk = K >> 4;
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256 v127 = _mm256_set1_ps(127.0f);
    const __m256 vm127 = _mm256_set1_ps(-127.0f);
    for (int b = 0; b < n_blk; ++b) {
        __m256 ev = _mm256_loadu_ps(x_eo + (b << 4));
        __m256 od = _mm256_loadu_ps(x_eo + (b << 4) + 8);
        __m256 am = _mm256_max_ps(_mm256_and_ps(ev, abs_mask), _mm256_and_ps(od, abs_mask));
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(am), _mm256_extractf128_ps(am, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
        float max_abs = _mm_cvtss_f32(m4);
        if (max_abs <= 1e-30f) {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(qx_eo + (b << 4)), _mm_setzero_si128());
            q_scale[b] = 0.0f;
            q_bias[b] = 0;
            continue;
        }
        float sc = max_abs * (1.0f / 127.0f);
        float inv_sc = 127.0f / max_abs;
        __m256 vis = _mm256_set1_ps(inv_sc);
        __m256i ie = _mm256_cvtps_epi32(_mm256_max_ps(vm127, _mm256_min_ps(v127, _mm256_mul_ps(ev, vis))));
        __m256i io = _mm256_cvtps_epi32(_mm256_max_ps(vm127, _mm256_min_ps(v127, _mm256_mul_ps(od, vis))));

        // Sum all 16 int32s for q_bias = -12 * sum(q)
        __m256i s8 = _mm256_add_epi32(ie, io);
        __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(s8), _mm256_extracti128_si256(s8, 1));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, _MM_SHUFFLE(1, 0, 3, 2)));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, _MM_SHUFFLE(2, 3, 0, 1)));
        int32_t qsum = _mm_cvtsi128_si32(s4);

        // Pack 8 int32 evens + 8 int32 odds -> 16 int8 [ev0..7, od0..7]
        __m128i e16 = _mm_packs_epi32(_mm256_castsi256_si128(ie), _mm256_extracti128_si256(ie, 1));
        __m128i o16 = _mm_packs_epi32(_mm256_castsi256_si128(io), _mm256_extracti128_si256(io, 1));
        __m128i q16 = _mm_packs_epi16(e16, o16);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(qx_eo + (b << 4)), q16);
        q_scale[b] = sc;
        q_bias[b] = -12 * qsum;
    }
}

static inline float hsum128_ps(__m128 a) {
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    a = _mm_add_ss(a, _mm_shuffle_ps(a, a, 1));
    return _mm_cvtss_f32(a);
}

static inline float hsum256_ps(__m256 v) {
    return hsum128_ps(_mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1)));
}

#if defined(__AVXVNNI__)
// AVX-VNNI Q8-activation NVFP4 row dot product (B=1 specialization, 64 elements/iter = 4 blocks).
static inline float nvfp4_row_dot_vnni_b1(
    const uint8_t* __restrict__ w_row,
    const uint8_t* __restrict__ s_row,
    const int8_t* __restrict__ qx_eo,
    const float* __restrict__ q_scale,
    const int32_t* __restrict__ q_bias,
    int K) {
    const __m128i m0f = _mm_set1_epi8(0x0F);
    const __m128i lut_u8 = _mm_setr_epi8(12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0);
    const __m256i z256 = _mm256_setzero_si256();
    __m128 acc = _mm_setzero_ps();
    const int n_blk = K >> 4;
    for (int b = 0; b < n_blk; b += 4) {
        __m128i b01 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w_row + (b << 3)));
        __m128i lo01 = _mm_shuffle_epi8(lut_u8, _mm_and_si128(b01, m0f));
        __m128i hi01 = _mm_shuffle_epi8(lut_u8, _mm_and_si128(_mm_srli_epi16(b01, 4), m0f));
        __m256i w01 = _mm256_set_m128i(_mm_unpackhi_epi64(lo01, hi01), _mm_unpacklo_epi64(lo01, hi01));

        __m128i b23 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w_row + ((b + 2) << 3)));
        __m128i lo23 = _mm_shuffle_epi8(lut_u8, _mm_and_si128(b23, m0f));
        __m128i hi23 = _mm_shuffle_epi8(lut_u8, _mm_and_si128(_mm_srli_epi16(b23, 4), m0f));
        __m256i w23 = _mm256_set_m128i(_mm_unpackhi_epi64(lo23, hi23), _mm_unpacklo_epi64(lo23, hi23));

        __m256i x01 = _mm256_load_si256(reinterpret_cast<const __m256i*>(qx_eo + (b << 4)));
        __m256i x23 = _mm256_load_si256(reinterpret_cast<const __m256i*>(qx_eo + ((b + 2) << 4)));
        __m256i p01 = _mm256_dpbusd_avx_epi32(z256, w01, x01);
        __m256i p23 = _mm256_dpbusd_avx_epi32(z256, w23, x23);

        __m256i h0 = _mm256_hadd_epi32(p01, p23);
        __m256i h1 = _mm256_hadd_epi32(h0, h0);
        __m128i dots4 = _mm_unpacklo_epi32(_mm256_castsi256_si128(h1), _mm256_extracti128_si256(h1, 1));
        dots4 = _mm_add_epi32(dots4, _mm_load_si128(reinterpret_cast<const __m128i*>(q_bias + b)));

        __m128 wsc = _mm_setr_ps(
            g_e4m3_half_lut[s_row[b]],
            g_e4m3_half_lut[s_row[b + 1]],
            g_e4m3_half_lut[s_row[b + 2]],
            g_e4m3_half_lut[s_row[b + 3]]);
        __m128 sc = _mm_mul_ps(wsc, _mm_load_ps(q_scale + b));
        acc = _mm_fmadd_ps(_mm_cvtepi32_ps(dots4), sc, acc);
    }
    return hsum128_ps(acc);
}

// AVX-VNNI Q8-activation NVFP4 row dot product for 2..8 tokens sharing one expert (decodes weights ONCE!).
static inline void nvfp4_row_dot_vnni_multi(
    const uint8_t* __restrict__ w_row,
    const uint8_t* __restrict__ s_row,
    const int8_t* const* __restrict__ qx_ptrs,
    const float* const* __restrict__ qs_ptrs,
    const int32_t* const* __restrict__ qb_ptrs,
    int n_toks,
    int K,
    float* __restrict__ out_dots) {
    const __m128i m0f = _mm_set1_epi8(0x0F);
    const __m128i lut_u8 = _mm_setr_epi8(12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0);
    const __m256i z256 = _mm256_setzero_si256();
    __m128 acc[MTP_BLOCK_SIZE];
    for (int t = 0; t < n_toks; ++t) acc[t] = _mm_setzero_ps();

    const int n_blk = K >> 4;
    for (int b = 0; b < n_blk; b += 4) {
        __m128i b01 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w_row + (b << 3)));
        __m128i lo01 = _mm_shuffle_epi8(lut_u8, _mm_and_si128(b01, m0f));
        __m128i hi01 = _mm_shuffle_epi8(lut_u8, _mm_and_si128(_mm_srli_epi16(b01, 4), m0f));
        __m256i w01 = _mm256_set_m128i(_mm_unpackhi_epi64(lo01, hi01), _mm_unpacklo_epi64(lo01, hi01));

        __m128i b23 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w_row + ((b + 2) << 3)));
        __m128i lo23 = _mm_shuffle_epi8(lut_u8, _mm_and_si128(b23, m0f));
        __m128i hi23 = _mm_shuffle_epi8(lut_u8, _mm_and_si128(_mm_srli_epi16(b23, 4), m0f));
        __m256i w23 = _mm256_set_m128i(_mm_unpackhi_epi64(lo23, hi23), _mm_unpacklo_epi64(lo23, hi23));

        __m128 wsc = _mm_setr_ps(
            g_e4m3_half_lut[s_row[b]],
            g_e4m3_half_lut[s_row[b + 1]],
            g_e4m3_half_lut[s_row[b + 2]],
            g_e4m3_half_lut[s_row[b + 3]]);

        const int x_off = b << 4;
        for (int t = 0; t < n_toks; ++t) {
            __m256i x01 = _mm256_load_si256(reinterpret_cast<const __m256i*>(qx_ptrs[t] + x_off));
            __m256i x23 = _mm256_load_si256(reinterpret_cast<const __m256i*>(qx_ptrs[t] + x_off + 32));
            __m256i p01 = _mm256_dpbusd_avx_epi32(z256, w01, x01);
            __m256i p23 = _mm256_dpbusd_avx_epi32(z256, w23, x23);
            __m256i h0 = _mm256_hadd_epi32(p01, p23);
            __m256i h1 = _mm256_hadd_epi32(h0, h0);
            __m128i dots4 = _mm_unpacklo_epi32(_mm256_castsi256_si128(h1), _mm256_extracti128_si256(h1, 1));
            dots4 = _mm_add_epi32(dots4, _mm_load_si128(reinterpret_cast<const __m128i*>(qb_ptrs[t] + b)));
            __m128 sc = _mm_mul_ps(wsc, _mm_load_ps(qs_ptrs[t] + b));
            acc[t] = _mm_fmadd_ps(_mm_cvtepi32_ps(dots4), sc, acc[t]);
        }
    }
    for (int t = 0; t < n_toks; ++t) {
        out_dots[t] = hsum128_ps(acc[t]);
    }
}
#endif

// Exact FP32 AVX2+FMA NVFP4 row dot product (B=1 specialization).
static inline float nvfp4_row_dot_f32_b1(
    const uint8_t* __restrict__ w_row,
    const uint8_t* __restrict__ s_row,
    const float* __restrict__ x_eo,
    int K) {
    const __m128i m0f = _mm_set1_epi8(0x0F);
    const __m128i lut2x = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    const int n_blk = K >> 4;
    for (int b = 0; b < n_blk; b += 2) {
        __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w_row + (b << 3)));
        __m128i lo = _mm_shuffle_epi8(lut2x, _mm_and_si128(bytes, m0f));
        __m128i hi = _mm_shuffle_epi8(lut2x, _mm_and_si128(_mm_srli_epi16(bytes, 4), m0f));

        __m256 sc0 = _mm256_set1_ps(g_e4m3_half_lut[s_row[b]]);
        __m256 w0e = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
        __m256 w0o = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
        const float* xp0 = x_eo + (b << 4);
        __m256 p0 = _mm256_fmadd_ps(w0o, _mm256_loadu_ps(xp0 + 8), _mm256_mul_ps(w0e, _mm256_loadu_ps(xp0)));
        acc0 = _mm256_fmadd_ps(p0, sc0, acc0);

        __m256 sc1 = _mm256_set1_ps(g_e4m3_half_lut[s_row[b + 1]]);
        __m256 w1e = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
        __m256 w1o = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)));
        const float* xp1 = xp0 + 16;
        __m256 p1 = _mm256_fmadd_ps(w1o, _mm256_loadu_ps(xp1 + 8), _mm256_mul_ps(w1e, _mm256_loadu_ps(xp1)));
        acc1 = _mm256_fmadd_ps(p1, sc1, acc1);
    }
    return hsum256_ps(_mm256_add_ps(acc0, acc1));
}

// Exact FP32 AVX2+FMA NVFP4 row dot product for 2..8 active tokens sharing one expert.
static inline void nvfp4_row_dot_f32_multi(
    const uint8_t* __restrict__ w_row,
    const uint8_t* __restrict__ s_row,
    const float* const* __restrict__ x_ptrs,
    int n_toks,
    int K,
    float* __restrict__ out_dots) {
    const __m128i m0f = _mm_set1_epi8(0x0F);
    const __m128i lut2x = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    __m256 acc[MTP_BLOCK_SIZE];
    for (int t = 0; t < n_toks; ++t) acc[t] = _mm256_setzero_ps();

    const int n_blk = K >> 4;
    for (int b = 0; b < n_blk; b += 2) {
        __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w_row + (b << 3)));
        __m128i lo = _mm_shuffle_epi8(lut2x, _mm_and_si128(bytes, m0f));
        __m128i hi = _mm_shuffle_epi8(lut2x, _mm_and_si128(_mm_srli_epi16(bytes, 4), m0f));

        __m256 sc0 = _mm256_set1_ps(g_e4m3_half_lut[s_row[b]]);
        __m256 w0e = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), sc0);
        __m256 w0o = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)), sc0);

        __m256 sc1 = _mm256_set1_ps(g_e4m3_half_lut[s_row[b + 1]]);
        __m256 w1e = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))), sc1);
        __m256 w1o = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8))), sc1);

        const int x_off = b << 4;
        for (int t = 0; t < n_toks; ++t) {
            const float* xp = x_ptrs[t] + x_off;
            __m256 a = acc[t];
            a = _mm256_fmadd_ps(w0e, _mm256_loadu_ps(xp), a);
            a = _mm256_fmadd_ps(w0o, _mm256_loadu_ps(xp + 8), a);
            a = _mm256_fmadd_ps(w1e, _mm256_loadu_ps(xp + 16), a);
            a = _mm256_fmadd_ps(w1o, _mm256_loadu_ps(xp + 24), a);
            acc[t] = a;
        }
    }
    for (int t = 0; t < n_toks; ++t) {
        out_dots[t] = hsum256_ps(acc[t]);
    }
}

static inline int eo_index(int j) {
    int blk = j & ~15;
    int r = j & 15;
    return blk + ((r & 1) ? (8 + (r >> 1)) : (r >> 1));
}

struct CpuRank {
    int cpu;
    long max_khz;
};

static int build_ranked_cpus(int* out_cpus, int max_out, int* out_non_lpe) {
    long hc_l = sysconf(_SC_NPROCESSORS_ONLN);
    int hc = (hc_l > 0) ? std::min(static_cast<int>(hc_l), 64) : 8;
    CpuRank ranks[64];
    long best_khz = 0;
    for (int i = 0; i < hc; ++i) {
        long khz = 0;
        char path[128];
        std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
        if (FILE* f = std::fopen(path, "r")) {
            if (std::fscanf(f, "%ld", &khz) != 1) khz = 0;
            std::fclose(f);
        }
        ranks[i] = {i, khz};
        if (khz > best_khz) best_khz = khz;
    }
    std::stable_sort(ranks, ranks + hc, [](const CpuRank& a, const CpuRank& b) {
        return a.max_khz > b.max_khz;
    });
    int non_lpe = 0;
    int n = std::min(hc, max_out);
    for (int i = 0; i < n; ++i) {
        out_cpus[i] = ranks[i].cpu;
        // Treat cores >= 65% of max P-core freq as ring-resident P/E cores (excludes 2.5 GHz LP-E cores).
        if (best_khz == 0 || ranks[i].max_khz * 100 >= best_khz * 65) {
            non_lpe++;
        }
    }
    if (out_non_lpe) *out_non_lpe = std::max(1, non_lpe);
    return n;
}

static void pin_current_thread(int cpu_id) {
    if (cpu_id < 0) return;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu_id, &mask);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
}

constexpr int W13_TILE_ROWS = 32; // exact divisor of INTERMEDIATE = 1152 (36 tiles/task)
constexpr int W2_TILE_ROWS  = 32; // exact divisor of HIDDEN = 2304 (72 tiles/task)

} // namespace

struct CpuMoePool {
    CpuMoeBell* bell = nullptr;
    const uint8_t* host_experts[NUM_LAYERS] = {};
    const float* host_scale_w13 = nullptr; // [24 * 128]
    const float* host_scale_w2  = nullptr; // [24 * 128]
    int n_threads = 8;
    int cpu_map[32] = {};
    bool pin_threads = true;
    bool use_vnni = true;
    WaitMode wait_mode = WaitMode::UmwaitC02;
    uint64_t wait_ticks = 32000ULL;

    // Preallocated scratch (exact FP32 + Q8 VNNI in per-block-16 [even0..7, odd0..7] order; zero malloc!)
    alignas(64) float   x_eo[MTP_BLOCK_SIZE][HIDDEN];
    alignas(64) int8_t  qx_eo[MTP_BLOCK_SIZE][HIDDEN];
    alignas(64) float   qx_scale[MTP_BLOCK_SIZE][HIDDEN >> 4];
    alignas(64) int32_t qx_bias[MTP_BLOCK_SIZE][HIDDEN >> 4];

    alignas(64) float   act_eo[CPU_MOE_MAX_OUTS][INTERMEDIATE];
    alignas(64) int8_t  qact_eo[CPU_MOE_MAX_OUTS][INTERMEDIATE];
    alignas(64) float   qact_scale[CPU_MOE_MAX_OUTS][INTERMEDIATE >> 4];
    alignas(64) int32_t qact_bias[CPU_MOE_MAX_OUTS][INTERMEDIATE >> 4];

    // Spin-barrier + dynamic work-stealing + POSIX sleep state (zero libstdc++ ABI dependency!)
    pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t  cv = PTHREAD_COND_INITIALIZER;
    std::atomic<bool> stop{false};
    std::atomic<bool> awake{false};

    alignas(64) std::atomic<uint32_t> phase_gen{0};
    alignas(64) std::atomic<uint32_t> phase_done{0};
    alignas(64) std::atomic<int> work_cursor{0};
    int cur_phase = 0; // 1 = w13, 2 = w2
    int cur_layer = 0;
    int cur_n_tasks = 0;
    int cur_n_outs = 0;
    int cur_total_tiles = 0;

    struct WorkerArg {
        CpuMoePool* pool;
        int tid;
    };
    pthread_t workers[32] = {};
    WorkerArg worker_args[32] = {};
    pthread_t master{};

    void run_w13_worksteal() {
        constexpr int TILES_PER_TASK = INTERMEDIATE / W13_TILE_ROWS; // 36
        const int total_tiles = cur_total_tiles;
        const uint8_t* layer_bank = host_experts[cur_layer];
        const float* layer_s13 = host_scale_w13 + cur_layer * NUM_EXPERTS;
        constexpr size_t W_STRIDE = HIDDEN >> 1;
        constexpr size_t S_STRIDE = HIDDEN >> 4;
        const bool vnni = use_vnni;

        int cur_t = -1;
        const uint8_t* w_data = nullptr;
        const uint8_t* w_scales = nullptr;
        float es = 1.0f;
        int n_toks = 0;
        int out_base = 0;
        const float*   x_ptrs[MTP_BLOCK_SIZE];
        const int8_t*  qx_ptrs[MTP_BLOCK_SIZE];
        const float*   qs_ptrs[MTP_BLOCK_SIZE];
        const int32_t* qb_ptrs[MTP_BLOCK_SIZE];

        while (true) {
            int tile = work_cursor.fetch_add(1, std::memory_order_relaxed);
            if (tile >= total_tiles) break;

            int t = tile / TILES_PER_TASK;
            int j0 = (tile - t * TILES_PER_TASK) * W13_TILE_ROWS;
            if (t != cur_t) {
                cur_t = t;
                int exp_id = bell->task_expert[t];
                const uint8_t* ebase = layer_bank + (size_t)exp_id * EXPERT_SLOT_BYTES;
                w_data = ebase + EXPERT_W13_DATA_OFF;
                w_scales = ebase + EXPERT_W13_SCALE_OFF;
                es = layer_s13[exp_id];
                out_base = bell->task_out_base[t];
                uint8_t mask = bell->task_token_mask[t];
                n_toks = 0;
                for (int b = 0; b < MTP_BLOCK_SIZE; ++b) {
                    if (mask & (1u << b)) {
                        x_ptrs[n_toks] = x_eo[b];
                        qx_ptrs[n_toks] = qx_eo[b];
                        qs_ptrs[n_toks] = qx_scale[b];
                        qb_ptrs[n_toks] = qx_bias[b];
                        n_toks++;
                    }
                }
            }

            // Process 8 rows per inner batch so SwiGLU runs vectorized in AVX2!
            alignas(32) float gbuf[MTP_BLOCK_SIZE][8];
            alignas(32) float ubuf[MTP_BLOCK_SIZE][8];
            alignas(32) float abuf[8];
            for (int jj = 0; jj < W13_TILE_ROWS; jj += 8) {
                const int jb = j0 + jj;
                for (int r = 0; r < 8; ++r) {
                    const int j = jb + r;
                    const size_t rg = (size_t)(2 * j);
                    const size_t ru = rg + 1;
                    if (j + 1 < j0 + W13_TILE_ROWS) {
                        _mm_prefetch(reinterpret_cast<const char*>(w_data + (rg + 2) * W_STRIDE), _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char*>(w_data + (ru + 2) * W_STRIDE), _MM_HINT_T0);
                    }
                    if (n_toks == 1) {
#if defined(__AVXVNNI__)
                        if (vnni) {
                            gbuf[0][r] = nvfp4_row_dot_vnni_b1(w_data + rg * W_STRIDE, w_scales + rg * S_STRIDE, qx_ptrs[0], qs_ptrs[0], qb_ptrs[0], HIDDEN) * es;
                            ubuf[0][r] = nvfp4_row_dot_vnni_b1(w_data + ru * W_STRIDE, w_scales + ru * S_STRIDE, qx_ptrs[0], qs_ptrs[0], qb_ptrs[0], HIDDEN) * es;
                        } else
#endif
                        {
                            gbuf[0][r] = nvfp4_row_dot_f32_b1(w_data + rg * W_STRIDE, w_scales + rg * S_STRIDE, x_ptrs[0], HIDDEN) * es;
                            ubuf[0][r] = nvfp4_row_dot_f32_b1(w_data + ru * W_STRIDE, w_scales + ru * S_STRIDE, x_ptrs[0], HIDDEN) * es;
                        }
                    } else {
                        float gv[MTP_BLOCK_SIZE], uv[MTP_BLOCK_SIZE];
#if defined(__AVXVNNI__)
                        if (vnni) {
                            nvfp4_row_dot_vnni_multi(w_data + rg * W_STRIDE, w_scales + rg * S_STRIDE, qx_ptrs, qs_ptrs, qb_ptrs, n_toks, HIDDEN, gv);
                            nvfp4_row_dot_vnni_multi(w_data + ru * W_STRIDE, w_scales + ru * S_STRIDE, qx_ptrs, qs_ptrs, qb_ptrs, n_toks, HIDDEN, uv);
                        } else
#endif
                        {
                            nvfp4_row_dot_f32_multi(w_data + rg * W_STRIDE, w_scales + rg * S_STRIDE, x_ptrs, n_toks, HIDDEN, gv);
                            nvfp4_row_dot_f32_multi(w_data + ru * W_STRIDE, w_scales + ru * S_STRIDE, x_ptrs, n_toks, HIDDEN, uv);
                        }
                        for (int k = 0; k < n_toks; ++k) {
                            gbuf[k][r] = gv[k] * es;
                            ubuf[k][r] = uv[k] * es;
                        }
                    }
                }
                for (int k = 0; k < n_toks; ++k) {
                    __m256 av = eval_swiglu_softcap_avx2(_mm256_load_ps(gbuf[k]), _mm256_load_ps(ubuf[k]));
                    _mm256_store_ps(abuf, av);
                    float* dst = act_eo[out_base + k];
                    for (int r = 0; r < 8; ++r) {
                        dst[eo_index(jb + r)] = abuf[r];
                    }
                }
            }
            // If VNNI is active, quantize the 2 newly completed 16-element blocks (32 rows) in-place right here!
            if (vnni) {
                const int blk0 = j0 >> 4;
                for (int k = 0; k < n_toks; ++k) {
                    quantize_f32_eo_to_q8(
                        act_eo[out_base + k] + j0,
                        qact_eo[out_base + k] + j0,
                        qact_scale[out_base + k] + blk0,
                        qact_bias[out_base + k] + blk0,
                        W13_TILE_ROWS);
                }
            }
        }
    }

    void run_w2_worksteal() {
        constexpr int TILES_PER_TASK = HIDDEN / W2_TILE_ROWS; // 72
        const int total_tiles = cur_total_tiles;
        const uint8_t* layer_bank = host_experts[cur_layer];
        const float* layer_s2 = host_scale_w2 + cur_layer * NUM_EXPERTS;
        constexpr size_t W_STRIDE = INTERMEDIATE >> 1;
        constexpr size_t S_STRIDE = INTERMEDIATE >> 4;
        const bool vnni = use_vnni;

        int cur_t = -1;
        const uint8_t* w_data = nullptr;
        const uint8_t* w_scales = nullptr;
        float es = 1.0f;
        int n_toks = 0;
        int out_base = 0;
        const float*   a_ptrs[MTP_BLOCK_SIZE];
        const int8_t*  qa_ptrs[MTP_BLOCK_SIZE];
        const float*   qs_ptrs[MTP_BLOCK_SIZE];
        const int32_t* qb_ptrs[MTP_BLOCK_SIZE];
        float rw[MTP_BLOCK_SIZE];

        while (true) {
            int tile = work_cursor.fetch_add(1, std::memory_order_relaxed);
            if (tile >= total_tiles) break;

            int t = tile / TILES_PER_TASK;
            int j0 = (tile - t * TILES_PER_TASK) * W2_TILE_ROWS;
            if (t != cur_t) {
                cur_t = t;
                int exp_id = bell->task_expert[t];
                const uint8_t* ebase = layer_bank + (size_t)exp_id * EXPERT_SLOT_BYTES;
                w_data = ebase + EXPERT_W2_DATA_OFF;
                w_scales = ebase + EXPERT_W2_SCALE_OFF;
                es = layer_s2[exp_id];
                out_base = bell->task_out_base[t];
                uint8_t mask = bell->task_token_mask[t];
                n_toks = 0;
                for (int b = 0; b < MTP_BLOCK_SIZE; ++b) {
                    if (mask & (1u << b)) {
                        a_ptrs[n_toks] = act_eo[out_base + n_toks];
                        qa_ptrs[n_toks] = qact_eo[out_base + n_toks];
                        qs_ptrs[n_toks] = qact_scale[out_base + n_toks];
                        qb_ptrs[n_toks] = qact_bias[out_base + n_toks];
                        rw[n_toks] = bell->out_rw[out_base + n_toks] * es;
                        n_toks++;
                    }
                }
            }
            for (int r = 0; r < W2_TILE_ROWS; ++r) {
                const int j = j0 + r;
                if (r + 2 < W2_TILE_ROWS) {
                    _mm_prefetch(reinterpret_cast<const char*>(w_data + (size_t)(j + 2) * W_STRIDE), _MM_HINT_T0);
                }
                if (n_toks == 1) {
                    float d;
#if defined(__AVXVNNI__)
                    if (vnni) {
                        d = nvfp4_row_dot_vnni_b1(w_data + (size_t)j * W_STRIDE, w_scales + (size_t)j * S_STRIDE, qa_ptrs[0], qs_ptrs[0], qb_ptrs[0], INTERMEDIATE);
                    } else
#endif
                    {
                        d = nvfp4_row_dot_f32_b1(w_data + (size_t)j * W_STRIDE, w_scales + (size_t)j * S_STRIDE, a_ptrs[0], INTERMEDIATE);
                    }
                    bell->out_buf[out_base][j] = d * rw[0];
                } else {
                    float dv[MTP_BLOCK_SIZE];
#if defined(__AVXVNNI__)
                    if (vnni) {
                        nvfp4_row_dot_vnni_multi(w_data + (size_t)j * W_STRIDE, w_scales + (size_t)j * S_STRIDE, qa_ptrs, qs_ptrs, qb_ptrs, n_toks, INTERMEDIATE, dv);
                    } else
#endif
                    {
                        nvfp4_row_dot_f32_multi(w_data + (size_t)j * W_STRIDE, w_scales + (size_t)j * S_STRIDE, a_ptrs, n_toks, INTERMEDIATE, dv);
                    }
                    for (int k = 0; k < n_toks; ++k) {
                        bell->out_buf[out_base + k][j] = dv[k] * rw[k];
                    }
                }
            }
        }
    }

    void dispatch_phase(int phase) {
        cur_phase = phase;
        cur_total_tiles = cur_n_tasks * ((phase == 1) ? (INTERMEDIATE / W13_TILE_ROWS) : (HIDDEN / W2_TILE_ROWS));
        work_cursor.store(0, std::memory_order_relaxed);
        if (n_threads > 1) {
            phase_done.store(0, std::memory_order_relaxed);
            phase_gen.fetch_add(1, std::memory_order_release);
        }
        if (phase == 1) run_w13_worksteal();
        else            run_w2_worksteal();
        if (n_threads > 1) {
            const uint32_t want = static_cast<uint32_t>(n_threads - 1);
            uint32_t cur = phase_done.load(std::memory_order_acquire);
            while (cur < want) {
                wait_on_atomic(&phase_done, cur, wait_mode, wait_ticks);
                cur = phase_done.load(std::memory_order_acquire);
            }
        }
    }

    void wait_until_awake() {
        pthread_mutex_lock(&mu);
        while (!awake.load(std::memory_order_acquire) && !stop.load(std::memory_order_acquire)) {
            pthread_cond_wait(&cv, &mu);
        }
        pthread_mutex_unlock(&mu);
    }

    void worker_loop(int tid) {
        if (pin_threads) pin_current_thread(cpu_map[tid]);
        uint32_t seen_gen = 0;
        while (true) {
            if (!awake.load(std::memory_order_acquire) && !stop.load(std::memory_order_acquire)) {
                wait_until_awake();
            }
            if (stop.load(std::memory_order_acquire)) break;

            uint32_t g = phase_gen.load(std::memory_order_acquire);
            if (g != seen_gen) {
                seen_gen = g;
                if (cur_phase == 1) run_w13_worksteal();
                else if (cur_phase == 2) run_w2_worksteal();
                phase_done.fetch_add(1, std::memory_order_release);
            } else {
                wait_on_atomic(&phase_gen, seen_gen, wait_mode, wait_ticks);
            }
        }
    }

    void master_loop() {
        if (pin_threads) pin_current_thread(cpu_map[0]);
        uint32_t last_seq = 0;
        auto* req_atom = reinterpret_cast<std::atomic<uint32_t>*>(const_cast<uint32_t*>(&bell->req_seq));
        auto* ack_atom = reinterpret_cast<std::atomic<uint32_t>*>(const_cast<uint32_t*>(&bell->ack_seq));

        while (true) {
            if (!awake.load(std::memory_order_acquire) && !stop.load(std::memory_order_acquire)) {
                wait_until_awake();
            }
            if (stop.load(std::memory_order_acquire)) break;

            uint32_t req = req_atom->load(std::memory_order_acquire);
            if (req == last_seq) {
                wait_on_u32(&bell->req_seq, last_seq, wait_mode, wait_ticks);
                continue;
            }
            last_seq = req;

            cur_layer = bell->layer_idx;
            const int B = bell->batch_size;
            cur_n_tasks = bell->n_tasks;
            cur_n_outs = bell->n_outs;

            if (cur_n_tasks > 0 && cur_n_outs > 0 && cur_layer >= 0 && cur_layer < NUM_LAYERS) {
                for (int b = 0; b < B; ++b) {
                    unpack_fp16_to_f32_eo(bell->x_h + (size_t)b * HIDDEN, x_eo[b], HIDDEN);
                    if (use_vnni) {
                        quantize_f32_eo_to_q8(x_eo[b], qx_eo[b], qx_scale[b], qx_bias[b], HIDDEN);
                    }
                }
                dispatch_phase(1);
                dispatch_phase(2);
            }

            std::atomic_thread_fence(std::memory_order_release);
            ack_atom->store(req, std::memory_order_release);
        }
    }

    static void* worker_trampoline(void* arg) {
        auto* wa = static_cast<WorkerArg*>(arg);
        wa->pool->worker_loop(wa->tid);
        return nullptr;
    }

    static void* master_trampoline(void* arg) {
        static_cast<CpuMoePool*>(arg)->master_loop();
        return nullptr;
    }
};

} // namespace squid

extern "C" {

void* squid_cpu_moe_create(
    void* h_bell,
    const uint8_t* const* host_experts,
    const float* host_scale_w13,
    const float* host_scale_w2,
    int32_t num_threads) {
    squid::init_e4m3_lut();
    void* mem = nullptr;
    if (posix_memalign(&mem, 64, sizeof(squid::CpuMoePool)) != 0 || !mem) return nullptr;
    auto* p = new (mem) squid::CpuMoePool();
    p->bell = static_cast<squid::CpuMoeBell*>(h_bell);
    for (int l = 0; l < squid::NUM_LAYERS; ++l) {
        p->host_experts[l] = host_experts[l];
    }
    p->host_scale_w13 = host_scale_w13;
    p->host_scale_w2  = host_scale_w2;

    int non_lpe = 8;
    int total_cpus = squid::build_ranked_cpus(p->cpu_map, 32, &non_lpe);
    int nt = num_threads;
    if (nt <= 0) {
        // Default to up to 12 ring-resident P/E cores (excludes slow LP-E cores).
        nt = std::max(1, std::min(12, non_lpe));
    }
    if (nt > total_cpus) nt = total_cpus;
    if (nt > 32) nt = 32;
    p->n_threads = nt;

    if (const char* ev = std::getenv("SQUID_CPU_MOE_PIN")) {
        p->pin_threads = (ev[0] != '0');
    }
#if defined(__AVXVNNI__)
    p->use_vnni = true;
    if (const char* ev = std::getenv("SQUID_CPU_MOE_VNNI")) {
        p->use_vnni = (ev[0] != '0');
    }
#else
    p->use_vnni = false;
#endif
    if (const char* ev = std::getenv("SQUID_CPU_MOE_WAIT")) {
        if (std::strcmp(ev, "umwait0") == 0)      p->wait_mode = squid::WaitMode::UmwaitC02;
        else if (std::strcmp(ev, "umwait1") == 0) p->wait_mode = squid::WaitMode::UmwaitC01;
        else if (std::strcmp(ev, "tpause") == 0)  p->wait_mode = squid::WaitMode::TpauseC02;
        else if (std::strcmp(ev, "pause") == 0)   p->wait_mode = squid::WaitMode::Pause;
    }
    if (const char* ev = std::getenv("SQUID_CPU_MOE_WAIT_TICKS")) {
        long t = std::atol(ev);
        if (t >= 1000 && t <= 10000000) p->wait_ticks = static_cast<uint64_t>(t);
    }

    for (int t = 1; t < nt; ++t) {
        p->worker_args[t] = {p, t};
        pthread_create(&p->workers[t], nullptr, &squid::CpuMoePool::worker_trampoline, &p->worker_args[t]);
    }
    pthread_create(&p->master, nullptr, &squid::CpuMoePool::master_trampoline, p);
    return p;
}

void squid_cpu_moe_wake(void* pool) {
    if (!pool) return;
    auto* p = static_cast<squid::CpuMoePool*>(pool);
    if (p->awake.load(std::memory_order_acquire)) return;
    pthread_mutex_lock(&p->mu);
    p->awake.store(true, std::memory_order_release);
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);
}

void squid_cpu_moe_nap(void* pool) {
    if (!pool) return;
    auto* p = static_cast<squid::CpuMoePool*>(pool);
    p->awake.store(false, std::memory_order_release);
}

void squid_cpu_moe_destroy(void* pool) {
    if (!pool) return;
    auto* p = static_cast<squid::CpuMoePool*>(pool);
    if (p->bell) p->bell->abort_flag = 1;
    pthread_mutex_lock(&p->mu);
    p->stop.store(true, std::memory_order_release);
    p->awake.store(true, std::memory_order_release);
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);
    pthread_join(p->master, nullptr);
    for (int t = 1; t < p->n_threads; ++t) {
        pthread_join(p->workers[t], nullptr);
    }
    pthread_cond_destroy(&p->cv);
    pthread_mutex_destroy(&p->mu);
    p->~CpuMoePool();
    std::free(p);
}

} // extern "C"
