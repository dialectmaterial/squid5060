// Persistent Exact FP32 AVX2+FMA+F16C NVFP4 CPU MoE Worker Pool for Hybrid CPU+GPU Miss Execution.
// Runs in parallel with GPU VRAM-hit & PCIe-miss kernels via a zero-copy mapped doorbell.
// Exact FP32 dequantization (0.0000 KLD vs GPU, ~35 GB/s DDR5 bandwidth, 0% CPU when idle).
// Pure POSIX pthreads (zero libstdc++ ABI dependency; zero heap allocations on hot path).

#include "cpu_moe.hpp"

#include <immintrin.h>
#include <pthread.h>
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

static inline float fast_tanh_cpu(float x) {
    return std::tanh(x);
}

static inline float fast_sigmoid_cpu(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// Convert fp16 vector x_h[K] to FP32 packed in per-block-16 [even0..7, odd0..7] order
// matching the lo/hi nibble layout of NVFP4 bytes with zero permutes in the inner loop!
static void unpack_fp16_to_f32_eo(const uint16_t* __restrict__ x_h, float* __restrict__ x_eo, int K) {
    const int n_blk = K >> 4;
    for (int b = 0; b < n_blk; ++b) {
        __m256 f0 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(x_h + (b << 4))));
        __m256 f1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(x_h + (b << 4) + 8)));
        // f0 = [0,1,2,3 | 4,5,6,7], f1 = [8,9,10,11 | 12,13,14,15]
        // Shuffle within 128-bit lanes: evens [0,2,8,10 | 4,6,12,14], odds [1,3,9,11 | 5,7,13,15]
        __m256 ev_mix = _mm256_shuffle_ps(f0, f1, _MM_SHUFFLE(2, 0, 2, 0)); // [0,2,8,10 | 4,6,12,14]
        __m256 od_mix = _mm256_shuffle_ps(f0, f1, _MM_SHUFFLE(3, 1, 3, 1)); // [1,3,9,11 | 5,7,13,15]
        const __m256i perm = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
        __m256 ev = _mm256_permutevar8x32_ps(ev_mix, perm); // [0,2,4,6,8,10,12,14]
        __m256 od = _mm256_permutevar8x32_ps(od_mix, perm); // [1,3,5,7,9,11,13,15]
        _mm256_storeu_ps(x_eo + (b << 4), ev);
        _mm256_storeu_ps(x_eo + (b << 4) + 8, od);
    }
}

static inline float hsum256_ps(__m256 v) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    a = _mm_add_ss(a, _mm_shuffle_ps(a, a, 1));
    return _mm_cvtss_f32(a);
}

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

// Exact FP32 AVX2+FMA NVFP4 row dot product for 2..8 active tokens sharing one expert (loads weights ONCE!).
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

} // namespace

struct CpuMoePool {
    CpuMoeBell* bell = nullptr;
    const uint8_t* host_experts[NUM_LAYERS] = {};
    const float* host_scale_w13 = nullptr; // [24 * 128]
    const float* host_scale_w2  = nullptr; // [24 * 128]
    int n_threads = 8;

    // Preallocated scratch (exact FP32 in per-block-16 [even0..7, odd0..7] order; zero malloc!)
    alignas(64) float x_eo[MTP_BLOCK_SIZE][HIDDEN];
    alignas(64) float act_eo[CPU_MOE_MAX_OUTS][INTERMEDIATE];

    // Spin-barrier + POSIX sleep state (zero libstdc++ ABI dependency!)
    pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t  cv = PTHREAD_COND_INITIALIZER;
    std::atomic<bool> stop{false};
    std::atomic<bool> awake{false};

    alignas(64) std::atomic<uint32_t> phase_gen{0};
    alignas(64) std::atomic<uint32_t> phase_done{0};
    int cur_phase = 0; // 1 = w13, 2 = w2
    int cur_layer = 0;
    int cur_n_tasks = 0;
    int cur_n_outs = 0;

    struct WorkerArg {
        CpuMoePool* pool;
        int tid;
    };
    pthread_t workers[32] = {};
    WorkerArg worker_args[32] = {};
    pthread_t master{};

    void run_w13_slice(int tid) {
        const int n_tasks = cur_n_tasks;
        const int total_work = n_tasks * INTERMEDIATE;
        const int chunk = (total_work + n_threads - 1) / n_threads;
        const int w_start = tid * chunk;
        const int w_end = std::min(total_work, w_start + chunk);
        if (w_start >= w_end) return;

        const uint8_t* layer_bank = host_experts[cur_layer];
        const float* layer_s13 = host_scale_w13 + cur_layer * NUM_EXPERTS;
        constexpr size_t W_STRIDE = HIDDEN >> 1;
        constexpr size_t S_STRIDE = HIDDEN >> 4;

        int cur_t = -1;
        const uint8_t* w_data = nullptr;
        const uint8_t* w_scales = nullptr;
        float es = 1.0f;
        int n_toks = 0;
        int out_base = 0;
        const float* x_ptrs[MTP_BLOCK_SIZE];

        for (int w = w_start; w < w_end; ++w) {
            int t = w / INTERMEDIATE;
            int j = w - t * INTERMEDIATE;
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
                        n_toks++;
                    }
                }
            }
            const size_t rg = (size_t)(2 * j);
            const size_t ru = rg + 1;
            const int dst_j = eo_index(j);
            if (n_toks == 1) {
                float g = nvfp4_row_dot_f32_b1(w_data + rg * W_STRIDE, w_scales + rg * S_STRIDE, x_ptrs[0], HIDDEN) * es;
                float u = nvfp4_row_dot_f32_b1(w_data + ru * W_STRIDE, w_scales + ru * S_STRIDE, x_ptrs[0], HIDDEN) * es;
                float g_cap = fast_tanh_cpu(g * (1.0f / GATE_SOFTCAP)) * GATE_SOFTCAP;
                float u_cap = fast_tanh_cpu(u * (1.0f / UP_SOFTCAP)) * UP_SOFTCAP;
                act_eo[out_base][dst_j] = (g_cap * fast_sigmoid_cpu(g)) * u_cap;
            } else {
                float gv[MTP_BLOCK_SIZE], uv[MTP_BLOCK_SIZE];
                nvfp4_row_dot_f32_multi(w_data + rg * W_STRIDE, w_scales + rg * S_STRIDE, x_ptrs, n_toks, HIDDEN, gv);
                nvfp4_row_dot_f32_multi(w_data + ru * W_STRIDE, w_scales + ru * S_STRIDE, x_ptrs, n_toks, HIDDEN, uv);
                for (int k = 0; k < n_toks; ++k) {
                    float g = gv[k] * es;
                    float u = uv[k] * es;
                    float g_cap = fast_tanh_cpu(g * (1.0f / GATE_SOFTCAP)) * GATE_SOFTCAP;
                    float u_cap = fast_tanh_cpu(u * (1.0f / UP_SOFTCAP)) * UP_SOFTCAP;
                    act_eo[out_base + k][dst_j] = (g_cap * fast_sigmoid_cpu(g)) * u_cap;
                }
            }
        }
    }

    void run_w2_slice(int tid) {
        const int n_tasks = cur_n_tasks;
        const int total_work = n_tasks * HIDDEN;
        const int chunk = (total_work + n_threads - 1) / n_threads;
        const int w_start = tid * chunk;
        const int w_end = std::min(total_work, w_start + chunk);
        if (w_start >= w_end) return;

        const uint8_t* layer_bank = host_experts[cur_layer];
        const float* layer_s2 = host_scale_w2 + cur_layer * NUM_EXPERTS;
        constexpr size_t W_STRIDE = INTERMEDIATE >> 1;
        constexpr size_t S_STRIDE = INTERMEDIATE >> 4;

        int cur_t = -1;
        const uint8_t* w_data = nullptr;
        const uint8_t* w_scales = nullptr;
        float es = 1.0f;
        int n_toks = 0;
        int out_base = 0;
        const float* a_ptrs[MTP_BLOCK_SIZE];
        float rw[MTP_BLOCK_SIZE];

        for (int w = w_start; w < w_end; ++w) {
            int t = w / HIDDEN;
            int j = w - t * HIDDEN;
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
                        rw[n_toks] = bell->out_rw[out_base + n_toks] * es;
                        n_toks++;
                    }
                }
            }
            if (n_toks == 1) {
                float d = nvfp4_row_dot_f32_b1(w_data + (size_t)j * W_STRIDE, w_scales + (size_t)j * S_STRIDE, a_ptrs[0], INTERMEDIATE);
                bell->out_buf[out_base][j] = d * rw[0];
            } else {
                float dv[MTP_BLOCK_SIZE];
                nvfp4_row_dot_f32_multi(w_data + (size_t)j * W_STRIDE, w_scales + (size_t)j * S_STRIDE, a_ptrs, n_toks, INTERMEDIATE, dv);
                for (int k = 0; k < n_toks; ++k) {
                    bell->out_buf[out_base + k][j] = dv[k] * rw[k];
                }
            }
        }
    }

    void dispatch_phase(int phase) {
        cur_phase = phase;
        if (n_threads > 1) {
            phase_done.store(0, std::memory_order_relaxed);
            phase_gen.fetch_add(1, std::memory_order_release);
        }
        if (phase == 1) run_w13_slice(0);
        else            run_w2_slice(0);
        if (n_threads > 1) {
            const uint32_t want = static_cast<uint32_t>(n_threads - 1);
            while (phase_done.load(std::memory_order_acquire) < want) {
                _mm_pause();
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
        uint32_t seen_gen = 0;
        while (true) {
            if (!awake.load(std::memory_order_acquire) && !stop.load(std::memory_order_acquire)) {
                wait_until_awake();
            }
            if (stop.load(std::memory_order_acquire)) break;

            uint32_t g = phase_gen.load(std::memory_order_acquire);
            if (g != seen_gen) {
                seen_gen = g;
                if (cur_phase == 1) run_w13_slice(tid);
                else if (cur_phase == 2) run_w2_slice(tid);
                phase_done.fetch_add(1, std::memory_order_release);
            } else {
                _mm_pause();
            }
        }
    }

    void master_loop() {
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
                _mm_pause();
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
    int nt = num_threads;
    if (nt <= 0) {
        long hc_l = sysconf(_SC_NPROCESSORS_ONLN);
        int hc = (hc_l > 0) ? static_cast<int>(hc_l) : 8;
        nt = std::max(1, std::min(10, hc > 4 ? (hc * 5) / 8 : hc));
    }
    if (nt > 32) nt = 32;
    p->n_threads = nt;

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
