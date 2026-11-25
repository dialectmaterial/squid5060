// On-Demand Streaming Vision Encoder for Inkling-2-Mini (23-layer ViT + 2x2 Merge Projector).
// Zero permanent VRAM overhead: streams 23 BF16 layers one at a time through borrowed GPU scratch
// (e.g. ExpertCache prefill tail) using Blackwell BF16 Tensor Core GEMMs + axial 2D RoPE + FlashAttention.
// Supports PNG (libpng16) and JPEG (libjpeg) with RGBA checkerboard compositing and bicubic downscale.

#include "vision.hpp"
#include "config.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <jpeglib.h>
#include <png.h>
#include <algorithm>
#include <cmath>
#include <csetjmp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace squid {
cublasHandle_t squid_get_shared_cublas(cudaStream_t stream);
namespace vision {

template <typename T>
struct RawBuf {
    T* ptr = nullptr;
    size_t len = 0;
    RawBuf() = default;
    explicit RawBuf(size_t n) { resize(n); }
    ~RawBuf() { clear(); }
    RawBuf(const RawBuf&) = delete;
    RawBuf& operator=(const RawBuf&) = delete;
    bool resize(size_t n) {
        clear();
        if (n == 0) return true;
        ptr = static_cast<T*>(std::malloc(n * sizeof(T)));
        if (!ptr) return false;
        len = n;
        return true;
    }
    void clear() {
        if (ptr) { std::free(ptr); ptr = nullptr; }
        len = 0;
    }
    void swap(RawBuf& o) {
        T* tp = ptr; ptr = o.ptr; o.ptr = tp;
        size_t tl = len; len = o.len; o.len = tl;
    }
    T* data() { return ptr; }
    const T* data() const { return ptr; }
    T& operator[](size_t i) { return ptr[i]; }
    const T& operator[](size_t i) const { return ptr[i]; }
};

constexpr int VIT_DIM = 1024;
constexpr int VIT_HEADS = 16;
constexpr int VIT_HD = 64;
constexpr int VIT_MLP = 4096;
constexpr int VIT_LAYERS = 23;
constexpr int VIT_PATCH = 20;
constexpr int VIT_CELL = 40; // 2x2 merge of 20x20 patches
constexpr int VIT_POS_GRID = 32;
constexpr float VIT_LN_EPS = 1e-5f;
constexpr float VIT_ROPE_THETA = 10000.0f;

// --- PNG / JPEG Decoders (CPU, zero permanent RAM) ---

struct PngMemReader {
    const uint8_t* data;
    size_t len;
    size_t pos;
};

static void png_read_mem(png_structp png_ptr, png_bytep out, png_size_t count) {
    auto* r = static_cast<PngMemReader*>(png_get_io_ptr(png_ptr));
    if (r->pos + count > r->len) {
        png_error(png_ptr, "PNG truncated");
        return;
    }
    std::memcpy(out, r->data + r->pos, count);
    r->pos += count;
}

static bool decode_png_rgba(const uint8_t* data, size_t len, int& w, int& h, RawBuf<uint8_t>& rgba) {
    if (len < 8 || png_sig_cmp(data, 0, 8) != 0) return false;
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) return false;
    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, nullptr, nullptr);
        return false;
    }
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        return false;
    }
    PngMemReader reader{data, len, 0};
    png_set_read_fn(png, &reader, png_read_mem);
    png_read_info(png, info);

    png_uint_32 pw = 0, ph = 0;
    int bit_depth = 0, color_type = 0;
    png_get_IHDR(png, info, &pw, &ph, &bit_depth, &color_type, nullptr, nullptr, nullptr);
    if (pw == 0 || ph == 0 || pw > 16384 || ph > 16384) {
        png_destroy_read_struct(&png, &info, nullptr);
        return false;
    }
    if (bit_depth == 16) png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA) png_set_gray_to_rgb(png);
    if (!(color_type & PNG_COLOR_MASK_ALPHA)) png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
    png_read_update_info(png, info);

    w = static_cast<int>(pw);
    h = static_cast<int>(ph);
    if (!rgba.resize((size_t)w * h * 4)) { png_destroy_read_struct(&png, &info, nullptr); return false; }
    RawBuf<png_bytep> rows(h);
    if (!rows.data()) { png_destroy_read_struct(&png, &info, nullptr); return false; }
    for (int y = 0; y < h; ++y) rows[y] = rgba.data() + (size_t)y * w * 4;
    png_read_image(png, rows.data());
    png_destroy_read_struct(&png, &info, nullptr);
    return true;
}

struct JpegErrMgr {
    jpeg_error_mgr pub;
    jmp_buf jbuf;
};

static void jpeg_err_exit(j_common_ptr cinfo) {
    auto* err = reinterpret_cast<JpegErrMgr*>(cinfo->err);
    longjmp(err->jbuf, 1);
}

static bool decode_jpeg_rgba(const uint8_t* data, size_t len, int& w, int& h, RawBuf<uint8_t>& rgba) {
    if (len < 3 || data[0] != 0xFF || data[1] != 0xD8) return false;
    jpeg_decompress_struct cinfo{};
    JpegErrMgr jerr{};
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = jpeg_err_exit;
    if (setjmp(jerr.jbuf)) {
        jpeg_destroy_decompress(&cinfo);
        return false;
    }
    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, data, static_cast<unsigned long>(len));
    if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
        jpeg_destroy_decompress(&cinfo);
        return false;
    }
    cinfo.out_color_space = JCS_RGB;
    jpeg_start_decompress(&cinfo);
    if (cinfo.output_width == 0 || cinfo.output_height == 0 ||
        cinfo.output_width > 16384 || cinfo.output_height > 16384) {
        jpeg_destroy_decompress(&cinfo);
        return false;
    }
    w = static_cast<int>(cinfo.output_width);
    h = static_cast<int>(cinfo.output_height);
    if (!rgba.resize((size_t)w * h * 4)) { jpeg_destroy_decompress(&cinfo); return false; }
    RawBuf<uint8_t> row_rgb((size_t)w * 3);
    if (!row_rgb.data()) { jpeg_destroy_decompress(&cinfo); return false; }
    while (cinfo.output_scanline < cinfo.output_height) {
        int y = static_cast<int>(cinfo.output_scanline);
        JSAMPROW ptr = row_rgb.data();
        jpeg_read_scanlines(&cinfo, &ptr, 1);
        uint8_t* dst = rgba.data() + (size_t)y * w * 4;
        for (int x = 0; x < w; ++x) {
            dst[4 * x + 0] = row_rgb[3 * x + 0];
            dst[4 * x + 1] = row_rgb[3 * x + 1];
            dst[4 * x + 2] = row_rgb[3 * x + 2];
            dst[4 * x + 3] = 255;
        }
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    return true;
}

static void composite_rgba_to_rgb(const RawBuf<uint8_t>& rgba, int w, int h, RawBuf<uint8_t>& rgb) {
    rgb.resize((size_t)w * h * 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const uint8_t* p = &rgba[((size_t)y * w + x) * 4];
            uint8_t* d = &rgb[((size_t)y * w + x) * 3];
            uint8_t a = p[3];
            if (a == 255) {
                d[0] = p[0]; d[1] = p[1]; d[2] = p[2];
            } else {
                uint8_t bg = ((((x >> 3) + (y >> 3)) & 1) != 0) ? 153 : 204;
                d[0] = static_cast<uint8_t>((p[0] * a + bg * (255 - a) + 127) / 255);
                d[1] = static_cast<uint8_t>((p[1] * a + bg * (255 - a) + 127) / 255);
                d[2] = static_cast<uint8_t>((p[2] * a + bg * (255 - a) + 127) / 255);
            }
        }
    }
}

static inline float cubic_weight(float t) {
    float a = std::fabs(t);
    if (a < 1.0f) return (1.5f * a - 2.5f) * a * a + 1.0f;
    if (a < 2.0f) return ((-0.5f * a + 2.5f) * a - 4.0f) * a + 2.0f;
    return 0.0f;
}

static void resize_rgb_bicubic(
    const RawBuf<uint8_t>& src, int sw, int sh,
    RawBuf<uint8_t>& dst, int dw, int dh) {
    // Separable bicubic resize (horizontal then vertical)
    RawBuf<float> tmp((size_t)sh * dw * 3);
    float sx_scale = static_cast<float>(sw) / dw;
    float sy_scale = static_cast<float>(sh) / dh;
    float supp_x = std::max(1.0f, sx_scale);
    float supp_y = std::max(1.0f, sy_scale);
    int rad_x = static_cast<int>(std::ceil(2.0f * supp_x));
    int rad_y = static_cast<int>(std::ceil(2.0f * supp_y));

    for (int y = 0; y < sh; ++y) {
        const uint8_t* srow = &src[(size_t)y * sw * 3];
        float* drow = &tmp[(size_t)y * dw * 3];
        for (int x = 0; x < dw; ++x) {
            float cx = (x + 0.5f) * sx_scale - 0.5f;
            int ix = static_cast<int>(std::floor(cx));
            float acc[3] = {0, 0, 0}, wsum = 0;
            for (int k = ix - rad_x; k <= ix + rad_x; ++k) {
                float w = cubic_weight((k - cx) / supp_x);
                if (w == 0.0f) continue;
                int ck = std::max(0, std::min(sw - 1, k));
                acc[0] += w * srow[3 * ck + 0];
                acc[1] += w * srow[3 * ck + 1];
                acc[2] += w * srow[3 * ck + 2];
                wsum += w;
            }
            float inv = (wsum > 0.0f) ? (1.0f / wsum) : 1.0f;
            drow[3 * x + 0] = acc[0] * inv;
            drow[3 * x + 1] = acc[1] * inv;
            drow[3 * x + 2] = acc[2] * inv;
        }
    }

    dst.resize((size_t)dh * dw * 3);
    for (int y = 0; y < dh; ++y) {
        float cy = (y + 0.5f) * sy_scale - 0.5f;
        int iy = static_cast<int>(std::floor(cy));
        uint8_t* drow = &dst[(size_t)y * dw * 3];
        for (int x = 0; x < dw; ++x) {
            float acc[3] = {0, 0, 0}, wsum = 0;
            for (int k = iy - rad_y; k <= iy + rad_y; ++k) {
                float w = cubic_weight((k - cy) / supp_y);
                if (w == 0.0f) continue;
                int ck = std::max(0, std::min(sh - 1, k));
                const float* sp = &tmp[((size_t)ck * dw + x) * 3];
                acc[0] += w * sp[0];
                acc[1] += w * sp[1];
                acc[2] += w * sp[2];
                wsum += w;
            }
            float inv = (wsum > 0.0f) ? (1.0f / wsum) : 1.0f;
            for (int c = 0; c < 3; ++c) {
                int v = static_cast<int>(std::lround(acc[c] * inv));
                drow[3 * x + c] = static_cast<uint8_t>(std::max(0, std::min(255, v)));
            }
        }
    }
}

static inline uint16_t f32_to_bf16_rn(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return static_cast<uint16_t>(u >> 16);
}

// --- ViT GPU Kernels ---

__global__ void vit_add_pos_and_cls_kernel(
    __nv_bfloat16* __restrict__ x,          // [N, 1024] (row 0 = CLS, rows 1..n = patches)
    const __nv_bfloat16* __restrict__ cls_emb,
    const __nv_bfloat16* __restrict__ cls_pos,
    const __nv_bfloat16* __restrict__ pos_emb, // [32*32, 1024]
    int cols, int rows) {
    int t = blockIdx.x;
    int gx = 2 * cols;
    int gy = 2 * rows;
    int n = gx * gy;
    if (t > n) return;

    __nv_bfloat16* row = x + (size_t)t * VIT_DIM;
    if (t == 0) {
        for (int d = threadIdx.x; d < VIT_DIM; d += blockDim.x) {
            float v = __bfloat162float(cls_emb[d]) + __bfloat162float(cls_pos[d]);
            row[d] = __float2bfloat16(v);
        }
        return;
    }

    int p = t - 1;
    int g = p >> 2;
    int r = p & 3;
    int py = (g / cols) * 2 + (r >> 1);
    int px = (g % cols) * 2 + (r & 1);

    float sy = (py + 0.5f) * (static_cast<float>(VIT_POS_GRID) / static_cast<float>(gy)) - 0.5f;
    float sx = (px + 0.5f) * (static_cast<float>(VIT_POS_GRID) / static_cast<float>(gx)) - 0.5f;
    sy = fminf(fmaxf(sy, 0.0f), static_cast<float>(VIT_POS_GRID - 1));
    sx = fminf(fmaxf(sx, 0.0f), static_cast<float>(VIT_POS_GRID - 1));

    int y0 = static_cast<int>(floorf(sy));
    int x0 = static_cast<int>(floorf(sx));
    int y1 = min(y0 + 1, VIT_POS_GRID - 1);
    int x1 = min(x0 + 1, VIT_POS_GRID - 1);
    float wy = sy - y0;
    float wx = sx - x0;

    float w00 = (1.0f - wy) * (1.0f - wx);
    float w01 = (1.0f - wy) * wx;
    float w10 = wy * (1.0f - wx);
    float w11 = wy * wx;

    const __nv_bfloat16* p00 = pos_emb + (size_t)(y0 * VIT_POS_GRID + x0) * VIT_DIM;
    const __nv_bfloat16* p01 = pos_emb + (size_t)(y0 * VIT_POS_GRID + x1) * VIT_DIM;
    const __nv_bfloat16* p10 = pos_emb + (size_t)(y1 * VIT_POS_GRID + x0) * VIT_DIM;
    const __nv_bfloat16* p11 = pos_emb + (size_t)(y1 * VIT_POS_GRID + x1) * VIT_DIM;

    for (int d = threadIdx.x; d < VIT_DIM; d += blockDim.x) {
        float interp = w00 * __bfloat162float(p00[d]) + w01 * __bfloat162float(p01[d])
                     + w10 * __bfloat162float(p10[d]) + w11 * __bfloat162float(p11[d]);
        row[d] = __float2bfloat16(__bfloat162float(row[d]) + interp);
    }
}

__global__ void vit_layernorm_kernel(
    const __nv_bfloat16* __restrict__ in,
    const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    int N) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;
    const __nv_bfloat16* src = in + (size_t)row * VIT_DIM;
    __nv_bfloat16* dst = out + (size_t)row * VIT_DIM;

    float sum = 0.0f;
    for (int d = tid; d < VIT_DIM; d += 256) sum += __bfloat162float(src[d]);
    __shared__ float sm[8];
    int lane = tid & 31, warp = tid >> 5;
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
    if (lane == 0) sm[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < 8) ? sm[lane] : 0.0f;
        for (int off = 4; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
        if (lane == 0) sm[0] = v * (1.0f / VIT_DIM);
    }
    __syncthreads();
    float mean = sm[0];

    float vsum = 0.0f;
    for (int d = tid; d < VIT_DIM; d += 256) {
        float diff = __bfloat162float(src[d]) - mean;
        vsum += diff * diff;
    }
    for (int off = 16; off > 0; off >>= 1) vsum += __shfl_xor_sync(0xffffffffu, vsum, off);
    if (lane == 0) sm[warp] = vsum;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < 8) ? sm[lane] : 0.0f;
        for (int off = 4; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
        if (lane == 0) sm[0] = rsqrtf(v * (1.0f / VIT_DIM) + VIT_LN_EPS);
    }
    __syncthreads();
    float inv = sm[0];

    for (int d = tid; d < VIT_DIM; d += 256) {
        float y = (__bfloat162float(src[d]) - mean) * inv * __bfloat162float(w[d]) + __bfloat162float(b[d]);
        dst[d] = __float2bfloat16(y);
    }
}

__global__ void vit_add_bias_rope_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ qb,
    const __nv_bfloat16* __restrict__ kb,
    const __nv_bfloat16* __restrict__ vb,
    int N, int cols) {
    int t = blockIdx.x;
    if (t >= N) return;
    int tid = threadIdx.x; // 0..511 (16 heads * 32 interleaved pairs = 512)
    int head = tid >> 5;   // 0..15
    int pair = tid & 31;   // 0..31 (0..15 = X, 16..31 = Y)
    int d0 = head * VIT_HD + 2 * pair;
    int d1 = d0 + 1;

    size_t base = (size_t)t * VIT_DIM;
    v[base + d0] = __float2bfloat16(__bfloat162float(v[base + d0]) + __bfloat162float(vb[d0]));
    v[base + d1] = __float2bfloat16(__bfloat162float(v[base + d1]) + __bfloat162float(vb[d1]));

    float q0 = __bfloat162float(q[base + d0]) + __bfloat162float(qb[d0]);
    float q1 = __bfloat162float(q[base + d1]) + __bfloat162float(qb[d1]);
    float k0 = __bfloat162float(k[base + d0]) + __bfloat162float(kb[d0]);
    float k1 = __bfloat162float(k[base + d1]) + __bfloat162float(kb[d1]);

    if (t > 0) {
        int p = t - 1;
        int g = p >> 2;
        int r = p & 3;
        int py = (g / cols) * 2 + (r >> 1);
        int px = (g % cols) * 2 + (r & 1);
        float coord = (pair < 16) ? static_cast<float>(px) : static_cast<float>(py);
        int j = (pair < 16) ? pair : (pair - 16);
        float inv_freq = 1.0f / powf(VIT_ROPE_THETA, (2.0f * j) / 32.0f);
        float ang = coord * inv_freq;
        float ca = cosf(ang), sa = sinf(ang);
        float nq0 = q0 * ca - q1 * sa;
        float nq1 = q0 * sa + q1 * ca;
        float nk0 = k0 * ca - k1 * sa;
        float nk1 = k0 * sa + k1 * ca;
        q0 = nq0; q1 = nq1; k0 = nk0; k1 = nk1;
    }

    q[base + d0] = __float2bfloat16(q0);
    q[base + d1] = __float2bfloat16(q1);
    k[base + d0] = __float2bfloat16(k0);
    k[base + d1] = __float2bfloat16(k1);
}

// Bidirectional Online-Softmax FlashAttention for ViT (grid=(N, 16), block=64, 1 thread per head_dim=64!).
// Zero O(N^2) attention matrix memory!
__global__ void __launch_bounds__(64)
vit_flash_attn_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    int N) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    int d = threadIdx.x; // 0..63
    if (t >= N) return;

    const float scale = 0.125f; // 1/sqrt(64)
    float qd = __bfloat162float(q[(size_t)t * VIT_DIM + h * VIT_HD + d]) * scale;

    float m_i = -1e30f;
    float l_i = 0.0f;
    float o_d = 0.0f;

    constexpr int TILE_K = 32;
    __shared__ __nv_bfloat16 s_k[TILE_K][VIT_HD];
    __shared__ __nv_bfloat16 s_v[TILE_K][VIT_HD];
    __shared__ float s_dot[2]; // 2 warps reduction

    int lane = d & 31;
    int warp = d >> 5;

    for (int k0 = 0; k0 < N; k0 += TILE_K) {
        int tile_len = min(TILE_K, N - k0);
        for (int j = 0; j < tile_len; ++j) {
            size_t off = (size_t)(k0 + j) * VIT_DIM + h * VIT_HD + d;
            s_k[j][d] = k[off];
            s_v[j][d] = v[off];
        }
        __syncthreads();

        for (int j = 0; j < tile_len; ++j) {
            float prod = qd * __bfloat162float(s_k[j][d]);
            for (int off = 16; off > 0; off >>= 1) prod += __shfl_xor_sync(0xffffffffu, prod, off);
            if (lane == 0) s_dot[warp] = prod;
            __syncthreads();
            float score = s_dot[0] + s_dot[1];

            float m_new = fmaxf(m_i, score);
            float alpha = expf(m_i - m_new);
            float p = expf(score - m_new);
            l_i = l_i * alpha + p;
            o_d = o_d * alpha + p * __bfloat162float(s_v[j][d]);
            m_i = m_new;
            __syncthreads();
        }
    }

    out[(size_t)t * VIT_DIM + h * VIT_HD + d] = __float2bfloat16(o_d / fmaxf(l_i, 1e-12f));
}

__global__ void vit_bias_layerscale_add_kernel(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ proj,
    const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ gamma,
    int N) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)N * VIT_DIM;
    if (idx >= total) return;
    int d = static_cast<int>(idx % VIT_DIM);
    float y = __bfloat162float(x[idx]) + __bfloat162float(gamma[d]) * (__bfloat162float(proj[idx]) + __bfloat162float(bias[d]));
    x[idx] = __float2bfloat16(y);
}

__global__ void vit_bias_gelu_kernel(
    __nv_bfloat16* __restrict__ h,
    const __nv_bfloat16* __restrict__ bias, // optional (nullptr for proj_in)
    int N, int D) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)N * D;
    if (idx >= total) return;
    float v = __bfloat162float(h[idx]);
    if (bias) v += __bfloat162float(bias[idx % D]);
    float g = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
    h[idx] = __float2bfloat16(g);
}

// Row-major C[M, N] = A[M, K] @ B[N, K]^T (all BF16 or C FP32)
static void vit_gemm_bf16(
    cublasHandle_t h, const void* A, const void* B, void* C,
    int M, int N, int K, bool out_f32) {
    float alpha = 1.0f, beta = 0.0f;
    cudaDataType_t ctype = out_f32 ? CUDA_R_32F : CUDA_R_16BF;
    cublasGemmEx(
        h, CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16BF, K,
        A, CUDA_R_16BF, K,
        &beta,
        C, ctype, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

static size_t align256(size_t x) { return (x + 255u) & ~size_t(255u); }

} // namespace vision
} // namespace squid

extern "C" int32_t squid_vision_encode_image(
    const uint8_t* img_bytes,
    size_t img_len,
    const CVisionWeightsHost* weights,
    void* scratch_dev,
    size_t scratch_bytes,
    int32_t max_tokens,
    float* out_embeds_host,
    int32_t* out_num_tokens,
    int32_t* out_cols,
    int32_t* out_rows,
    void* stream_raw) {
    using namespace squid::vision;
    if (!img_bytes || img_len < 8 || !weights || !out_embeds_host || !out_num_tokens) return -1;

    int w = 0, h = 0;
    RawBuf<uint8_t> rgba;
    bool ok = decode_png_rgba(img_bytes, img_len, w, h, rgba);
    if (!ok) ok = decode_jpeg_rgba(img_bytes, img_len, w, h, rgba);
    if (!ok || w <= 0 || h <= 0) return -2;

    RawBuf<uint8_t> rgb;
    composite_rgba_to_rgb(rgba, w, h, rgb);
    rgba.clear();

    int cap_cells = (max_tokens > 0) ? std::min(max_tokens, 1024) : 1024;
    int cols = (w + VIT_CELL - 1) / VIT_CELL;
    int rows = (h + VIT_CELL - 1) / VIT_CELL;
    if (cols * rows > cap_cells) {
        double scale = std::sqrt(static_cast<double>(cap_cells) / static_cast<double>(cols * rows));
        int nw = std::max(1, static_cast<int>(std::floor(w * scale)));
        int nh = std::max(1, static_cast<int>(std::floor(h * scale)));
        while (((nw + VIT_CELL - 1) / VIT_CELL) * ((nh + VIT_CELL - 1) / VIT_CELL) > cap_cells && (nw > VIT_CELL || nh > VIT_CELL)) {
            nw = std::max(1, nw - 8);
            nh = std::max(1, nh - 8);
        }
        RawBuf<uint8_t> resized;
        resize_rgb_bicubic(rgb, w, h, resized, nw, nh);
        rgb.swap(resized);
        w = nw;
        h = nh;
        cols = (w + VIT_CELL - 1) / VIT_CELL;
        rows = (h + VIT_CELL - 1) / VIT_CELL;
    }

    const int m = cols * rows;
    const int gx = 2 * cols;
    const int gy = 2 * rows;
    const int n = gx * gy;
    const int N = n + 1;

    // Extract patches in 2x2 merge order [n, 1200] BF16 (padded outside w,h with -1.0f)
    RawBuf<uint16_t> h_patches((size_t)n * 1200);
    if (!h_patches.data()) return -3;
    const uint16_t bf16_neg1 = f32_to_bf16_rn(-1.0f);
    for (int t = 0; t < n; ++t) {
        int g = t >> 2;
        int r = t & 3;
        int py = (g / cols) * 2 + (r >> 1);
        int px = (g % cols) * 2 + (r & 1);
        uint16_t* dst = &h_patches[(size_t)t * 1200];
        for (int dy = 0; dy < VIT_PATCH; ++dy) {
            int iy = py * VIT_PATCH + dy;
            for (int dx = 0; dx < VIT_PATCH; ++dx) {
                int ix = px * VIT_PATCH + dx;
                int p_off = (dy * VIT_PATCH + dx) * 3;
                if (ix < w && iy < h) {
                    const uint8_t* p = &rgb[((size_t)iy * w + ix) * 3];
                    dst[p_off + 0] = f32_to_bf16_rn(p[0] * (2.0f / 255.0f) - 1.0f);
                    dst[p_off + 1] = f32_to_bf16_rn(p[1] * (2.0f / 255.0f) - 1.0f);
                    dst[p_off + 2] = f32_to_bf16_rn(p[2] * (2.0f / 255.0f) - 1.0f);
                } else {
                    dst[p_off + 0] = bf16_neg1;
                    dst[p_off + 1] = bf16_neg1;
                    dst[p_off + 2] = bf16_neg1;
                }
            }
        }
    }

    // Compute exact GPU scratch layout (all carved from borrowed `scratch_dev` or temp cudaMalloc!)
    const size_t x_bytes       = align256((size_t)N * VIT_DIM * 2);
    const size_t mlp_h_bytes   = align256((size_t)N * VIT_MLP * 2);
    const size_t patches_bytes = align256((size_t)n * 1200 * 2);
    const size_t w_big_bytes   = align256((size_t)VIT_MLP * VIT_MLP * 2); // 32 MiB (holds proj_in or any layer weight!)
    const size_t w_aux_bytes   = align256((size_t)VIT_MLP * 2 * 16);      // biases, norms, gammas
    const size_t out_f32_bytes = align256((size_t)m * squid::HIDDEN * 4);
    const size_t total_need = 5 * x_bytes + std::max(mlp_h_bytes, std::max(patches_bytes, out_f32_bytes))
                            + w_big_bytes + w_aux_bytes;

    uint8_t* base = nullptr;
    bool temp_alloc = false;
    if (scratch_dev && scratch_bytes >= total_need) {
        base = static_cast<uint8_t*>(scratch_dev);
    } else {
        if (cudaMalloc(&base, total_need) != cudaSuccess) return -3;
        temp_alloc = true;
    }

    size_t off = 0;
    auto take = [&](size_t sz) -> void* {
        void* p = base + off;
        off += align256(sz);
        return p;
    };

    __nv_bfloat16* d_x      = static_cast<__nv_bfloat16*>(take(x_bytes));
    __nv_bfloat16* d_norm   = static_cast<__nv_bfloat16*>(take(x_bytes));
    __nv_bfloat16* d_q      = static_cast<__nv_bfloat16*>(take(x_bytes));
    __nv_bfloat16* d_k      = static_cast<__nv_bfloat16*>(take(x_bytes));
    __nv_bfloat16* d_v      = static_cast<__nv_bfloat16*>(take(x_bytes));
    void*          d_work   = take(std::max(mlp_h_bytes, std::max(patches_bytes, out_f32_bytes)));
    __nv_bfloat16* d_wbig   = static_cast<__nv_bfloat16*>(take(w_big_bytes));
    uint8_t*       d_aux    = static_cast<uint8_t*>(take(w_aux_bytes));

    cudaStream_t stream = static_cast<cudaStream_t>(stream_raw);
    cublasHandle_t cublas = squid::squid_get_shared_cublas(stream);

    // 1. Patch embedding: d_x[1..n] = patches[n, 1200] @ patch_embedding_w[1024, 1200]^T
    cudaMemcpyAsync(d_work, h_patches.data(), (size_t)n * 1200 * 2, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_wbig, weights->patch_embedding_w, (size_t)VIT_DIM * 1200 * 2, cudaMemcpyHostToDevice, stream);
    vit_gemm_bf16(cublas, d_work, d_wbig, d_x + VIT_DIM, n, VIT_DIM, 1200, false);

    // 2. Position embedding interpolation + CLS token
    __nv_bfloat16* d_cls_e = reinterpret_cast<__nv_bfloat16*>(d_aux);
    __nv_bfloat16* d_cls_p = d_cls_e + VIT_DIM;
    __nv_bfloat16* d_ln_w  = d_cls_p + VIT_DIM;
    __nv_bfloat16* d_ln_b  = d_ln_w  + VIT_DIM;
    cudaMemcpyAsync(d_cls_e, weights->class_embedding, VIT_DIM * 2, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_cls_p, weights->cls_position_embedding, VIT_DIM * 2, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_wbig, weights->position_embedding, (size_t)VIT_DIM * VIT_DIM * 2, cudaMemcpyHostToDevice, stream);
    vit_add_pos_and_cls_kernel<<<N, 256, 0, stream>>>(d_x, d_cls_e, d_cls_p, d_wbig, cols, rows);

    // 3. ln_pre (in-place on d_x)
    cudaMemcpyAsync(d_ln_w, weights->ln_pre_w, VIT_DIM * 2, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_ln_b, weights->ln_pre_b, VIT_DIM * 2, cudaMemcpyHostToDevice, stream);
    vit_layernorm_kernel<<<N, 256, 0, stream>>>(d_x, d_ln_w, d_ln_b, d_x, N);

    // Small aux pointers reused across all 23 layers
    __nv_bfloat16* d_b0 = reinterpret_cast<__nv_bfloat16*>(d_aux);           // 1024
    __nv_bfloat16* d_b1 = d_b0 + VIT_DIM;                                   // 1024
    __nv_bfloat16* d_b2 = d_b1 + VIT_DIM;                                   // 1024
    __nv_bfloat16* d_b3 = d_b2 + VIT_DIM;                                   // 1024
    __nv_bfloat16* d_b4 = d_b3 + VIT_DIM;                                   // 1024
    __nv_bfloat16* d_fc_b = d_b4 + VIT_DIM;                                 // 4096

    // 4. Stream 23 ViT layers one at a time
    for (int l = 0; l < VIT_LAYERS; ++l) {
        const CVisionLayerPtrs& L = weights->layers[l];

        // ln_1
        cudaMemcpyAsync(d_b0, L.ln1_w, VIT_DIM * 2, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_b1, L.ln1_b, VIT_DIM * 2, cudaMemcpyHostToDevice, stream);
        vit_layernorm_kernel<<<N, 256, 0, stream>>>(d_x, d_b0, d_b1, d_norm, N);

        // Q, K, V projections
        cudaMemcpyAsync(d_wbig, L.q_w, (size_t)VIT_DIM * VIT_DIM * 2, cudaMemcpyHostToDevice, stream);
        vit_gemm_bf16(cublas, d_norm, d_wbig, d_q, N, VIT_DIM, VIT_DIM, false);
        cudaMemcpyAsync(d_wbig, L.k_w, (size_t)VIT_DIM * VIT_DIM * 2, cudaMemcpyHostToDevice, stream);
        vit_gemm_bf16(cublas, d_norm, d_wbig, d_k, N, VIT_DIM, VIT_DIM, false);
        cudaMemcpyAsync(d_wbig, L.v_w, (size_t)VIT_DIM * VIT_DIM * 2, cudaMemcpyHostToDevice, stream);
        vit_gemm_bf16(cublas, d_norm, d_wbig, d_v, N, VIT_DIM, VIT_DIM, false);

        // Q/K/V bias + axial 2D RoPE on Q/K
        cudaMemcpyAsync(d_b0, L.q_b, VIT_DIM * 2, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_b1, L.k_b, VIT_DIM * 2, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_b2, L.v_b, VIT_DIM * 2, cudaMemcpyHostToDevice, stream);
        vit_add_bias_rope_kernel<<<N, 512, 0, stream>>>(d_q, d_k, d_v, d_b0, d_b1, d_b2, N, cols);

        // Bidirectional FlashAttention -> writes into d_norm
        vit_flash_attn_kernel<<<dim3(N, VIT_HEADS), 64, 0, stream>>>(d_q, d_k, d_v, d_norm, N);

        // out_proj + bias + ls_1.gamma residual add (uses d_q as temp proj output)
        cudaMemcpyAsync(d_wbig, L.o_w, (size_t)VIT_DIM * VIT_DIM * 2, cudaMemcpyHostToDevice, stream);
        vit_gemm_bf16(cublas, d_norm, d_wbig, d_q, N, VIT_DIM, VIT_DIM, false);
        cudaMemcpyAsync(d_b0, L.o_b, VIT_DIM * 2, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_b1, L.ls1_gamma, VIT_DIM * 2, cudaMemcpyHostToDevice, stream);
        vit_bias_layerscale_add_kernel<<<((size_t)N * VIT_DIM + 255) / 256, 256, 0, stream>>>(d_x, d_q, d_b0, d_b1, N);

        // ln_2
        cudaMemcpyAsync(d_b0, L.ln2_w, VIT_DIM * 2, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_b1, L.ln2_b, VIT_DIM * 2, cudaMemcpyHostToDevice, stream);
        vit_layernorm_kernel<<<N, 256, 0, stream>>>(d_x, d_b0, d_b1, d_norm, N);

        // MLP c_fc + bias + gelu_erf -> d_work [N, 4096]
        cudaMemcpyAsync(d_wbig, L.fc_w, (size_t)VIT_MLP * VIT_DIM * 2, cudaMemcpyHostToDevice, stream);
        vit_gemm_bf16(cublas, d_norm, d_wbig, d_work, N, VIT_MLP, VIT_DIM, false);
        cudaMemcpyAsync(d_fc_b, L.fc_b, VIT_MLP * 2, cudaMemcpyHostToDevice, stream);
        vit_bias_gelu_kernel<<<((size_t)N * VIT_MLP + 255) / 256, 256, 0, stream>>>(
            static_cast<__nv_bfloat16*>(d_work), d_fc_b, N, VIT_MLP);

        // MLP c_proj + bias + ls_2.gamma residual add
        cudaMemcpyAsync(d_wbig, L.proj_w, (size_t)VIT_DIM * VIT_MLP * 2, cudaMemcpyHostToDevice, stream);
        vit_gemm_bf16(cublas, d_work, d_wbig, d_q, N, VIT_DIM, VIT_MLP, false);
        cudaMemcpyAsync(d_b0, L.proj_b, VIT_DIM * 2, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_b1, L.ls2_gamma, VIT_DIM * 2, cudaMemcpyHostToDevice, stream);
        vit_bias_layerscale_add_kernel<<<((size_t)N * VIT_DIM + 255) / 256, 256, 0, stream>>>(d_x, d_q, d_b0, d_b1, N);
    }

    // 5. Projector (NO post-norm!): x[1..n] as [m, 4096] -> proj_in [4096, 4096] -> gelu_erf -> proj_out [2304, 4096] -> FP32!
    // Write proj_in output into d_work [m, 4096] BF16, then proj_out into d_norm.. (actually reuse d_q+d_k+d_v? No, check size!)
    // Size of out_f32 [m, 2304] F32 = m * 9216 B <= 9.4 MiB; size of 4 * x_bytes = 4 * N * 2048 B = 4 * (4*m+1) * 2048 B = 32768 * m B >> 9216 * m B!
    // So d_norm..d_v (contiguous in `base`!) easily holds out_f32 [m, 2304] F32!
    cudaMemcpyAsync(d_wbig, weights->proj_in_w, (size_t)VIT_MLP * VIT_MLP * 2, cudaMemcpyHostToDevice, stream);
    vit_gemm_bf16(cublas, d_x + VIT_DIM, d_wbig, d_work, m, VIT_MLP, VIT_MLP, false);
    vit_bias_gelu_kernel<<<((size_t)m * VIT_MLP + 255) / 256, 256, 0, stream>>>(
        static_cast<__nv_bfloat16*>(d_work), nullptr, m, VIT_MLP);

    float* d_out_f32 = reinterpret_cast<float*>(d_norm); // contiguous with d_q, d_k, d_v (>= 3.5x larger than out_f32!)
    cudaMemcpyAsync(d_wbig, weights->proj_out_w, (size_t)squid::HIDDEN * VIT_MLP * 2, cudaMemcpyHostToDevice, stream);
    vit_gemm_bf16(cublas, d_work, d_wbig, d_out_f32, m, squid::HIDDEN, VIT_MLP, true);

    cudaMemcpyAsync(out_embeds_host, d_out_f32, (size_t)m * squid::HIDDEN * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (temp_alloc) cudaFree(base);
    *out_num_tokens = m;
    if (out_cols) *out_cols = cols;
    if (out_rows) *out_rows = rows;
    return 0;
}
