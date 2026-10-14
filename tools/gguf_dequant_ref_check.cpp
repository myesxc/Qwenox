// gguf_dequant_ref_check.cpp — bit-exact check of gguf::dequant_row (IQ3_S, Q6_K)
// against llama.cpp ggml-quants.c reference functions copied verbatim below
// (llama.cpp commit 85c5522, MIT License, Copyright (c) 2023-2026 The ggml authors).
// Build: g++ -O1 -std=c++17 -I src tools/gguf_dequant_ref_check.cpp -o /tmp/gdrc && /tmp/gdrc
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <random>
#include "gguf.h"
#define QK_K 256
#define GGML_RESTRICT
typedef uint16_t ggml_half;
#define GGML_FP16_TO_FP32(x) gguf::h2f(x)
static const uint32_t iq3s_grid[512] = GGUF_IQ3S_GRID_INIT;
static const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};
#define IQ3S_N_SCALE QK_K/64
typedef struct {
    ggml_half d;
    uint8_t qs[QK_K/4];
    uint8_t qh[QK_K/32];
    uint8_t signs[QK_K/8];
    uint8_t scales[IQ3S_N_SCALE];
} block_iq3_s;
typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    ggml_half d;             // super-block scale
} block_q6_K;
void dequantize_row_iq3_s(const block_iq3_s * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * signs = x[i].signs;

        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const float db1 = d * (1 + 2*(x[i].scales[ib32/2] & 0xf));
            const float db2 = d * (1 + 2*(x[i].scales[ib32/2] >>  4));
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[0] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[0] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = db1 * grid1[j] * (signs[l] & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    y[j+4] = db1 * grid2[j] * (signs[l] & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[1] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[1] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = db2 * grid1[j] * (signs[l] & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    y[j+4] = db2 * grid2[j] * (signs[l] & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
}
void dequantize_row_q6_K(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * GGML_RESTRICT ql = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const int8_t  * GGML_RESTRICT sc = x[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}
int main() {
  std::mt19937 rng(1);
  for (int t = 0; t < 2; t++) {
    const int bs = t ? 210 : 110, nb = 4096;
    std::vector<uint8_t> buf(bs * nb);
    for (auto& b : buf) b = rng();
    for (int i = 0; i < nb; i++) {  // finite fp16 d in [2^-10, 2^-6)
      uint16_t h = (uint16_t)(((5 + rng() % 4) << 10) | (rng() & 0x3ff)) | (rng() & 1 ? 0x8000 : 0);
      memcpy(&buf[i * bs + (t ? 208 : 0)], &h, 2);
    }
    std::vector<float> a(256 * nb), b(256 * nb);
    if (t) dequantize_row_q6_K((const block_q6_K*)buf.data(), a.data(), 256 * nb);
    else dequantize_row_iq3_s((const block_iq3_s*)buf.data(), a.data(), 256 * nb);
    gguf::dequant_row(t ? gguf::Q6_K : gguf::IQ3_S, buf.data(), b.data(), 256 * nb);
    long bad = 0; double mx = 0;
    for (size_t i = 0; i < a.size(); i++) { if (a[i] != b[i]) bad++; mx = fmax(mx, fabs(a[i])); }
    printf("%s: %zu elems, mismatches %ld, max|y| %.4g -> %s\n", t ? "Q6_K" : "IQ3_S", a.size(), bad, mx, bad ? "FAIL" : "PASS");
  }
}
