// q8_wmma_test.cu — dense WMMA q8g32 GEMV (k_q8g32_wmma, GGUF Q8_0 dense) unit
// test + microbench.   bash tools/ht_wmma_test.sh q8   (ends Q8_WMMA TEST: PASS/FAIL)
// Per GGUF dense shape (rows x cols as the engine stores them):
//   1. accuracy vs a CPU double reference on the exact q8g32 weights (x rounded
//      the same way the kernel sees it: bf16 inputs exact, f32 inputs -> f16 is
//      the kernel's only rounding), P = 1..48;
//   2. row invariance: the same x row gives bit-identical y at any P and any
//      position in the batch (shifted windows, reversed batch), and a row slice
//      (gemv row_off / nrows) equals the same rows of the full launch;
//   3. determinism: repeated launches bit-identical;
//   4. timing: old <=8-row q8g32 multi-row gemv (chunked for P > 8) vs new,
//      weights rotated over copies larger than the 32 MB MALL (DRAM-bound like decode).
// qwenox.cpp's main() is renamed so this file can link its own.
#define main qwenox_real_main
#include "qwenox.cpp"
#undef main

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#define TCK(x)                                                                        \
  do {                                                                                \
    hipError_t e_ = (x);                                                              \
    if (e_ != hipSuccess) {                                                           \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__,     \
              __LINE__);                                                              \
      exit(1);                                                                        \
    }                                                                                 \
  } while (0)

static std::mt19937 rng(20261013);
static int fails = 0;

struct Shape {
  const char* name;
  int R, C;
  bool xbf;
  int per_step;  // calls per decode step (48-layer model)
};
static const Shape kShapes[] = {
    {"gdn.qkv", 10240, 2560, false, 36}, {"gdn.z", 6144, 2560, false, 36},
    {"gdn.out", 2560, 6144, false, 36},  {"qsa.q", 12288, 2560, false, 12},
    {"qsa.o", 2560, 6144, false, 12},    {"qsa.k", 512, 2560, false, 24},
    {"se.gu", 640, 2560, false, 96},     {"se.down", 2560, 640, false, 48},
    {"hc.down", 320, 10240, true, 97},   {"hc.up", 10240, 320, false, 97},
    {"lm_head", 248320, 2560, false, 1},
};

static void run_old(const WView& w, const void* x, bool xbf, float* y, int P) {
  for (int off = 0; off < P; off += 8) {
    const int pn = std::min(8, P - off);
    const void* xp = xbf ? (const void*)((const uint16_t*)x + (size_t)off * w.cols)
                         : (const void*)((const float*)x + (size_t)off * w.cols);
    if (!gemv_multi_q4cp_le8(w, xp, y + (size_t)off * w.rows, pn, w.cols, xbf))
      throw std::runtime_error("q8g32 gemv_mr");
  }
}
static void run_new(const WView& w, const void* x, bool xbf, float* y, int P,
                    uint64_t row_off = 0, uint64_t nrows = 0) {
  if (!q8g32_wmma(w, x, xbf, w.cols, y, row_off || nrows ? nrows : w.rows, P, row_off,
                  nrows ? nrows : w.rows))
    throw std::runtime_error("q8g32_wmma refused");
}
static std::vector<float> dget(const float* d, size_t n) {
  std::vector<float> h(n);
  TCK(hipMemcpy(h.data(), d, n * 4, hipMemcpyDeviceToHost));
  return h;
}
static uint16_t f2bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}
static float bf2f(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
static float h2f_(uint16_t h) {
  __half x;
  memcpy(&x, &h, 2);
  return __half2float(x);
}

int main() {
  TCK(hipMalloc(&g_q8w_part, kQ8wPartAlloc * 4));
  hipEvent_t e0, e1;
  TCK(hipEventCreate(&e0));
  TCK(hipEventCreate(&e1));
  const int MAXP = 48;
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_real_distribution<float> ud(0.f, 1.f);
  double step_old[3] = {0, 0, 0}, step_new[3] = {0, 0, 0};
  printf("%-8s %6s %5s | %9s %9s | %s\n", "shape", "R", "C", "maxerr/rms", "meanerr", "checks");
  for (const Shape& sh : kShapes) {
    const int R = sh.R, C = sh.C, G = C / 32;
    const size_t wbytes = (size_t)R * C + (size_t)R * G * 2;
    const int ncopy = (int)std::max<size_t>(1, std::min<size_t>(16, (256ull << 20) / wbytes));
    std::vector<uint8_t> hw(wbytes);
    int8_t* hq = (int8_t*)hw.data();
    uint16_t* hs = (uint16_t*)(hw.data() + (size_t)R * C);
    for (size_t i = 0; i < (size_t)R * C; i++) hq[i] = (int8_t)((int)(rng() % 255) - 127);
    for (size_t i = 0; i < (size_t)R * G; i++) {
      const __half h = __float2half((0.0005f + 0.002f * ud(rng)) / sqrtf((float)C / 32.f));
      memcpy(&hs[i], &h, 2);
    }
    uint8_t* dw;
    TCK(hipMalloc(&dw, wbytes * ncopy));
    for (int c = 0; c < ncopy; c++) TCK(hipMemcpy(dw + wbytes * c, hw.data(), wbytes, hipMemcpyHostToDevice));
    WView w{8, dw, (uint64_t)R, (uint64_t)C, 0, nullptr, nullptr};
    std::vector<float> hx((size_t)MAXP * C);
    for (auto& v : hx) v = nd(rng) * (ud(rng) < 0.002f ? 30.f : 1.f);
    std::vector<uint16_t> hxb(hx.size());
    for (size_t i = 0; i < hx.size(); i++) {
      hxb[i] = f2bf(hx[i]);
      if (sh.xbf) hx[i] = bf2f(hxb[i]);
    }
    const size_t esz = sh.xbf ? 2 : 4;
    std::vector<uint8_t> xin((size_t)MAXP * C * esz), xrev(xin.size());
    if (sh.xbf) memcpy(xin.data(), hxb.data(), xin.size());
    else memcpy(xin.data(), hx.data(), xin.size());
    for (int p = 0; p < MAXP; p++)
      memcpy(&xrev[(size_t)p * C * esz], &xin[(size_t)(MAXP - 1 - p) * C * esz], C * esz);
    uint8_t *dx, *dxr;
    float *dyo, *dyn, *dy2;
    TCK(hipMalloc(&dx, xin.size()));
    TCK(hipMalloc(&dxr, xin.size()));
    TCK(hipMemcpy(dx, xin.data(), xin.size(), hipMemcpyHostToDevice));
    TCK(hipMemcpy(dxr, xrev.data(), xin.size(), hipMemcpyHostToDevice));
    TCK(hipMalloc(&dyo, (size_t)MAXP * R * 4));
    TCK(hipMalloc(&dyn, (size_t)MAXP * R * 4));
    TCK(hipMalloc(&dy2, (size_t)MAXP * R * 4));
    auto xrow = [&](uint8_t* base, int off) { return (void*)(base + (size_t)off * C * esz); };

    // 1. accuracy vs CPU double (rows 0, 17, 47 of a P = 48 launch; x as the kernel sees it)
    run_new(w, dx, sh.xbf, dyn, MAXP);
    TCK(hipDeviceSynchronize());
    auto yref = dget(dyn, (size_t)MAXP * R);
    double maxe = 0, sume = 0;
    long cnt = 0;
    bool finite = true;
    for (float v : yref) finite &= std::isfinite(v);
    const int rstep = R > 20000 ? 97 : 1;  // lm_head: sample rows
    for (int p : {0, 17, 47}) {
      std::vector<double> xs(C);
      for (int c = 0; c < C; c++)
        xs[c] = sh.xbf ? (double)hx[(size_t)p * C + c]
                       : (double)__half2float(__float2half(hx[(size_t)p * C + c]));
      std::vector<double> ref;
      double ss = 0;
      for (int r = 0; r < R; r += rstep) {
        double acc = 0;
        for (int g = 0; g < G; g++) {
          double d = 0;
          for (int k = 0; k < 32; k++) d += (double)hq[(size_t)r * C + g * 32 + k] * xs[g * 32 + k];
          acc += (double)h2f_(hs[(size_t)r * G + g]) * d;
        }
        ref.push_back(acc);
        ss += acc * acc;
      }
      const double rms = sqrt(ss / ref.size()) + 1e-30;
      int i = 0;
      for (int r = 0; r < R; r += rstep, i++) {
        const double e = fabs((double)yref[(size_t)p * R + r] - ref[i]) / rms;
        maxe = std::max(maxe, e);
        sume += e;
        cnt++;
      }
    }
    const double meane = sume / cnt;
    const bool acc_ok = finite && maxe < 2e-4 && meane < 2e-5;

    // 2/3. row invariance, row slices, determinism
    long bad = 0;
    for (auto [P, off] : std::vector<std::pair<int, int>>{
             {1, 0}, {1, 13}, {1, 47}, {2, 9}, {3, 5}, {5, 11}, {8, 40}, {9, 2}, {16, 32},
             {17, 7}, {24, 24}, {31, 17}, {33, 1}, {40, 8}, {47, 1}}) {
      run_new(w, xrow(dx, off), sh.xbf, dy2, P);
      TCK(hipDeviceSynchronize());
      auto y = dget(dy2, (size_t)P * R);
      if (memcmp(y.data(), &yref[(size_t)off * R], (size_t)P * R * 4)) bad++;
    }
    run_new(w, dxr, sh.xbf, dy2, MAXP);
    TCK(hipDeviceSynchronize());
    {
      auto y = dget(dy2, (size_t)MAXP * R);
      for (int p = 0; p < MAXP; p++)
        if (memcmp(&y[(size_t)p * R], &yref[(size_t)(MAXP - 1 - p) * R], (size_t)R * 4)) {
          bad++;
          break;
        }
    }
    if (R >= 64) {  // row slice [R/4 rounded to 16, + R/2) at P = 1 and 5
      const uint64_t r0 = (uint64_t)(R / 4) / 16 * 16, nr = (uint64_t)R / 2;
      for (int P : {1, 5}) {
        run_new(w, dx, sh.xbf, dy2, P, r0, nr);
        TCK(hipDeviceSynchronize());
        auto y = dget(dy2, (size_t)P * nr);
        for (int p = 0; p < P; p++)
          if (memcmp(&y[(size_t)p * nr], &yref[(size_t)p * R + r0], nr * 4)) {
            bad++;
            break;
          }
      }
    }
    for (int r = 0; r < 3; r++) {
      run_new(w, dx, sh.xbf, dy2, MAXP);
      TCK(hipDeviceSynchronize());
      auto y = dget(dy2, (size_t)MAXP * R);
      if (memcmp(y.data(), yref.data(), y.size() * 4)) bad++;
    }
    const bool inv_ok = bad == 0;
    int kw, gps, ns;
    q8w_config(R, C, kw, gps, ns);
    printf("%-8s %6d %5d | %9.2e %9.2e | inv %s acc %s (KW %d gps %d ns %d)\n", sh.name, R, C,
           maxe, meane, inv_ok ? "ok" : "BAD", acc_ok ? "ok" : "BAD", kw, gps, ns);

    // 4. timing
    auto time_ms = [&](auto&& fn, int P) {
      const int reps = R > 20000 ? 10 : 40;
      std::vector<float> ts;
      for (int r = 0; r < reps + 3; r++) {
        WView wc = w;
        wc.data = dw + wbytes * (r % ncopy);
        TCK(hipEventRecord(e0, 0));
        fn(wc, P);
        TCK(hipEventRecord(e1, 0));
        TCK(hipEventSynchronize(e1));
        float ms;
        TCK(hipEventElapsedTime(&ms, e0, e1));
        if (r >= 3) ts.push_back(ms);
      }
      std::sort(ts.begin(), ts.end());
      return (double)ts[ts.size() / 2];
    };
    auto fo = [&](const WView& wc, int P) { run_old(wc, dx, sh.xbf, dyo, P); };
    auto fn = [&](const WView& wc, int P) { run_new(wc, dx, sh.xbf, dyn, P); };
    const int Ps[] = {1, 2, 4, 5, 8, 16, 24, 40, 48};
    printf("   P:");
    for (int P : Ps) printf(" %7d", P);
    printf("\n  old:");
    std::vector<double> to, tn;
    for (int P : Ps) {
      to.push_back(time_ms(fo, P));
      printf(" %7.1f", to.back() * 1e3);
    }
    printf("  us (<=8-row gemv, chunked)\n  new:");
    for (int P : Ps) {
      tn.push_back(time_ms(fn, P));
      printf(" %7.1f", tn.back() * 1e3);
    }
    printf("  us  (P=1 %.0f GB/s)\n  o/n:", wbytes / (tn[0] * 1e-3) / 1e9);
    for (size_t i = 0; i < to.size(); i++) printf(" %7.2f", to[i] / tn[i]);
    printf("\n");
    const int idx[3] = {0, 3, 7};  // P = 1, 5, 40
    for (int k = 0; k < 3; k++) {
      step_old[k] += sh.per_step * to[idx[k]];
      step_new[k] += sh.per_step * tn[idx[k]];
    }
    if (!acc_ok || !inv_ok) fails++;
    TCK(hipFree(dw));
    TCK(hipFree(dx));
    TCK(hipFree(dxr));
    TCK(hipFree(dyo));
    TCK(hipFree(dyn));
    TCK(hipFree(dy2));
  }
  printf("dense q8g32 per decode step (weighted by calls), old -> new ms: P=1 %.2f -> %.2f | "
         "P=5 %.2f -> %.2f | P=40 %.2f -> %.2f\n",
         step_old[0], step_new[0], step_old[1], step_new[1], step_old[2], step_new[2]);
  printf("Q8_WMMA TEST: %s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
