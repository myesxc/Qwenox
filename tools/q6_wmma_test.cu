// q6_wmma_test.cu — dense WMMA q6g64 GEMV (k_q6g64_wmma) unit test + microbench.
//   bash tools/q6_wmma_test.sh          (ends Q6_WMMA TEST: PASS/FAIL)
// Per HC shape of the batched-verify path: accuracy vs the fp32 q6g64_gemv
// (chunked <=8-row passes), row invariance (shifted windows / reversed batch
// bitwise), determinism, timing over weight copies larger than the MALL.
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
  int per_round;  // calls per 8x5 batched-verify round (S0 GEMV_STAT / 5)
};
static const Shape kShapes[] = {
    {"hc.down", 320, 10240, true, 97},
    {"hc.down32", 320, 10240, false, 0},
    {"hc.up", 10240, 320, false, 97},
};

static void run_old(const WView& w, const void* x, bool xbf, float* y, int P) {
  for (int off = 0; off < P; off += 8) {
    const int pn = std::min(8, P - off);
    const void* xp = xbf ? (const void*)((const uint16_t*)x + (size_t)off * w.cols)
                         : (const void*)((const float*)x + (size_t)off * w.cols);
    if (!q6g64_gemv(w, xp, xbf, w.cols, y + (size_t)off * w.rows, w.rows, pn, 0, w.rows))
      throw std::runtime_error("q6g64_gemv");
  }
}
static void run_new(const WView& w, const void* x, bool xbf, float* y, int P) {
  if (!q6g64_wmma(w, x, xbf, w.cols, y, w.rows, P, 0, w.rows))
    throw std::runtime_error("q6g64_wmma refused");
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
static uint16_t f2h(float f) {
  const __half h = __float2half(f);
  uint16_t u;
  memcpy(&u, &h, 2);
  return u;
}

int main() {
  TCK(hipMalloc(&g_q6g_part, (size_t)16 * 8 * 2048 * 4));
  hipEvent_t e0, e1;
  TCK(hipEventCreate(&e0));
  TCK(hipEventCreate(&e1));
  const int MAXP = 48;
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_real_distribution<float> ud(0.f, 1.f);
  double tot_old[3] = {0, 0, 0}, tot_new[3] = {0, 0, 0};
  // the 4-row inject must be refused (old path keeps it)
  {
    WView w4{24, nullptr, 4, 10240, q6g64_stride(10240), nullptr, nullptr};
    if (q6g64_wmma(w4, nullptr, true, 10240, nullptr, 4, 5, 0, 4)) {
      printf("4x10240 accepted (expected refuse) FAIL\n");
      fails++;
    }
  }
  printf("%-9s %6s %5s | %9s %9s | %s\n", "shape", "R", "C", "maxerr/rms", "meanerr", "timing");
  for (const Shape& sh : kShapes) {
    const int R = sh.R, C = sh.C;
    const int st = (int)q6g64_stride(C);
    const size_t wbytes = (size_t)R * st;
    const int ncopy = (int)std::max<size_t>(1, std::min<size_t>(32, (384ull << 20) / wbytes));
    std::vector<uint8_t> hw(wbytes, 0);
    for (int r = 0; r < R; r++) {
      uint8_t* row = &hw[(size_t)r * st];
      for (int i = 0; i < 3 * C / 4; i++) row[i] = (uint8_t)(rng() & 0xff);
      for (int g = 0; g < C / 64; g++) {
        const float sc = (0.002f + 0.01f * ud(rng)) / sqrtf((float)C / 64.f);
        const float bi = -sc * (24.f + 16.f * ud(rng));
        const uint16_t a = f2h(sc), b = f2h(bi);
        memcpy(row + 3 * C / 4 + g * 4, &a, 2);
        memcpy(row + 3 * C / 4 + g * 4 + 2, &b, 2);
      }
    }
    uint8_t* dw;
    TCK(hipMalloc(&dw, wbytes * ncopy));
    for (int c = 0; c < ncopy; c++) {
      if (c)
        for (int r = 0; r < R; r++) hw[(size_t)r * st + (c * 7 + r) % (C / 2)] ^= (uint8_t)c;
      TCK(hipMemcpy(dw + wbytes * c, hw.data(), wbytes, hipMemcpyHostToDevice));
    }
    WView w{24, dw, (uint64_t)R, (uint64_t)C, (uint64_t)st, nullptr, nullptr};
    std::vector<float> hx((size_t)MAXP * C);
    for (auto& v : hx) v = nd(rng) * (ud(rng) < 0.002f ? 30.f : 1.f);
    std::vector<uint16_t> hxb(hx.size());
    for (size_t i = 0; i < hx.size(); i++) {
      hxb[i] = f2bf(hx[i]);
      if (sh.xbf) hx[i] = bf2f(hxb[i]);
    }
    // x in the shape's dtype (+ reversed copy)
    const size_t esz = sh.xbf ? 2 : 4;
    std::vector<uint8_t> xin((size_t)MAXP * C * esz), xrev(xin.size());
    if (sh.xbf)
      memcpy(xin.data(), hxb.data(), xin.size());
    else
      memcpy(xin.data(), hx.data(), xin.size());
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

    double maxe = 0, sume = 0;
    long cnt = 0;
    bool finite = true;
    for (int P : {1, 5, 8, 16, 17, 40, 48}) {
      run_old(w, dx, sh.xbf, dyo, P);
      run_new(w, dx, sh.xbf, dyn, P);
      TCK(hipDeviceSynchronize());
      auto yo = dget(dyo, (size_t)P * R), yn = dget(dyn, (size_t)P * R);
      for (int p = 0; p < P; p++) {
        double ss = 0;
        for (int i = 0; i < R; i++) ss += (double)yo[(size_t)p * R + i] * yo[(size_t)p * R + i];
        const double rms = sqrt(ss / R) + 1e-30;
        for (int i = 0; i < R; i++) {
          const float a = yn[(size_t)p * R + i];
          if (!std::isfinite(a)) finite = false;
          const double e = fabs((double)a - yo[(size_t)p * R + i]) / rms;
          maxe = std::max(maxe, e);
          sume += e;
          cnt++;
        }
      }
    }
    const double meane = sume / cnt;
    const bool acc_ok = finite && maxe < 5e-3 && meane < 5e-4;

    run_new(w, dx, sh.xbf, dyn, MAXP);
    TCK(hipDeviceSynchronize());
    auto yref = dget(dyn, (size_t)MAXP * R);
    long bad = 0;
    for (auto [P, off] : std::vector<std::pair<int, int>>{
             {1, 0}, {1, 13}, {1, 47}, {3, 5}, {5, 11}, {8, 40}, {9, 2}, {16, 32}, {17, 7},
             {24, 24}, {31, 17}, {33, 1}, {40, 8}, {47, 1}}) {
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
    for (int r = 0; r < 3; r++) {
      run_new(w, dx, sh.xbf, dy2, MAXP);
      TCK(hipDeviceSynchronize());
      auto y = dget(dy2, (size_t)MAXP * R);
      if (memcmp(y.data(), yref.data(), y.size() * 4)) bad++;
    }
    const bool inv_ok = bad == 0;

    auto time_ms = [&](auto&& fn, int P) {
      const int reps = 40;
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
    printf("%-9s %6d %5d | %9.2e %9.2e | inv %s acc %s\n", sh.name, R, C, maxe, meane,
           inv_ok ? "ok" : "BAD", acc_ok ? "ok" : "BAD");
    printf("   P:");
    const int Ps[] = {1, 2, 4, 5, 6, 8, 12, 16, 24, 40, 48};
    for (int P : Ps) printf(" %7d", P);
    printf("\n  old:");
    std::vector<double> to, tn;
    for (int P : Ps) {
      to.push_back(time_ms(fo, P));
      printf(" %7.1f", to.back() * 1e3);
    }
    printf("  us\n  new:");
    for (int P : Ps) {
      tn.push_back(time_ms(fn, P));
      printf(" %7.1f", tn.back() * 1e3);
    }
    printf("  us\n  o/n:");
    for (size_t i = 0; i < to.size(); i++) printf(" %7.2f", to[i] / tn[i]);
    printf("\n");
    const int idx[3] = {0, 3, 9};  // P = 1, 5, 40
    for (int k = 0; k < 3; k++) {
      tot_old[k] += sh.per_round * to[idx[k]];
      tot_new[k] += sh.per_round * tn[idx[k]];
    }
    if (!acc_ok || !inv_ok) fails++;
    TCK(hipFree(dw));
    TCK(hipFree(dx));
    TCK(hipFree(dxr));
    TCK(hipFree(dyo));
    TCK(hipFree(dyn));
    TCK(hipFree(dy2));
  }
  printf("round-weighted HC total (8x5 call mix, ms): P=1 old %.2f new %.2f | P=5 old %.2f new "
         "%.2f | P=40 old %.2f new %.2f\n",
         tot_old[0], tot_new[0], tot_old[1], tot_new[1], tot_old[2], tot_new[2]);
  printf(fails ? "Q6_WMMA TEST: FAIL (%d)\n" : "Q6_WMMA TEST: PASS\n", fails);
  return fails ? 1 : 0;
}
