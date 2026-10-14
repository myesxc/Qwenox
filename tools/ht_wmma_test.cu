// ht_wmma_test.cu — dense WMMA ht GEMV (k_ht_wmma) unit test + microbench.
//   bash tools/ht_wmma_test.sh          (Windows Git Bash / Linux, ends PASS/FAIL)
// Checks, per v2 ht projection shape of the batched-verify path:
//   1. accuracy vs the fp32 k_ht_gemv (chunked <=8-row passes), P = 1..48;
//   2. row invariance: the same row gives bit-identical y at any P and any
//      position in the batch (shifted windows, reversed batch);
//   3. determinism: repeated launches bit-identical;
//   4. timing old (chunked gemv) vs new (wmma), weights rotated over copies
//      larger than the 32 MB MALL so every launch streams from DRAM like decode.
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

static std::mt19937 rng(20261012);
static int fails = 0;

struct Shape {
  const char* name;
  int O, K;
  int per_round;  // calls per batched-verify round (S0 GEMV_STAT, 8x5 R=40)
};
static const Shape kShapes[] = {
    {"gdn.qkv", 10240, 2560, 45}, {"gdn.z", 6144, 2560, 45},  {"gdn.out", 2560, 6144, 45},
    {"qsa.q", 12288, 2560, 15},   {"qsa.o", 2560, 6144, 15},  {"qsa.kv", 512, 2560, 30},
    {"se.gu", 640, 2560, 135},    {"se.down", 2560, 640, 60},
};

static void run_old(const WView& w, const float* x, float* y, int P) {
  for (int off = 0; off < P; off += 8) {
    const int pn = std::min(8, P - off);
    if (!ht_gemv(w, x + (size_t)off * w.cols, false, w.cols, y + (size_t)off * w.rows, w.rows,
                 pn, 0, w.rows))
      throw std::runtime_error("ht_gemv");
  }
}
static void run_new(const WView& w, const void* x, bool xbf, float* y, int P) {
  if (!ht_wmma(w, x, xbf, w.cols, y, w.rows, P, 0, w.rows)) throw std::runtime_error("ht_wmma");
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

int main() {
  TCK(hipMalloc(&g_ht_zs, kHtZs * 4));
  TCK(hipMemset(g_ht_zs, 0, kHtZs * 4));
  TCK(hipMalloc(&g_ht_cnt, 4096 * 4));
  TCK(hipMemset(g_ht_cnt, 0, 4096 * 4));
  hipEvent_t e0, e1;
  TCK(hipEventCreate(&e0));
  TCK(hipEventCreate(&e1));
  const int MAXP = 48;
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_real_distribution<float> ud(0.f, 1.f);
  double tot_old[3] = {0, 0, 0}, tot_new[3] = {0, 0, 0};  // round-weighted ms @P=1,5,40
  printf("%-8s %6s %5s | %9s %9s | %s\n", "shape", "O", "K", "maxerr/rms", "meanerr", "timing");
  for (const Shape& sh : kShapes) {
    const int O = sh.O, K = sh.K;
    const size_t wbytes = (size_t)O * K / 2;
    const int ncopy = (int)std::max<size_t>(1, std::min<size_t>(32, (384ull << 20) / wbytes));
    std::vector<uint8_t> hc(wbytes);
    for (auto& b : hc) b = (uint8_t)(rng() & 0xff);
    std::vector<float> hsuh(K), hsvh(O);
    for (auto& v : hsuh) v = (rng() & 1 ? 1.f : -1.f) * (0.75f + 0.5f * ud(rng));
    for (auto& v : hsvh) v = (rng() & 1 ? 1.f : -1.f) * (0.01f + 0.04f * ud(rng));
    uint8_t* dcodes;
    float *dsuh, *dsvh;
    TCK(hipMalloc(&dcodes, wbytes * ncopy));
    for (int c = 0; c < ncopy; c++) {
      if (c) for (size_t i = 0; i < wbytes; i += 4093) hc[i] ^= (uint8_t)c;  // distinct copies
      TCK(hipMemcpy(dcodes + wbytes * c, hc.data(), wbytes, hipMemcpyHostToDevice));
    }
    TCK(hipMalloc(&dsuh, K * 4));
    TCK(hipMalloc(&dsvh, O * 4));
    TCK(hipMemcpy(dsuh, hsuh.data(), K * 4, hipMemcpyHostToDevice));
    TCK(hipMemcpy(dsvh, hsvh.data(), O * 4, hipMemcpyHostToDevice));
    WView w{16, dcodes, (uint64_t)O, (uint64_t)K, 0, dsuh, dsvh};
    // activations: gaussian with sparse 30x outliers (f16 range check)
    std::vector<float> hx((size_t)MAXP * K);
    for (auto& v : hx) v = nd(rng) * (ud(rng) < 0.002f ? 30.f : 1.f);
    std::vector<float> hxr((size_t)MAXP * K);  // reversed row order
    for (int p = 0; p < MAXP; p++)
      memcpy(&hxr[(size_t)p * K], &hx[(size_t)(MAXP - 1 - p) * K], K * 4);
    std::vector<uint16_t> hxb(hx.size());
    for (size_t i = 0; i < hx.size(); i++) hxb[i] = f2bf(hx[i]);
    float *dx, *dxr, *dyo, *dyn, *dy2;
    uint16_t* dxb;
    TCK(hipMalloc(&dx, hx.size() * 4));
    TCK(hipMalloc(&dxr, hx.size() * 4));
    TCK(hipMalloc(&dxb, hx.size() * 2));
    TCK(hipMemcpy(dx, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
    TCK(hipMemcpy(dxr, hxr.data(), hx.size() * 4, hipMemcpyHostToDevice));
    TCK(hipMemcpy(dxb, hxb.data(), hxb.size() * 2, hipMemcpyHostToDevice));
    TCK(hipMalloc(&dyo, (size_t)MAXP * O * 4));
    TCK(hipMalloc(&dyn, (size_t)MAXP * O * 4));
    TCK(hipMalloc(&dy2, (size_t)MAXP * O * 4));

    // 1. accuracy (P = 48 covers every row; spot P values for the tile paths)
    double maxe = 0, sume = 0;
    long cnt = 0;
    bool finite = true;
    for (int P : {1, 5, 8, 16, 17, 40, 48}) {
      run_old(w, dx, dyo, P);
      run_new(w, dx, false, dyn, P);
      TCK(hipDeviceSynchronize());
      auto yo = dget(dyo, (size_t)P * O), yn = dget(dyn, (size_t)P * O);
      for (int p = 0; p < P; p++) {
        double ss = 0;
        for (int i = 0; i < O; i++) ss += (double)yo[(size_t)p * O + i] * yo[(size_t)p * O + i];
        const double rms = sqrt(ss / O) + 1e-30;
        for (int i = 0; i < O; i++) {
          const float a = yn[(size_t)p * O + i];
          if (!std::isfinite(a)) finite = false;
          const double e = fabs((double)a - yo[(size_t)p * O + i]) / rms;
          maxe = std::max(maxe, e);
          sume += e;
          cnt++;
        }
      }
    }
    const double meane = sume / cnt;
    const bool acc_ok = finite && maxe < 5e-3 && meane < 5e-4;
    // bf16-x path vs the old bf16-x path (P = 40)
    {
      for (int off = 0; off < 40; off += 8)
        ht_gemv(w, dxb + (size_t)off * K, true, K, dyo + (size_t)off * O, O, 8, 0, O);
      run_new(w, dxb, true, dyn, 40);
      TCK(hipDeviceSynchronize());
      auto yo = dget(dyo, (size_t)40 * O), yn = dget(dyn, (size_t)40 * O);
      double ss = 0, me = 0;
      for (float v : yo) ss += (double)v * v;
      const double rms = sqrt(ss / yo.size());
      for (size_t i = 0; i < yo.size(); i++) me = std::max(me, fabs((double)yn[i] - yo[i]) / rms);
      if (!(me < 5e-3)) {
        printf("  %s bf16-x maxerr/rms %.3e FAIL\n", sh.name, me);
        fails++;
      }
    }

    // 2. row invariance: reference = P=48 batch; windows [off, off+P) and the
    //    reversed batch must reproduce the same rows bit for bit.
    run_new(w, dx, false, dyn, MAXP);
    TCK(hipDeviceSynchronize());
    auto yref = dget(dyn, (size_t)MAXP * O);
    long bad = 0;
    for (auto [P, off] : std::vector<std::pair<int, int>>{
             {1, 0}, {1, 13}, {1, 47}, {3, 5}, {5, 11}, {8, 40}, {9, 2}, {16, 32}, {17, 7},
             {24, 24}, {31, 17}, {33, 1}, {40, 8}, {47, 1}}) {
      run_new(w, dx + (size_t)off * K, false, dy2, P);
      TCK(hipDeviceSynchronize());
      auto y = dget(dy2, (size_t)P * O);
      if (memcmp(y.data(), &yref[(size_t)off * O], (size_t)P * O * 4)) bad++;
    }
    run_new(w, dxr, false, dy2, MAXP);
    TCK(hipDeviceSynchronize());
    {
      auto y = dget(dy2, (size_t)MAXP * O);
      for (int p = 0; p < MAXP; p++)
        if (memcmp(&y[(size_t)p * O], &yref[(size_t)(MAXP - 1 - p) * O], (size_t)O * 4)) {
          bad++;
          break;
        }
    }
    // 3. determinism
    for (int r = 0; r < 3; r++) {
      run_new(w, dx, false, dy2, MAXP);
      TCK(hipDeviceSynchronize());
      auto y = dget(dy2, (size_t)MAXP * O);
      if (memcmp(y.data(), yref.data(), y.size() * 4)) bad++;
    }
    const bool inv_ok = bad == 0;

    // 4. timing (median of reps, each rep on the next weight copy)
    auto time_ms = [&](auto&& fn, int P) {
      const int reps = 40;
      std::vector<float> ts;
      for (int r = 0; r < reps + 3; r++) {
        WView wc = w;
        wc.data = dcodes + wbytes * (r % ncopy);
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
    auto fo = [&](const WView& wc, int P) { run_old(wc, dx, dyo, P); };
    auto fn = [&](const WView& wc, int P) { run_new(wc, dx, false, dyn, P); };
    printf("%-8s %6d %5d | %9.2e %9.2e | inv %s acc %s\n", sh.name, O, K, maxe, meane,
           inv_ok ? "ok" : "BAD", acc_ok ? "ok" : "BAD");
    printf("   P:");
    const int Ps[] = {1, 2, 4, 5, 8, 16, 24, 40, 48};
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
    const int idx[3] = {0, 3, 7};  // P = 1, 5, 40
    for (int k = 0; k < 3; k++) {
      tot_old[k] += sh.per_round * to[idx[k]];
      tot_new[k] += sh.per_round * tn[idx[k]];
    }
    if (!acc_ok || !inv_ok) fails++;
    TCK(hipFree(dcodes));
    TCK(hipFree(dsuh));
    TCK(hipFree(dsvh));
    TCK(hipFree(dx));
    TCK(hipFree(dxr));
    TCK(hipFree(dxb));
    TCK(hipFree(dyo));
    TCK(hipFree(dyn));
    TCK(hipFree(dy2));
  }
  printf("round-weighted ht total (S0 call mix, ms): P=1 old %.2f new %.2f | P=5 old %.2f new "
         "%.2f | P=40 old %.2f new %.2f\n",
         tot_old[0], tot_new[0], tot_old[1], tot_new[1], tot_old[2], tot_new[2]);

  // ---- pair launch (shared expert gate + up, 2 x 640x2560): ht_wmma_pair vs
  // two ht_wmma launches (close, not bitwise: ks differs), pair row
  // invariance, timing pair vs 2 separate.
  {
    const int O = 640, K = 2560;
    const size_t wbytes = (size_t)O * K / 2;
    const int ncopy = 32;
    uint8_t* dc[2];
    float *dsuh[2], *dsvh[2];
    for (int k = 0; k < 2; k++) {
      std::vector<uint8_t> hc(wbytes * ncopy);
      for (auto& b : hc) b = (uint8_t)(rng() & 0xff);
      std::vector<float> hsuh(K), hsvh(O);
      for (auto& v : hsuh) v = (rng() & 1 ? 1.f : -1.f) * (0.75f + 0.5f * ud(rng));
      for (auto& v : hsvh) v = (rng() & 1 ? 1.f : -1.f) * (0.01f + 0.04f * ud(rng));
      TCK(hipMalloc(&dc[k], hc.size()));
      TCK(hipMemcpy(dc[k], hc.data(), hc.size(), hipMemcpyHostToDevice));
      TCK(hipMalloc(&dsuh[k], K * 4));
      TCK(hipMalloc(&dsvh[k], O * 4));
      TCK(hipMemcpy(dsuh[k], hsuh.data(), K * 4, hipMemcpyHostToDevice));
      TCK(hipMemcpy(dsvh[k], hsvh.data(), O * 4, hipMemcpyHostToDevice));
    }
    WView wa{16, dc[0], (uint64_t)O, (uint64_t)K, 0, dsuh[0], dsvh[0]};
    WView wb{16, dc[1], (uint64_t)O, (uint64_t)K, 0, dsuh[1], dsvh[1]};
    std::vector<float> hx((size_t)MAXP * K);
    for (auto& v : hx) v = nd(rng);
    float *dx, *ya, *yb, *ra, *rb, *ta, *tb;
    TCK(hipMalloc(&dx, hx.size() * 4));
    TCK(hipMemcpy(dx, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
    for (float** p : {&ya, &yb, &ra, &rb, &ta, &tb}) TCK(hipMalloc(p, (size_t)MAXP * O * 4));
    // accuracy: pair vs separate at P = 48
    if (!ht_wmma_pair(wa, wb, dx, false, K, ya, yb, O, MAXP)) {
      printf("pair refused FAIL\n");
      fails++;
    }
    run_new(wa, dx, false, ra, MAXP);
    run_new(wb, dx, false, rb, MAXP);
    TCK(hipDeviceSynchronize());
    double me = 0;
    {
      auto a = dget(ya, (size_t)MAXP * O), b = dget(yb, (size_t)MAXP * O);
      auto c = dget(ra, (size_t)MAXP * O), d = dget(rb, (size_t)MAXP * O);
      double ss = 0;
      for (float v : c) ss += (double)v * v;
      const double rms = sqrt(ss / c.size());
      for (size_t i = 0; i < a.size(); i++)
        me = std::max(me, std::max(fabs((double)a[i] - c[i]), fabs((double)b[i] - d[i])) / rms);
    }
    // row invariance of the pair
    auto refa = dget(ya, (size_t)MAXP * O), refb = dget(yb, (size_t)MAXP * O);
    long bad = 0;
    for (auto [P, off] : std::vector<std::pair<int, int>>{
             {1, 0}, {1, 29}, {5, 11}, {8, 40}, {9, 2}, {17, 7}, {33, 1}, {40, 8}}) {
      ht_wmma_pair(wa, wb, dx + (size_t)off * K, false, K, ta, tb, O, P);
      TCK(hipDeviceSynchronize());
      auto a = dget(ta, (size_t)P * O), b = dget(tb, (size_t)P * O);
      if (memcmp(a.data(), &refa[(size_t)off * O], (size_t)P * O * 4) ||
          memcmp(b.data(), &refb[(size_t)off * O], (size_t)P * O * 4))
        bad++;
    }
    const bool ok = me < 1e-4 && bad == 0;
    printf("se.pair  2x%d %5d | pair-vs-separate maxerr/rms %.2e | inv %s\n", O, K, me,
           bad ? "BAD" : "ok");
    if (!ok) fails++;
    auto tm = [&](bool pair, int P) {
      std::vector<float> ts;
      for (int r = 0; r < 43; r++) {
        WView a = wa, b = wb;
        a.data = dc[0] + wbytes * (r % ncopy);
        b.data = dc[1] + wbytes * (r % ncopy);
        TCK(hipEventRecord(e0, 0));
        if (pair) {
          ht_wmma_pair(a, b, dx, false, K, ya, yb, O, P);
        } else {
          run_new(a, dx, false, ra, P);
          run_new(b, dx, false, rb, P);
        }
        TCK(hipEventRecord(e1, 0));
        TCK(hipEventSynchronize(e1));
        float ms;
        TCK(hipEventElapsedTime(&ms, e0, e1));
        if (r >= 3) ts.push_back(ms);
      }
      std::sort(ts.begin(), ts.end());
      return ts[ts.size() / 2] * 1e3;
    };
    printf("   P:");
    for (int P : {1, 5, 8, 16, 40}) printf(" %7d", P);
    printf("\n  2x1:");
    for (int P : {1, 5, 8, 16, 40}) printf(" %7.1f", tm(false, P));
    printf("  us\n pair:");
    for (int P : {1, 5, 8, 16, 40}) printf(" %7.1f", tm(true, P));
    printf("  us\n");
  }
  printf(fails ? "HT_WMMA TEST: FAIL (%d)\n" : "HT_WMMA TEST: PASS\n", fails);
  return fails ? 1 : 0;
}
