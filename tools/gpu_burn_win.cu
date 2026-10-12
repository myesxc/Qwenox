// gpu_burn_win.cu — hold the GPU busy for N seconds (no disk, tiny VRAM) so disk
// I/O can be benchmarked under GPU load.
//   gpu_burn_win compute <secs>     bf16 WMMA issue loop (power/heat, no DRAM traffic)
//   gpu_burn_win bandwidth <secs>   D2D copies between two 1 GiB buffers (DRAM traffic)
// Prints elapsed time and the achieved rate every ~5 s.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
using sx16 = __attribute__((ext_vector_type(16))) short;
using fx8 = __attribute__((ext_vector_type(8))) float;
__global__ void __launch_bounds__(256) k_peak(const short* seed, float* out, int iters) {
  sx16 a, b;
  short t[16];
  for (int i = 0; i < 16; ++i) t[i] = seed[(threadIdx.x + i) & 63];
  memcpy(&a, t, 32);
  for (int i = 0; i < 16; ++i) t[i] = seed[(threadIdx.x + i + 32) & 63];
  memcpy(&b, t, 32);
  fx8 acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < 8; ++i)
      acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[i]);
  }
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) s += acc[i][e];
  if (s == 12345.f) out[threadIdx.x] = s;  // keep the loop alive
}
int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s compute|bandwidth <secs>\n", argv[0]); return 2; }
  const bool bw = !strcmp(argv[1], "bandwidth");
  const double secs = atof(argv[2]);
  auto t0 = std::chrono::steady_clock::now();
  auto tl = t0;
  if (!bw) {
    short* seed; float* out;
    hipMalloc(&seed, 1024); hipMalloc(&out, 4096); hipMemset(seed, 0x3c, 1024);
    const int iters = 20000, grid = 320;
    long n = 0, nw = 0;
    for (;;) {
      k_peak<<<grid, 256>>>(seed, out, iters);
      hipDeviceSynchronize();
      n++; nw++;
      auto now = std::chrono::steady_clock::now();
      double w = std::chrono::duration<double>(now - tl).count();
      if (w >= 5.0) {
        double flops = (double)grid * 8 * iters * 8 * (16.0 * 16 * 16 * 2) * nw;
        printf("  burn t=%5.0fs  %.1f TFLOPS\n", std::chrono::duration<double>(now - t0).count(), flops / w / 1e12);
        fflush(stdout); tl = now; nw = 0;
      }
      if (std::chrono::duration<double>(now - t0).count() >= secs) break;
    }
  } else {
    const size_t N = (size_t)1 << 30;
    void *a, *b;
    if (hipMalloc(&a, N) != hipSuccess || hipMalloc(&b, N) != hipSuccess) { fprintf(stderr, "hipMalloc failed\n"); return 1; }
    hipMemset(a, 1, N);
    long nw = 0;
    for (;;) {
      hipMemcpy(b, a, N, hipMemcpyDeviceToDevice);
      hipMemcpy(a, b, N, hipMemcpyDeviceToDevice);
      nw += 2;
      auto now = std::chrono::steady_clock::now();
      double w = std::chrono::duration<double>(now - tl).count();
      if (w >= 5.0) {
        printf("  burn t=%5.0fs  %.1f GB/s (r+w)\n", std::chrono::duration<double>(now - t0).count(), 2.0 * nw * N / w / 1e9);
        fflush(stdout); tl = now; nw = 0;
      }
      if (std::chrono::duration<double>(now - t0).count() >= secs) break;
    }
  }
  printf("burn done\n");
  return 0;
}
