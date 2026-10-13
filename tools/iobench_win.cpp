// iobench_win.cpp — random-read microbenchmark for the PLE table file (no GPU).
//   iobench_win <file> <mode> <depth> <count> [map]
//   mode: direct   = FILE_FLAG_NO_BUFFERING, 4 KB aligned reads, IOCP, `depth` in flight
//         sustain  = like direct, but runs for `count` SECONDS and prints IOPS every 5 s
//         sustainb = sustain with cached reads (immune to section serialisation)
//         buffered = cached overlapped reads of 160 B rows, IOCP, `depth` in flight
//         mmap     = touch 160 B rows through a MapViewOfFile view (page faults), serial
//   map: also keep a MapViewOfFile view of the whole file open during the run
//        (what the engine does), to expose cache-manager coherency effects.
// Offsets: uniform random 4 KB-aligned (direct) / 160 B rows (others) over the file.
#include <windows.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage: %s file direct|buffered|mmap depth count [map]\n", argv[0]); return 2; }
  const char* path = argv[1];
  const char* mode = argv[2];
  size_t depth = (size_t)atoi(argv[3]);
  size_t count = (size_t)atoi(argv[4]);
  bool keep_map = argc > 5 && !strcmp(argv[5], "map");
  uint64_t seed = argc > 6 ? strtoull(argv[6], nullptr, 10) : 12345;
  HANDLE hmapfile = INVALID_HANDLE_VALUE, hmap = nullptr;
  const uint8_t* view = nullptr;
  LARGE_INTEGER fsz{};
  {
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
    if (h == INVALID_HANDLE_VALUE) { fprintf(stderr, "open failed %lu\n", GetLastError()); return 1; }
    GetFileSizeEx(h, &fsz);
    CloseHandle(h);
  }
  if (keep_map || !strcmp(mode, "mmap")) {
    hmapfile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
    hmap = CreateFileMappingA(hmapfile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    view = (const uint8_t*)MapViewOfFile(hmap, FILE_MAP_READ, 0, 0, 0);
    if (!view) { fprintf(stderr, "map failed %lu\n", GetLastError()); return 1; }
    volatile uint8_t sink = view[0];  // make it a live data section
    (void)sink;
  }
  const uint64_t span = (uint64_t)fsz.QuadPart - 8192;
  std::mt19937_64 rng(seed);
  // sustain mode: `count` is seconds; unbuffered 4 KB random reads at `depth`
  // in flight, IOPS printed every 5 s (throttling / GC shows as a drop).
  // sustainb: same loop with cached (buffered) reads -- unbuffered I/O is
  // serialised by NTFS whenever the file has a cache section or a user mapping
  // (see PLAN.md), so compare both before blaming the drive.
  const bool sustain_buf = !strcmp(mode, "sustainb");
  if (!strcmp(mode, "sustain") || sustain_buf) {
    HANDLE fh = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                            FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS | (sustain_buf ? 0 : FILE_FLAG_NO_BUFFERING), nullptr);
    if (fh == INVALID_HANDLE_VALUE) { fprintf(stderr, "open failed %lu\n", GetLastError()); return 1; }
    HANDLE iocp = CreateIoCompletionPort(fh, nullptr, 0, 0);
    std::vector<OVERLAPPED> ovs(depth);
    std::vector<size_t> free_(depth);
    for (size_t i = 0; i < depth; i++) free_[i] = depth - 1 - i;
    size_t nfree = depth;
    uint8_t* buf = (uint8_t*)VirtualAlloc(nullptr, depth * 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    OVERLAPPED_ENTRY ents[64];
    const double secs = (double)count;
    auto t0 = std::chrono::steady_clock::now();
    auto tlast = t0;
    size_t done = 0, done_win = 0;
    printf("sustain%s depth=%zu for %.0f s (5 s windows)\n", sustain_buf ? " (buffered)" : " (unbuffered)", depth, secs);
    for (;;) {
      while (nfree > 0) {
        size_t slot = free_[--nfree];
        OVERLAPPED* ov = &ovs[slot];
        memset(ov, 0, sizeof *ov);
        uint64_t off = (rng() % span) & ~4095ull;
        ov->Offset = (DWORD)(off & 0xffffffffu); ov->OffsetHigh = (DWORD)(off >> 32);
        BOOL r = ReadFile(fh, buf + slot * 4096, 4096, nullptr, ov);
        if (!r && GetLastError() != ERROR_IO_PENDING) { fprintf(stderr, "ReadFile failed %lu\n", GetLastError()); return 1; }
      }
      ULONG n = 0;
      if (!GetQueuedCompletionStatusEx(iocp, ents, 64, &n, INFINITE, FALSE)) { fprintf(stderr, "GQCSEx failed %lu\n", GetLastError()); return 1; }
      for (ULONG k = 0; k < n; k++) { free_[nfree++] = (size_t)(ents[k].lpOverlapped - ovs.data()); done++; done_win++; }
      auto now = std::chrono::steady_clock::now();
      double w = std::chrono::duration<double>(now - tlast).count();
      if (w >= 5.0) {
        printf("  t=%5.0fs  %7.0f IOPS  (%.1f us/read)\n", std::chrono::duration<double>(now - t0).count(), done_win / w, w * 1e6 / done_win);
        fflush(stdout);
        tlast = now; done_win = 0;
      }
      if (std::chrono::duration<double>(now - t0).count() >= secs) break;
    }
    double s_all = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("sustain total: %zu reads in %.1f s = %.0f IOPS\n", done, s_all, done / s_all);
    return 0;
  }
  std::vector<uint64_t> offs(count);
  for (auto& o : offs) o = (rng() % span);
  const bool direct = !strcmp(mode, "direct");
  const size_t RB = 160;
  auto t0 = std::chrono::steady_clock::now();
  if (!strcmp(mode, "mmap")) {
    uint64_t acc = 0;
    for (size_t i = 0; i < count; i++) acc += view[offs[i]];
    auto t1 = std::chrono::steady_clock::now();
    double s = std::chrono::duration<double>(t1 - t0).count();
    printf("mmap   count=%zu  %.3f s  %.1f us/read  %.0f IOPS (acc %llu)\n", count, s, s * 1e6 / count, count / s, (unsigned long long)acc);
    return 0;
  }
  DWORD flags = FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS | (direct ? FILE_FLAG_NO_BUFFERING : 0);
  HANDLE fh = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, flags, nullptr);
  if (fh == INVALID_HANDLE_VALUE) { fprintf(stderr, "open(%s) failed %lu\n", mode, GetLastError()); return 1; }
  HANDLE iocp = CreateIoCompletionPort(fh, nullptr, 0, 0);
  if (depth > count) depth = count;
  std::vector<OVERLAPPED> ovs(depth);
  std::vector<size_t> free_(depth);
  for (size_t i = 0; i < depth; i++) free_[i] = depth - 1 - i;
  size_t nfree = depth, sub = 0, done = 0;
  uint8_t* buf = (uint8_t*)VirtualAlloc(nullptr, depth * 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
  std::vector<uint8_t> sbuf(depth * RB);
  OVERLAPPED_ENTRY ents[64];
  size_t sync_done = 0;
  while (done < count) {
    while (sub < count && nfree > 0) {
      size_t slot = free_[--nfree];
      OVERLAPPED* ov = &ovs[slot];
      memset(ov, 0, sizeof *ov);
      uint64_t off = offs[sub];
      void* dst; DWORD len;
      if (direct) { off &= ~4095ull; dst = buf + slot * 4096; len = 4096; }
      else { dst = sbuf.data() + slot * RB; len = (DWORD)RB; }
      ov->Offset = (DWORD)(off & 0xffffffffu); ov->OffsetHigh = (DWORD)(off >> 32);
      BOOL r = ReadFile(fh, dst, len, nullptr, ov);
      if (r) sync_done++;  // completed synchronously (packet still queued)
      else if (GetLastError() != ERROR_IO_PENDING) { fprintf(stderr, "ReadFile failed %lu\n", GetLastError()); return 1; }
      sub++;
    }
    ULONG n = 0;
    if (!GetQueuedCompletionStatusEx(iocp, ents, 64, &n, INFINITE, FALSE)) { fprintf(stderr, "GQCSEx failed %lu\n", GetLastError()); return 1; }
    for (ULONG k = 0; k < n; k++) {
      size_t slot = (size_t)(ents[k].lpOverlapped - ovs.data());
      if (ents[k].lpOverlapped->Internal != 0) { fprintf(stderr, "io status 0x%lx\n", (unsigned long)ents[k].lpOverlapped->Internal); return 1; }
      free_[nfree++] = slot; done++;
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  double s = std::chrono::duration<double>(t1 - t0).count();
  printf("%-8s depth=%-4zu count=%zu map=%d  %.3f s  %.1f us/read  %.0f IOPS  (sync-completed %zu)\n",
         mode, depth, count, (int)keep_map, s, s * 1e6 / count, count / s, sync_done);
  return 0;
}
