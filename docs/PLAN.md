# 计划清单（2026-10-12 起维护）

按优先级/可行性排序的待办项。每项带量化依据；做之前先量、做之后 A/B。
已完成项移到文末"已完成"。方法论与工具：`tools/engine_ab_win.sh`（引擎 A/B，
显存互斥）、`tools/chat_turns_replay.py`（固定多轮对话）、`tools/iobench_win.cpp`
（纯磁盘 I/O）、`tools/perf_ab_win.sh`（qwenox-bench 离线基准）。

## 1. kvsnap 检查点去重（写量减半）—— 已量化，待做

**数据（生产 2026-10-11/12 两天，249 个请求）**：720 个检查点，共写 **134.6 GB**
（检查点镜像 81.6 GB + 新页/MTP 段 53.0 GB），平均每请求 2.89 个 / 554 MB。
**27% 的检查点与前一个只隔 ≤64 token（中位数 2 token）**；agent 式多轮对话
（`chat_turns_replay.py`）里是 81%（中位 13 token）。写时间中位 0.15 s、p90 1.9 s、
最大 19.4 s（D: 是 94% 满的 QLC 盘，Crucial P3）。

**成因**：每轮写 ① 最后一个 SNAPS 切点（消息内容末尾，in-pass）② 回合终态
（内容末尾 + 结束标签 + 生成的几个 token）③ 恢复点之后再次经过的旧边界。
②与①通常只差 5–20 token；下一轮的 prompt 恰在①处分叉，②只对"同一上下文
原样续写"有用，而这种情况由 live slot（cont）直接服务。

**改法（候选）**：
- 终态检查点若与最近一个已捕获切点距离 < N token（N=64，与 `QWENOX_CUT_MIN_SPAN`
  同口径）则跳过 SSD 捕获（RAM rckpt 便宜可保留）。预期检查点镜像写量 −27%（生产）
  ~ −50%（agent）。
- 写队列（`QWENOX_KVSNAP_QUEUE_MB`，默认 1 GB）满时丢弃新快照而不是阻塞 GPU 线程
  （现状实测把下一请求堵过 2.3 s）。丢弃需同时撤销 `g_ck` 里的 pending 登记。
- 验证：`kvsnap: saved` 行计数/字节对比；`tools/api_regression_test.py` 的
  kvsnap/rckpt 用例；换对话再切回来的恢复命中率不降。

## 2. 终态 capture 的页打包（大 prefill 后省 ~150 ms）—— 已量化，低优先

下一请求在上一轮收尾上的等待：普通回合 8–34 ms，18K 冷 prefill 后 235 ms
（71 页 × 12 层 × (K,V,ik) ≈ 2800 次同步 hipMemcpy，85 µs/次是调用延迟）。
改法：gather kernel 把一页打包到设备暂存后一次 D2H（每页 2 次调用）。

## 3. 小 batch prefill 的 ~8.6 ms/token 底价 —— kernel 级，待评估

P≤32 的段 ≈ 60 ms + 8.6 ms/token：v2 ht/q6g64 的 `gemv_multi_q4cp` 每 ≤8 行扫一遍
全部权重（`gemvmr_maxp()`=24 后切 dequant+WMMA，作者已按实测调过交叉点）。要再降
需要新的小 P（16–64 行）权重直读 kernel。

## 4. QSA 长上下文（PREFILL.md §11.9/§11.10 遗留）

- `k_index_scores_t64` 换 bf16 WMMA：作者估 ~2× 余地（128K 时打分 ≈ 360 ms/8K chunk），
  不再逐位一致，需先做长上下文 KLD。
- `k_qsa_wmma` 调优（128K 时 640 ms/chunk，作者标"大工程"）。
- **本机剖析（2026-10-12，131072 token 冷 prefill，8K chunk，QWENOX_KPROF）**：每 chunk
  ~6.0 s（8K 处）→ 6.4 s（122K 处），增量全在 QSA：`qsa:idx` 68 → 334 ms（5×），
  `qsa:flash` 356 → 720 ms（2×，是 K/V gather 随上下文变大溢出 MALL 后的 DRAM 带宽
  问题，不是算力问题），`qsa:proj`/`oproj` 恒定 225/155 ms；GDN 1.9 s、MoE 2.0 s 恒定。
  QSA 占比 12% → 20%。→ 索引打分 WMMA 化的收益上限：128K 处 ~2.6%、230K 处 ~4%，
  32K 以下无感；flash 的 gather 字节数靠 kernel 调优降不下来。无本地 BF16 KLD 基准，
  长上下文验证只能做引擎 fp32 路径 vs WMMA 路径的自比 KLD（`--kld-save`/`--kld-base`）。
- **同一剖析暴露的更大问题（SSD 掉速）**：PLE 无缓冲读把 Crucial P3 推到 ~100K IOPS 上限
  后，盘会在持续高压 ~60 s 后进入 4–12× 慢速状态几十秒（131K 冷 prefill 里 4 个 chunk
  的 gather 5–16 s 而非 1.3 s，另一跑首块 15.3 s）。块间预取能藏住 ≤ 6 s 的 gather，
  藏不住掉速段。疑似热降频（需管理员权限读温度，或用 CrystalDiskInfo 看）。可试：
  降 `QWENOX_PLE_DEPTH`（限制 IOPS）、给盘加散热、或把 n-gram 文件放到另一块盘。
  注：每 chunk 固定 ~1.3 s 的 ple_wait 只在 `QWENOX_PHASE=1` 下出现（逐阶段同步让预取
  失效），生产里 chunk 2 起 ple_wait=0。

## 已完成（2026-10-12）
- 中间切点跨度 <64 不拆段（`QWENOX_CUT_MIN_SPAN`，commit 77d290d）：小回合 −39%。
- PLE 无缓冲 IOCP 并行读 + 撤 n-gram 映射（commit ddadfae）：冷读 3–8×。
- KVSNAP_MAX_GB 20 → 30（service.conf）。
- 引擎重启后首请求 502 修复（commit e9f65b1）。
