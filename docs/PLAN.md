# 计划清单（2026-10-12 起维护）

按优先级/可行性排序的待办项。每项带量化依据；做之前先量、做之后 A/B。
已完成项移到文末"已完成"。方法论与工具：`tools/engine_ab_win.sh`（引擎 A/B，
显存互斥）、`tools/chat_turns_replay.py`（固定多轮对话）、`tools/iobench_win.cpp`
（纯磁盘 I/O）、`tools/perf_ab_win.sh`（qwenox-bench 离线基准）。

## 1. kvsnap 写队列满时丢弃而非阻塞 —— 待做（去重后压力减半，优先级下调）

`enqueue()`（51_host_cfg.inc）在队列（`QWENOX_KVSNAP_QUEUE_MB`，默认 1 GB）满时阻塞
GPU 线程；盘慢时实测把下一请求堵过 2.3 s。改为丢弃新快照（同时撤销 `g_ck` 的
pending 登记、释放页引用），记一个计数到日志。去重（见已完成）把每回合写量从 2 个
镜像降到 1 个，触发概率已明显下降，但 QLC 盘掉速时仍可能发生。

## 2. 终态 capture 的页打包（大 prefill 后省 ~150 ms）—— 已量化，低优先

下一请求在上一轮收尾上的等待：普通回合 8–34 ms，18K 冷 prefill 后 235 ms
（71 页 × 12 层 × (K,V,ik) ≈ 2800 次同步 hipMemcpy，85 µs/次是调用延迟）。
改法：gather kernel 把一页打包到设备暂存后一次 D2H（每页 2 次调用）。

## 3. 小 batch prefill 的 ~8.6 ms/token 底价 —— kernel 级，待评估

P≤32 的段 ≈ 60 ms + 8.6 ms/token：v2 ht/q6g64 的 `gemv_multi_q4cp` 每 ≤8 行扫一遍
全部权重（`gemvmr_maxp()`=24 后切 dequant+WMMA，作者已按实测调过交叉点）。要再降
需要新的小 P（16–64 行）权重直读 kernel。

## 4. SSD 周期性慢速期 —— 已归因（2026-10-13 更正：不是热降频）

**更正**：10-12 的"热降频"判断是错的。用户用 CrystalDiskInfo 看到 48–52 °C；10-13 用提权脚本
（`Get-StorageReliabilityCounter`）在崩塌期间持续采样，温度 48–51 °C（SMART 历史最高 85 °C）。

**实测事实（`tools/iobench_win.cpp sustain`/`sustainb`，纯磁盘）**：
- Crucial P3（本机唯一一块盘，C/D 同盘，94% 满，QLC、无 DRAM）在持续随机 4K 读下，每隔
  ~200–300 s 进入一次 **~55 s 的慢速期：约 9K IOPS、单次读延迟 8 ms（正常 0.4 ms）、队列
  被塞满**；缓冲读、无缓冲读、QD16/64/256、n-gram 文件和权重文件都一样 → 整盘的固件
  后台行为（读干扰刷新/GC，备用空间少时更频繁），不是主机争用（崩塌期其他进程无 I/O）。
- 另一个独立问题：**无缓冲 I/O 在文件存在缓存节或用户映射时被 NTFS 串行化到 QD1
  （~5K IOPS）**；缓冲读只会被用户映射串行化（13K），缓存节下正常（87K）。10-12 深夜
  到 10-13 期间生产跑的无缓冲版引擎正是被这个坑过（生产自己的冒烟请求建了缓存节）。
  现版本（缓冲读 + 撤映射，commit bae0eae）对两者都免疫，实测 87K IOPS。

**对引擎的影响**：块间预取只能藏 ≤6 s 的 gather；慢速期内 131K 行/chunk 要 ~14 s，
每个 8K chunk 多等 ~8 s，一次 131K 冷 prefill 撞上一个慢速期多花 ~35 s（观测到的
chunk 8–11 变慢就是它）。小增量撞上慢速期：1747 token 从 0.3 s 变 ~3.5 s。

**可选对策（按代价）**：
1. ~~给盘腾出空间~~ **已实测无效（2026-10-13）**：D 盘 94% → 87%（空闲 437 GB，TRIM 开启），等 TRIM/GC 消化约 40 分钟后重测：12 分钟持续读里崩塌在 t=170/425/675 s 各一次（间隔 ~255 s、时长 55 s、~9.5K IOPS），与腾空间前（间隔 200–300 s、55 s）完全一致；最初两轮 6 分钟看似无崩塌只是相位巧合。备用空间不是诱因。
2. **gather-ahead：已实现（commit 19f800c），现役（2026-10-13 晚用户恢复 `.bak-20261013-gatherahead`）**。
   **归因更正（10-13 深夜）**：17:18–19:19 生产变慢（加权 1.376 vs 补丁前 0.938 ms/token）**不是补丁**，
   是机器状态问题（重启后同版本完全正常：decode 97 ms/round、冷 prefill 0.877）。证据：
   ① 慢速窗口内 **decode 每轮延迟 160 ms**（基线 95、重启后 97）——decode 不碰链式也不碰磁盘；
   ② 窗口内 kvsnap 写盘全部 0.11–0.15 s（盘健康，SSD 慢速期排除）；③ 窗口内 18:12 的 51K token
   大 prefill 跑满速 0.879；④ 受控 A/B（19:24–19:48，仍在慢速期内）链式更快（T1 19.4 vs 30.1 s）；
   ⑤ 时间线：16:42:36 进新型待机 35 min → 17:18:02 鼠标唤醒 → 17:18:41 部署补丁 → 17:20–18:10 最重
   → 18:12 起部分恢复 → 19:19 前残留抖动 → 19:51 用户重启清除。慢速来源未定名（待机唤醒后降频状态
   或外部程序争用，见第 6/7 项；下次靠诊断行抓现场）。原始实现说明：
   `prefill_batch` 对多 chunk 段启动后台链，把该段 PLE 行连续聚集进 K 槽环形 pinned 暂存
   （`QWENOX_PLE_AHEAD_MB` 默认 256 MB → 8K chunk 12 槽；=0 关闭）。实测领先量：chunk 4
   时 11.8 s、chunk 8 时 48 s、结尾 109.8 s（`QWENOX_PROF=1` 打 `ple-ahead:` 行）——
   55 s 慢速期从 chunk 10 起被完全吸收，中段（chunk 5–9）部分吸收，开头几个 chunk 无
   改善（还没攒出领先）。无慢速期时与旧路径等价（131K：101.3 vs 104.5 s）。显存零增加
   （device 87.38 GiB 不变，pinned host +0.2 GiB）。测试旋钮 `QWENOX_PLE_TEST_SLOW="<chunk>:<ms>"`。
3. 引擎内 160 B 行级 LRU 缓存（比 4 KB 页缓存省 25× 内存）：先用 `QWENOX_PROF` 统计一天
   内唯一行/总行的比例再决定；命中率高则磁盘读与慢速期一起减少。
4. 换一块 TLC + DRAM 的 NVMe 专放 47.7 GB n-gram 文件（最稳）。

## 5. QSA 长上下文（PREFILL.md §11.9/§11.10 遗留）—— 索引打分 WMMA 暂缓（2026-10-12 决定）

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

## 6. 生产慢速诊断行 —— 建议（10-13 慢速事件催生）

今天排查 17:18–19:19 变慢耗了很久，因为生产日志（无 QWENOX_PROF）只能看到每请求总时长，
分不清慢在 PLE 磁盘等待 / GPU / host。建议给生产常开的最小诊断：

- 扩展现有 `chain-sample` 行（decode 期每几秒一条）：追加 `free/cache`（GlobalMemoryStatusEx）、
  `ac=`（GetSystemPowerStatus 的 ACLineStatus/BatterySaver），便于回看时对齐系统状态。
- decode 每轮耗时滚动统计：中位数连续 N 轮 > 阈值（如 130 ms）时打一行分解
  （`ple_wait` 累加器引擎已有，QWENOX_PHASE 在用；再加 GPU 同步等待与 host 段），每 ~30 s 一条
  直到恢复。成本：每轮几个计数器，可忽略。
- 目的：下次变慢 5 分钟定位（是盘、是机、是别程序），不再靠事后猜。

## 7. 环境侧异常（2026-10-13 发现，供用户处理）

- **Windows Update 死循环**：KB5129195（2026-09 安全更新，26100.9457）从 16:36 起每 15–20 分钟
  "下载→安装→成功"循环一轮（IsolationSession 服务启动类型被反复改），**重启后仍在继续**
  （20:07/20:08/20:31）。它本身不解释 1.7× 变慢（重启后带它跑仍正常），但白耗 CPU/内存/磁盘。
  修法：暂停更新止血 → `DISM /Online /Cleanup-Image /RestoreHealth` + `sfc /scannow` →
  停 wuauserv/bits 清 `C:\Windows\SoftwareDistribution\Download|DataStore` 后重扫。
- **新型待机（Modern Standby）**：16:42:36–17:18:02 机器在待机 35 min，期间引擎继续服务但被节流
  （17:08 的 3 个请求：1-token decode 155–207 ms、prefill 37 ms/token ≈ 2× 慢）。慢速期紧随唤醒
  之后开始，疑似唤醒后降频/驱动降级状态，**重启才清掉**。建议：给这台"服务器"设永不休眠
  （或合盖不待机）；再遇唤醒后持续变慢，先重启验证。
- 旁证：17:50 用户跑的 GPU-Z（GPUQuery_External.exe）在 AMD OpenGL 驱动 atio6axx.dll 里崩过
  （BEX64）；火绒（HipsTray/wsctrlsvc）与 WSL（vmmemWSL 845 MB–1.9 GB）常驻。以上在重启后
  都与正常性能共存，非充分原因。

## 8. PLE IOCP 批次重叠 → 引擎 abort（2026-10-14，2 槽生产复现）

**现象**：合并且 PARALLEL=2 的引擎在第一路 16.6K prefill 到 74% 时，第二路请求被接入另一个
槽（该槽有 56645 token 活跃序列需要驱逐），6 ms 后引擎退出，日志一行
`ple_win: read completion failed (row=18446744073709551615, got=160, status=0x0)`。

**机理**（`40_model.inc` 的 `run()` fail-fast：`!pov || slot >= depth || Internal || got < need`
→ abort；`row=(size_t)-1` 表示完成包的 OVERLAPPED 不属于本批）：**同一 PleWin/IOCP 上有两个
读批次在飞**，一个回收线程偷到另一个批次的完成包。同类风险有两处（都只在并发下出现）：

1. Windows 上 `PLE_PUMP` 就是 `g_ple_win`，而上游新的 inline 小批量路径
   （`issue()/wait_all()`，覆盖 decode / spec verify / 合批 verify）挂在它上面；同时
   **逐 token 直读**路径直接调 `g_ple_win.run(..., 16)`（40_model.inc:2354/2383/6233）。
   inline 批次按设计会"投完读 → 排 GPU → 再 wait"，在飞窗口很长，另一路（或另一槽）的
   per-token 读就能踩上。
2. 本分支的 gather-ahead 链在 **GPU 回合之外**后台聚簇，且不登记进共享的 `ple_fl`
   飞行体——上游的串行化前提是"链内 gather + 同一时刻一个 flight"，两条都被绕开；
   `PARALLEL>1` 时链的 pump 读可与另一段的 legacy gather 落在同一批 pump 实例上。

**已做（commit 见下）**：`PARALLEL>1`（`vseg_multi()`）时关掉链式预取
（启动日志改为 `ple: gather-ahead off (N concurrent slots; QWENOX_PLE_AHEAD_FORCE=1 to override)`），
单序列保持原行为；这消除了第 2 类（最可能的一次也在 2 槽 + 并发准入时）。

**待办**：第 1 类属上游设计，若 2 槽下仍偶发，应把 inline 路径拆到独立 PleWin（或对其加
一次性串行），值得在 PR 里与作者讨论。合成复现（2 路长 prefill 交错、2 路长解码、带槽驱逐）
均未复现，属稀有竞态；`PARALLEL=1`（生产配置）从未出现。

**2026-10-14 晚补记（"输出被截断/回合退化"的排查）**：用户客户端报"已到输出 token 上限"，
但引擎日志里这些请求全是 `finish=stop`（45.9K 上下文下只输出 95/45 token），引擎侧从未按
长度截断——该提示来自客户端。真正的截断最可能是 18:26 那次 **引擎 abort** 把在飞的流掐断。
另两条与"退化"相关的实测：① 本模型会把整个输出预算花在 `reasoning_content` 上
（`max_tokens=300` → 300 token 全是思考、正文为空、`finish=length`），客户端只计正文时就会
报"到了上限但只用了 50 多 K"；② 该对话的快照是在 2 槽 + 链式预取（PLE 串号可能期）写下的，
按内容寻址会被后续会话持续复用，状态可能被"污染"——建议换新会话或清 `data/kvsnap` 重算。
确定性问题另测：同一长 prompt 冷跑 + 两次快照恢复三次输出**逐字节一致**（一次偶发差异
未能复现，疑与默认开启的 dense WMMA（split-K/原子顺序）在近似平局处翻转有关，待用作者的
conc_diverge 工具复核）。

## 9. 客户端"输出被截断 / 回合退化" —— 已归因（2026-10-14 深夜）

**结论：引擎没有截断，是 API 把"输出在 tool call 中途结束"错标成 `finish_reason="length"`。**

**证据链**（客户端 DSH 会话 `session-40b7cd02` 的 jsonl + 本机 API/引擎日志按 responseId 对齐）：
回合 2 第 1 步在客户端记录 `stopReason: "length"`（客户端据 OpenAII finish_reason 判定），
而同一 responseId `chatcmpl-cc346495cc06bb88941da293` 在本机日志是
`REQ 63 end finish=stop prompt=45890 gen=412`——引擎侧既没到长度上限也没出错。
chat/completions 两条路径里唯一能把"引擎 reason=stop"变成"length"的代码就是
`parser.has_partial_call()`：`toolparse.cpp:779` 在输出流结束（`finish()`）时仍停在
tool-call 块内（`State::Broken`，或 final 时不在 `State::Text`）= 输出中途断了/JSON 参数
不完整 → `partial_=true`。

**客户端因此看到的连锁反应**：finish_reason=length → DSH 显示"已达到输出 token 上限，
回答被截断，已有输出保留在对话中，发送继续可让模型接着输出"（用户困惑"才用 50 多 K"）
→ 用户/客户端发"继续"补写，可那个回合其实已经自然结束 → 模型只能输出填充语
（实测该步正文为空，reasoning 只有 "I need to investigate this further. Let me check
the details."）→ DSH 的退化检测触发 → 抄进会话的就是那句
`❌ [ACP] stream error: the turn degenerated again after the continuation nudge`。

**同时确认引擎侧无辜**（同一个会话逐请求核对）：回合 1 的 13 个步骤各自
`out=437/230/292/264/469/674/472/123/179/391/733/203/274`，与引擎
`REQ 48/49/50/51/52/53/54/55/56/57/59/60/62` 的 `gen=` 完全一致——模型和引擎都按
客户端的每一步如数输出，没有丢 token、没有重启、没有 502。合并版与合并前的生产
（`logs/engine-api-20261013-195257.log`）在 30–49K 上下文的解码速度同为 ~30 tok/s；
MTP acceptance 从 ~0.78 降到 ~0.40 是上游自适应 γ（γ 3–5、commit/round 2.7–3.6）的口径
变化，不是质量或性能回归（每轮耗时 84 ms vs 合并前基线 95 ms）。

**修复**：`finish_reason_for()`（`src/api/main.cpp`，两条 chat/completions 路径都接入）：
只有引擎 `reason=="length"` 才报 `length`；有完整 tool call 仍报 `tool_calls`；
只有"输出中途断在不完整 tool call 上且没有任何完整调用"才报 `stop`（该不完整调用照旧丢弃），
并往 stderr 打一行 `chat <id>: output ended inside a tool call ...`，便于以后统计频次。
`/v1/responses` 路径本来就区分 `incomplete_details.reason = invalid_tool_call`，无需改。
验证：重新编译的 API 起在 8741（旁路实例，连同一引擎）——无 tools 请求 `finish=stop`、
带 tools 且模型正常调用时 `finish=tool_calls` 且参数解析正确，均与改前一致。

**另一条本次实测**：20:44–20:45 三个 kvsnap 写盘 11.27 / 18.48 s、随后 5.78 s（此前整段
会话都是 0.11–0.19 s），检查点目录已 4720 个文件 / 25.6 GB（上限 30 GB）——盘又进了慢速期
（见第 4 项）。虽与本次截断无关，但期间任何在飞请求都会被拖慢/堵住，值得留意；
若频繁出现，可考虑把 KVSNAP_MAX_GB 调小或换盘放 `data/kvsnap`。

## 已完成（2026-10-12）
- SSD 慢速期归因（10-13 更正为盘固件后台行为，非热）+ PLE 读改回缓冲默认（commit bae0eae，已部署）：见第 4 项。
- kvsnap 检查点去重（commit 6fea337，已部署）：chat/responses 回合不再写终态 SSD 检查点。
  依据：生产 38 次恢复 0 次命中终态（prompt 由历史重渲染，在 assistant 开头就分叉），
  每回合 2 个镜像里 1 个从未被用。验证：8 轮对话检查点 16 → 8、写量 2.3 → 1.4 GB，
  重启后 8 轮全部从切点检查点恢复且回复逐字相同，小回合下一请求等待 11–23 → 7–10 ms。
  `QWENOX_KVSNAP_END=1` 恢复旧行为；raw completions（无提示）不受影响。
- 中间切点跨度 <64 不拆段（`QWENOX_CUT_MIN_SPAN`，commit 77d290d）：小回合 −39%。
- PLE 无缓冲 IOCP 并行读 + 撤 n-gram 映射（commit ddadfae）：冷读 3–8×。
- KVSNAP_MAX_GB 20 → 30（service.conf）。
- 引擎重启后首请求 502 修复（commit e9f65b1）。
