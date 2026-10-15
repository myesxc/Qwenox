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

## 10. 输出在标记串处中断（`<|image_pad|>` / `<analysis>`）—— 审计与计划（2026-10-14 深夜）

**现象**：让模型解释 vision 占位符那个 bug 时，答案每次都在写到标记串后就结束——5 个会话
6 次收尾全部如此（`<|vision_start|><|image_pad|>`、`<|vision_end|><|image_pad|>`、
`<|im_start|><|image_pad|>`；"继续"后停在 `<analysis>`）。合并前的版本同样出现（用户实测）。

**已核实：不是被掐断，是模型自己发了 EOS**
- 今天所有请求的引擎结束原因只有 `| done`（EOS）66 次与 `| length`（预算）38 次，**零 `| cancel`**；
  API 侧零次停止串命中（`REQ n end ... (stop)` 后缀一次没出现）。
- 正文没丢字：从 `data/tcache.bin`（GDTC1 格式，u64 n + n×i32）解出那几次请求的生成 id 序列，
  末尾正是可见文本的最后一个字符，其后直接是 EOS（EOS 在发送前被拦下，故不在序列里）。
  例：REQ 37（41753 ids）末尾 = ``53948`_prompt' … 4307'Ġentire' 21685'Ġrendered' 1414'Ġtext'
  364'Ġfor' 328'Ġ"' `` + `248053:<|vision_start|>[SP]` + `'<','|','image','_pad','|','>'`。
- 结论：模型在写出这种"标记串"后立即输出 `<|im_end|>`。

**审计到的真问题：同一串标记文本在不同路径下编成不同 id（"杂交"形态）**
- 只有 image/video pad（248056/248057）在**文本**路径被排除（`encode_spliced` 的 `kTextNoPads`、
  `encode_prompt` 里"非模板放置的 pad"分支，均为 28047d4）；`<|vision_start|>`(248053)、
  `<|vision_end|>`(248054)、`<|vision_pad|>`(248055)、`<|audio_*|>` 等**只要出现在文本里就编成真
  special token**。
- 于是提示里是杂交形态：`<|vision_start|>` 是 special token，紧跟的 `<|image_pad|>` 是普通文本；
  **模型照抄提示，自己输出时也正是这个形态**（上面 tcache 末尾逐 id 可证）。
- 训练分布里没有这个形态：模板中 pad 永远是 special token，且 `<|vision_start|>…<|vision_end|>`
  图像块永远在 user 消息里、紧随其后就是 `<|im_end|>`。模型一"进入"该区域就把回合收尾——
  这是目前最可信的解释（另一候选见计划 B，需一次实验区分）。
- 次生影响：① 前缀/检查点命中率——引擎活跃序列含模型写出的 special token，而客户端重渲染的提示
  是普通文本，两边在标记处必然分叉，日志里成片的 `no live prefix; ... first mismatch at N` 就是它；
  ② 缓存复用会把这个不一致固化：`TokenCache::cut` 只拒绝"紧跟在 `<|vision_start|>`/pad 之后"切，
  复用前缀**中间**的 `vision_start`/`vision_end`/`audio_*` 仍以 special id 传下去，与本次新编的
  尾巴（普通文本）自相矛盾；③ `compute_snap_cuts` 用 `ng.text.rfind("<|im_end|>")` 取最后一条消息的
  收尾位置，正文里出现该字面串时会取到内容里的那个（只影响快照切点，不影响正确性）。
- 引擎侧的边界：提示里出现 stray pad 会被硬拒（`51_host_cfg.inc`：`vision: req N rejected: image
  placeholder id 248056 at prompt pos P without MROPE/VIMG`，其它标记不拒）；**生成**里的 added token
  只有两个 ckpt id（`<tool_call>`、``）走"暂停—串行 ingest—MTP 配对"的特殊路径，其余 added token
  直接走普通路径（计划 B 要判定这是否有问题）。

**计划**（A 立即可做；B 用一次实验决定是否动引擎；C/D 纵深防御）

**A. 标记文本的编码唯一化**（小改动，建议先做）
1. 把"非模板放置的 vision/audio 占位标记"在**文本**路径一律按普通文本编码：排除集从
   `{248056,248057}` 扩为 `{248053,248054,248055,248056,248057,248070,248071,248076}`
   （vision_start/end、vision_pad、image/video_pad、audio_start/end、audio_pad）。
   **作用域必须精确**：vision 路径里由模板 span（`kPadSpan` 的 `pad_offsets`）放置的那一对
   `<|vision_start|>/<|vision_end|>` 必须保持 special（真图片请求靠它们），只排除"落在模板 span
   之外"的同名标记；纯文本路径没有模板 span，可整集排除。
2. `TokenCache::cut` 的"不跨越"集合从 `{vstart, pads}` 扩为**整个排除集**：任何以 special 形式
   存在的这些 id 都不再被复用（否则第 1 条会被缓存复用绕回去）。
3. 单测：`tok_cli`/`vision_test` 增加"同一段含标记的文本在纯文本路径 / vision 路径 / 缓存复用
   路径下得到同一 id 序列"的断言；现有 8 例保持通过。
4. 预期：提示里不再有杂交标记，模型照抄提示写出的也是纯文本 → 触发条件消失；`no live prefix`
   出现率随之下降（可作指标）。风险低：真图片请求路径不变。

**A′. 输出侧禁止生成这些占位标记**（3 行的补充，建议与 A 同批）
请求未显式给 bias 时，默认给占位标记 id 加一个大负偏置（引擎的 `BIAS` 已支持、上限校验
`< 248320`）：模型再想写标记就只能写成普通文本，杂交形态在**源头**消失（历史文本自然是干净的）。
注意别把工具/思考语法（`<tool_call>`、``、`<tool_response>`）一起禁掉。

**B. 判定"EOS 是模型自己的选择还是引擎在 added token 处的状态问题"**（一次实验）
- 用 `data/tcache.bin` 里记录的**精确 prompt id** 重放（新工具 `tools/engine_replay_ids.py`：
  直接对引擎端口说 `GEN <req> <max_tokens> 2 248046 248044 <n> <ids...>`，收 `T` 行到 `D` 行，
  贪心、无 SAMPLE）；`--prompt-only` 与 `--with-generated`（把模型已生成的部分也喂回去接着写）
  两种都跑。
- 对照：`QWENOX_DRAFTER=chain`（默认）vs `QWENOX_DRAFTER=serial`（`0`，纯串行）。
  - 若 serial 不再在标记处结束 → **投机路径在 added token 上的状态不对**：把 `ckpt_pause` 那套
    （串行 ingest + `mtp_ingest_b` 配对）从 ckpt 两个 id 推广到**输出里出现的所有 added token**
    （`ckpt_toks` 扩为"模板切点 ∪ 输出 added token"，或对 added token 直接短路成串行一步）。
    这条与"合并前也有"吻合（投机机制早于合并）。
  - 若两组都在同一处结束 → 确认是模型分布问题，按 A/A′ 走。
- 同一工具还能验收 A：用"排除集"重新编码同一段文本再重放，看是否继续输出。

**C. 服务端纵深防御：不该进正文的 special token 就别进正文**
生成流里遇到"非预期"的 added token（不在工具/思考语法白名单里）时，按 llama.cpp 的
`skip_special_tokens` 语义从正文剔除或按边界收尾，并如实报 `finish_reason`（与 §9 的
`finish_reason_for` 同向）。需要一个白名单，避免把 `</tool_call>`/`` 剔掉。

**D. 引擎侧提示卫生观测（可选）**：把现在只查 pad 的 stray-placeholder 检查扩成"统计所有
structural added token（vision/audio 标记）在提示里出现的次数并打一行日志"（不拒绝、只观测），
下次一眼就能看出提示里有没有杂交标记。

**验证清单**：`build_win.sh test`（ktest 354/354）+ `vision_test` + `toolparse_test` + A3 新断言；
生产观察：`no live prefix` 出现率、结束原因分布（`| done`/`| length`）、"答案在标记处中断"的复现次数。

### 10.1 A / A′ 已实施（2026-10-15）

- **单一来源**：`vision::placeholder_marker_ids()`（`vision.h`/`vision.cpp`）=
  {248053,248054,248055,248056,248057,248070,248071,248076}，三处共用：① `vision.cpp`
  的 `encode_prompt` 里"落在模板 span 之外"的标记改为按普通文本重编（模板放置的那一对
  `vision_start/vision_end` 保持 special，真图片不受影响）；② `main.cpp` 纯文本路径
  （原 `kTextNoPads`）；③ `TokenCache::cut` 新增钳制：缓存条目里**任何**位置出现这些
  added token 就不再复用（走到该 id 之前），避免复用把 special 形态带回提示。
- **A′**：`run_generation` 在请求未显式给 bias 时，对 8 个标记 id 追加 −1e4 的默认偏置
  （客户端显式 bias 优先；`QWENOX_API_MARKER_BIAS=<v>` 覆盖、`0` 关闭）。**只对 sampled
  请求生效**——引擎把"没有 SAMPLE 却带 BIAS"的 GEN 行判为 reject（协议规定），greedy
  请求加了反而会让请求失败；greedy 侧要覆盖需要引擎内建禁表（并入 C 项）。
- **验证**（都在 Windows 侧跑过）：
  - `vision_test`（`-Wall -Wextra -Werror` 零警告）：新增"8 个标记字面串在文本路径与
    tokenizer 路径都不产生 added token"、"mixed 用例里模板 span 恰好保留 1 对
    vision_start/vision_end + pad_tokens 个 image_pad"等，全部 PASS。
  - `tools/api_regression_test.py`（对 `tools/api_fake_engine.py`）RESULT PASS——原有前端
    行为无回归。
  - 新 `tools/api_marker_verify.py` RESULT PASS：直接查引擎收到的 GEN 行——含标记字面串的
    请求 ids 里没有任何标记 id、标记以字节级 token 出现；sampled 请求带 8 个 id 的
    −10000 BIAS、greedy 不带 BIAS、客户端自带的 `logit_bias` 不被覆盖也不重复；并**构造
    一条把标记存成 248053 的 tcache 条目**重启 API 验证钳制：复用只到标记之前
    （日志 `tcache: reused 1 ids (12 of 91 bytes)`，去掉钳制时本可复用整段 18 ids 并
    把 248053 带回）。
  - 产物已就地更新：`build/qwenox-win.exe`（旧版备份 `build/qwenox-win.exe.bak-20261015-preA`）。
- **仍待**：B（判定投机路径是否贡献）与 C/D；greedy 请求的标记禁表（引擎侧）。

## 11. 502 "acp-proxy failure" + 引擎陪葬：API 空连接池 back() 崩溃（2026-10-15 已修）

**现象**：用户重启服务（API 00:35:38 起、引擎 00:35:48 由 API 拉起）后，00:36 第一个请求
失败——客户端报 `502 "acp-proxy failure"`（重试延迟 492 ms），引擎进程同时消失。
API 日志只剩 `engine connect failed` 与两行 `tcache: reused ...`，没有任何 `REQ n start`。

**定位**：Windows 应用程序日志里 `qwenox-win.exe` 异常 0xc0000005（访问违例），
**故障偏移 0xf1617**。用同源码同参数重建 + `-Xlinker "/MAP:..."`（Git Bash 下要
`MSYS_NO_PATHCONV=1`，否则 `/MAP` 会被当成路径改写）拿到符号表：该偏移落在
`run_generation+0x3b7`，故障指令是 `movq -0x8(%rax), %rcx`——即
`g_pool_free.back()` 在**空池**上的 `*(end()-1)`（main.cpp SlotGuard）。
引擎不是自己崩的：API 进程退出会连带结束它拉起的引擎（engine_sup 的看守），所以客户端
看到的是"引擎闪退 + 502"。

**机理**：`probe_engine()` 在 `g_ctl.connect()` 失败时**提前 return**，没有调用
`apply_engine_info` 建池（10-13 的修复只覆盖了"连上但 INFO 失败"那条路）。于是
"引擎监听之后、`reprobe_engine_loop` 补池之前"到达的请求，在 SlotGuard 里 pop 空池 → AV。
用户这次正是"服务刚起就发请求"，引擎冷加载 44 s，请求抢在 re-probe 前面。

**修复**：新增 `ensure_pool_locked()`（唯一的建池点，g_slot_mtx 下），三处调用——
`apply_engine_info`、`probe_engine` 的 connect 失败路径、以及 `SlotGuard` 取连接前
（纵深防御）。池里放未连接的 client 是设计内的：连接按需建立（engine_ready/bring_up），
引擎没起来时请求得到的是 502。

**验证**（都对 18741/18740 的旁路实例、无 GPU）：
- 旧二进制（`build/qwenox-win.exe.bak-20261015-preA`，即出事那版）：引擎不可达时启动 →
  第一个请求 **Segmentation fault（exit 139）**、无响应、进程消失——确定性复现；
- 新二进制：同一场景得到干净的 `502 {"message":"engine not reachable: ..."}`，进程存活；
- 迟到引擎：先起 API 再起引擎，re-probe 2 s 内补上 INFO，请求 200；
- `tools/api_regression_test.py`（含 engine-restart/池相关用例）与 `tools/api_marker_verify.py`
  在最终二进制上 RESULT PASS。
- 与 §10 的 A/A′ 无关：崩溃代码路径两边完全相同（旧二进制同样崩）。

**运维提示**：启动服务后立刻发第一个请求是安全的（修复后）；若再遇到"客户端 502 且引擎
同时没了"，先看 Windows 应用程序日志的 1000 事件（故障模块/偏移）——按上面的
`-Xlinker "/MAP:..."` 手法能直接定位到源码行。

## 12. 上下文扩容：YaRN 512K Windows 实测通过；1M 不可行（2026-10-15）

**支持情况**：引擎侧 YaRN 完整实现（`src/rope.h`，`QWENOX_ROPE_*` / `--rope-*`，KV 快照
指纹含全部 rope 参数；`QWENOX_QSA_UNION` 超 256K 自动禁用）；Linux 实机 512K factor 2 已于
2026-09-30 验证（YARN-512K-RESULTS.md，500K 检索 3/3、352K 尾部 PPL +0.4%）。

**1M 不可行**，三层各自独立成立：
1. 有效池硬闸：有效池 = max(KV_POOL_TOKENS, MAX_CONTEXT)，> 524288 直接拒
   （`service_conf.h validate_conf`，注释即"512K 以上任何配置都放不下"）。
2. arena 预算：Windows 一次性 hipMalloc 上限 95 GiB（96 GiB 实测失败）；实测 512K 配置
   estimate 91.69 + 4 slack = 95.69 → clamp 到 95.00；1M 估算 ≈ 111 GiB，超出 ~16 GiB。
3. 放不下的部分回退 hipMalloc → 共享内存——正是用户红线（占满共享显存触发死机）。
   引擎的 clamp 只打印一行警告，不会拒绝启动。

**512K Windows 实测（2026-10-15，当前二进制，含 cb738a7/1099cdb/4b98028）**：
- 启动（`MAX_CONTEXT=524288 ROPE_FACTOR=2`，生产其余参数不动）：`RoPE: YaRN factor=2
  original_ctx=262144 beta_fast=32 beta_slow=1 attention_scale=1.06931`；devarena clamp
  到 95.00 GiB（设计内行为，非错误）；**无任何 hipMalloc 回退**；arena-debug 实际峰值
  ~90.9 GiB；`HIP free 11.49/107.92 GiB`。基线对照：256K 生产 estimate 80.54 / request
  84.54，+256K 池约 +9.6 GiB，与 PORTING-WINDOWS.md 表格一致。
- `/health`：`context 524288`、`rope_scaling {type: yarn, factor: 2.0, attention_scale
  1.0693147180559945}`。
- 300K 针测试（`tools/yarn_ladder.py --only 300k`）：prompt 300149 token、**3/3 命中**、
  prefill 1200 tok/s、decode 44.4 tok/s（MTP acc 0.861）、255 s。短请求（yarn_short_probe.py）
  正常。测试实例设 `KVSNAP_MAX_GB=0` 保护生产 SSD 快照缓存。

**闸门修复（19d4ab5）**：此前只有 /admin/engine/start 与设置面板校验池上限，`start_win.sh`
与托盘菜单直调的 `enginesup::start` 都漏检——`MAX_CONTEXT=1048576 ROPE_FACTOR=4` 能从这两
条路静默启动并触发回退。已补：enginesup::start 开头 validate_conf；start_win.sh 补
MAX_CONTEXT 正整数检查 + 有效池 ≤ 524288 硬闸。验证：1M 两处均拒绝，512K/256K 通过，
`api_regression_test.py` + `api_marker_verify.py` 在新二进制上 RESULT PASS。

**切换配方**（要上 512K 时）：service.conf 改 `MAX_CONTEXT=524288`、`ROPE_FACTOR=2`，
其余不动，重启服务；客户端 contextWindow 同步调大。注意：YaRN factor 2 会改变短上下文
分布（短文本 mean KLD 0.0234 / same-top 93.5%，YARN-512K.md 建议不要给短流量常开）；
切换后 KV 快照指纹变化，旧 SSD 快照不复用（首次请求全量 prefill）。

**测试污染说明**：假引擎 rig 一次误从仓库根启动（cwd=repo），把 5 条测试条目写进了生产
`data/tcache.bin`（LRU 逐出 5 条旧条目；文件按 GDTC1 头逐条解析验证完好，无正确性影响）。

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
