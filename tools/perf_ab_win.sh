#!/usr/bin/env bash
# perf_ab_win.sh — Windows 离线性能 A/B 驱动（qwenox-bench，固定题面，可重复对比）。
#
# 流程：停引擎 → 等显存真正释放（进程消失 + 共享显存回落，双重校验）→ 以生产 env +
#      覆盖项跑 qwenox-bench run（全程采样共享显存）→ 存日志/摘要 → 自动恢复引擎。
#
# 显存安全（重要）：bench 与引擎各要 ~87 GiB 共享显存，同时运行必然溢出。
#   本脚本在启动 bench 前做两道硬校验：① 引擎进程不存在；② GPU 共享显存 < VRAM_FREE_MAX_GB。
#   任一不满足就等待/拒绝启动；运行中每 15s 采样一次共享显存，峰值写进摘要。
#
# 用法（Git Bash）：
#   bash tools/perf_ab_win.sh <label> [K=V ...]                    跑一组（label=配置名）
#   BASELINE=<label> bash tools/perf_ab_win.sh <label2> [K=V ...]  跑完与 BASELINE 对比
#   bash tools/perf_ab_win.sh --compare <labelA> <labelB>          只对比已有两组
#   bash tools/perf_ab_win.sh --gpu                                 只看当前共享显存（不跑测试）
#
# 常用缩短：BENCH_PREFILL_REPEATS=1 BENCH_DECODE_REPEATS=1 bash tools/perf_ab_win.sh ...
# 变量：QWENOX_DIR（默认脚本所在仓库根目录）、API_PORT（8731）、
#       ADMIN_KEY（API 配了 admin key 时传）、BENCH（覆盖 bench 路径）、
#       VRAM_FREE_MAX_GB（启动 bench 前的共享显存上限，默认 10）。
#
# 默认 env 取自 Windows 生产启动器（src/api/engine_sup.cpp:270-300）的那套开关；
# KVSNAP/RCKPT 关闭：离线基准不需要快照机制，避免写盘噪声。覆盖项 K=V 替换同名默认项。
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
QWENOX_DIR="${QWENOX_DIR:-$ROOT}"
API_PORT="${API_PORT:-8731}"
ADMIN_KEY="${ADMIN_KEY:-}"
BENCH="${BENCH:-$QWENOX_DIR/build/qwenox-bench.exe}"
RESULTS_DIR="$QWENOX_DIR/logs/perf_ab"
VRAM_FREE_MAX_GB="${VRAM_FREE_MAX_GB:-10}"

die() { echo "perf_ab: $*" >&2; exit 1; }

api() {  # api <curl args...>
    if [[ -n "$ADMIN_KEY" ]]; then
        curl -s -m 10 -H "X-Admin-Key: $ADMIN_KEY" "$@"
    else
        curl -s -m 10 "$@"
    fi
}

engine_state() {
    api "http://127.0.0.1:$API_PORT/admin/engine/status" 2>/dev/null \
        | grep -o '"state":"[a-z_]*"' | head -1 | cut -d'"' -f4
}

engine_procs() {  # 引擎进程数（不看 API 的 status，直接查进程表）
    tasklist //FI "IMAGENAME eq qwenox-engine-win.exe" 2>/dev/null \
        | grep -c "qwenox-engine-win.exe" || true
}

bench_procs() {
    tasklist //FI "IMAGENAME eq qwenox-bench.exe" 2>/dev/null \
        | grep -c "qwenox-bench.exe" || true
}

gpu_used_gib() {  # 每适配器 (Dedicated+Shared) 之和的最大值（GiB，一位小数）
    # 本机 gfx1151 iGPU 的 87 GiB 大分配计在 Dedicated 而非 Shared，两者都要算。
    powershell -NoProfile -Command '
        $c = (Get-Counter "\GPU Adapter Memory(*)\Dedicated Usage","\GPU Adapter Memory(*)\Shared Usage" -ErrorAction SilentlyContinue).CounterSamples
        $g = @{}
        foreach ($s in $c) { $g[$s.InstanceName] = [double]$g[$s.InstanceName] + $s.CookedValue }
        ($g.Values | Measure-Object -Maximum).Maximum
    ' 2>/dev/null | tr -d '\r' | awk '{printf "%.1f", $1/1073741824}'
}

if [[ "${1:-}" == --gpu ]]; then
    echo "GPU 显存（dedicated+shared，最大适配器）: $(gpu_used_gib) GiB"
    exit 0
fi

compare() {  # compare <labelA> <labelB>
    python - "$RESULTS_DIR/$1.summary" "$RESULTS_DIR/$2.summary" <<'PYEOF'
import sys
def load(p):
    rows = []
    for line in open(p, encoding='utf-8'):
        f = line.rstrip('\n').split('\t')
        if f[0] in ('prefill', 'decode') and len(f) >= 4:
            rows.append((f[0], f[1], float(f[3]), float(f[4]) if len(f) > 4 else None))
    return rows
a = load(sys.argv[1]); b = load(sys.argv[2])
bmap = {(k, n): (v, ac) for k, n, v, ac in b}
print('%-26s %10s %10s %9s' % ('item', 'base', 'new', 'delta'))
for k, n, v, ac in a:
    if (k, n) not in bmap:
        continue
    v2 = bmap[(k, n)][0]
    d = (v2 / v - 1) * 100 if v else 0.0
    print('%-26s %10.1f %10.1f %8.1f%%' % (k + ' ' + n, v, v2, d))
acc = [(n, ac, bmap[(k, n)][1]) for k, n, v, ac in a
       if k == 'decode' and ac is not None and (k, n) in bmap and bmap[(k, n)][1] is not None]
if acc:
    print()
    print('%-26s %10s %10s %9s' % ('decode MTP accept', 'base %', 'new %', 'delta pp'))
    for n, ac, ac2 in acc:
        print('%-26s %10.1f %10.1f %+8.1f' % (n, ac, ac2, ac2 - ac))
PYEOF
}

if [[ "${1:-}" == --compare ]]; then
    [[ $# == 3 ]] || die "用法: $0 --compare <labelA> <labelB>"
    [[ -f "$RESULTS_DIR/$2.summary" ]] || die "缺 $RESULTS_DIR/$2.summary"
    [[ -f "$RESULTS_DIR/$3.summary" ]] || die "缺 $RESULTS_DIR/$3.summary"
    compare "$2" "$3"
    exit $?
fi

[[ $# -ge 1 ]] || { sed -n '2,22p' "$0" >&2; exit 2; }
label="$(printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_')"; shift
[[ -f "$QWENOX_DIR/service.conf" ]] || die "找不到 $QWENOX_DIR/service.conf"
[[ -x "$BENCH" ]] || die "找不到 bench：$BENCH（bash build_win.sh bench 编译）"
mkdir -p "$RESULTS_DIR"
SAMPLER_LOG="$RESULTS_DIR/$label.gpu.log"

# 默认 env = Windows 生产启动器的那套开关（KVSNAP/RCKPT 关，见文件头注释）。
DEFAULT_ENV=(
    QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1
    QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1
    QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 QWENOX_NOWARMUP=1
    QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1 QWENOX_INDEX_FUSED2=1
    QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1
    QWENOX_KV_PAGED=1 QWENOX_KVSNAP=0 QWENOX_RCKPT_MAX=0
)
ENVS=()
for kv in "${DEFAULT_ENV[@]}"; do
    key="${kv%%=*}"
    for ov in "$@"; do
        [[ "${ov%%=*}" == "$key" ]] && continue 2
    done
    ENVS+=("$kv")
done
ENVS+=("$@")

RESTORE=0
BENCH_STARTED=0
SAMPLER_PID=""
cleanup() {
    if [[ -n "$SAMPLER_PID" ]]; then kill "$SAMPLER_PID" 2>/dev/null; fi
    if [[ "$BENCH_STARTED" == 1 ]]; then
        # 异常退出兜底：清掉还在跑（占着 87 GiB）的 bench。
        if [[ "$(bench_procs)" != 0 ]]; then
            echo "清理：终止仍在运行的 qwenox-bench.exe"
            taskkill //IM qwenox-bench.exe //F >/dev/null 2>&1
        fi
    fi
    if [[ "$RESTORE" == 1 ]]; then
        echo
        echo "== 恢复引擎（POST /admin/engine/start）=="
        api -X POST "http://127.0.0.1:$API_PORT/admin/engine/start" | head -c 300; echo
        echo "引擎后台加载权重约 2–3 分钟；进度看 $QWENOX_DIR/logs/engine-api-*.log 或网页 #/engine"
    fi
}
trap cleanup EXIT

echo "GPU 显存（当前）: $(gpu_used_gib) GiB"
state="$(engine_state)"
if [[ -z "$state" ]]; then
    echo "提示：API 127.0.0.1:$API_PORT 不可达，跳过停引擎（只做显存校验）"
elif [[ "$state" == running || "$state" == starting ]]; then
    echo "== 停引擎（POST /admin/engine/stop）=="
    api -X POST "http://127.0.0.1:$API_PORT/admin/engine/stop" | head -c 300; echo
    RESTORE=1
else
    echo "引擎状态：$state（无需停）"
fi

# ---- 显存互斥硬校验：进程消失 + 共享显存回落，二者都要满足 ----
for _ in $(seq 1 120); do
    [[ "$(engine_procs)" == 0 ]] && break
    sleep 1
done
[[ "$(engine_procs)" == 0 ]] || die "引擎进程 120s 内未退出，拒绝启动 bench（防显存叠加）"
for _ in $(seq 1 60); do
    g="$(gpu_used_gib)"
    awk -v g="$g" -v m="$VRAM_FREE_MAX_GB" 'BEGIN{exit !(g+0 <= m+0)}' && break
    sleep 2
done
g="$(gpu_used_gib)"
awk -v g="$g" -v m="$VRAM_FREE_MAX_GB" 'BEGIN{exit !(g+0 <= m+0)}' \
    || die "GPU 显存仍占 ${g} GiB（上限 ${VRAM_FREE_MAX_GB}），拒绝启动 bench（防显存叠加）"
echo "显存互斥校验通过：引擎进程 0 个，GPU 显存 ${g} GiB"
IDLE_GIB="$g"

echo
echo "== 跑基准：label=$label =="
printf '   env:'; for kv in "${ENVS[@]}"; do printf ' %s' "$kv"; done; echo
: > "$SAMPLER_LOG"
(
    while :; do
        printf '%s %s\n' "$(date +%H:%M:%S)" "$(gpu_used_gib)" >> "$SAMPLER_LOG"
        sleep 15
    done
) &
SAMPLER_PID=$!
BENCH_STARTED=1
# 注意：不要用 MSYS 的 env(1) 前缀（实测 `env VAR=1 ./qwenox-bench.exe run` 会静默退出、
# 无任何输出）；用 bash 自己的 export 传环境变量。
export "${ENVS[@]}"
( cd "$QWENOX_DIR" && exec "$BENCH" run ) > "$RESULTS_DIR/$label.log" 2>&1
rc=$?
BENCH_STARTED=0
kill "$SAMPLER_PID" 2>/dev/null; SAMPLER_PID=""
tail -n 32 "$RESULTS_DIR/$label.log"
[[ "$rc" == 0 ]] || echo "警告：bench 退出码 $rc（完整日志 $RESULTS_DIR/$label.log）"

python - "$RESULTS_DIR/$label.log" "$RESULTS_DIR/$label.summary" "$label" "${ENVS[*]}" \
    "$(sha1sum "$QWENOX_DIR/service.conf" | cut -d' ' -f1)" "$IDLE_GIB" "$SAMPLER_LOG" <<'PYEOF'
import sys
log_path, out_path, label, envs, conf_sha, idle_gib, gpu_log = sys.argv[1:8]
rows = []
mode = None
for line in open(log_path, encoding='utf-8', errors='replace'):
    line = line.rstrip('\n')
    if line.startswith('Prefill ('):
        mode = 'prefill'; continue
    if line.startswith('Decode / TG'):
        mode = 'decode'; continue
    if not line.startswith('  '):
        continue
    parts = line.split()
    if not parts or parts[0] in ('token', 'scenario'):
        continue
    if mode == 'prefill':
        if parts[0] == 'TOTAL':
            rows.append(('prefill', 'TOTAL', '', parts[2]))
        elif len(parts) >= 3:
            rows.append(('prefill', parts[0], parts[1], parts[2]))
    elif mode == 'decode':
        if parts[0] == 'TOTAL':
            rows.append(('decode', 'TOTAL', '', parts[2], parts[3].rstrip('%')))
        elif len(parts) >= 4:
            rows.append(('decode', parts[0], parts[1], parts[2], parts[3].rstrip('%')))
peak = 0.0
try:
    for line in open(gpu_log, encoding='utf-8', errors='replace'):
        f = line.split()
        if len(f) == 2:
            peak = max(peak, float(f[1]))
except FileNotFoundError:
    pass
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('meta\tlabel\t%s\n' % label)
    f.write('meta\tconf_sha1\t%s\n' % conf_sha)
    f.write('meta\tgpu_idle_gib\t%s\n' % idle_gib)
    f.write('meta\tgpu_peak_gib\t%.1f\n' % peak)
    f.write('meta\tenv\t%s\n' % envs)
    for r in rows:
        f.write('\t'.join(r) + '\n')
print('摘要: %d 行（prefill %d / decode %d），共享显存峰值 %.1f GiB（采样 %s）→ %s' %
      (len(rows), sum(1 for r in rows if r[0] == 'prefill'),
       sum(1 for r in rows if r[0] == 'decode'), peak, gpu_log, out_path))
PYEOF

if [[ -n "${BASELINE:-}" ]]; then
    if [[ -f "$RESULTS_DIR/$BASELINE.summary" ]]; then
        echo
        echo "== 对比 baseline=$BASELINE vs new=$label =="
        compare "$BASELINE" "$label"
    else
        echo "提示：缺 $RESULTS_DIR/$BASELINE.summary，未对比"
    fi
fi
