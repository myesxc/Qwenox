#!/usr/bin/env bash
# d2_gate_win.sh — tools/d2_gate_check.sh 的 Windows 精简版：同一 token 源
# （data/qsa-oracle/131072.tokens）、同一组公式（上限/估算/理想），env 直写
# （start_win.sh 同款生产内核组合）。悬崖修复（gemv_multi_q4cp P≤24）后复测。
#   bash tools/d2_gate_win.sh            # 完整矩阵（约 30-40 分钟）
#   CTXS=8192 bash tools/d2_gate_win.sh  # 只跑 8K
# 日志 logs/d2gw_*.log，汇总直接打在 stdout。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BIN=build/qwenox-engine-win.exe
M=./models/qwen38-flash-next-v2.hgn
NG=./models/qwen38-flash-next-ngram.hgn
MTP=./models/qwen38-flash-next-mtp.hgn
SRC=data/qsa-oracle/131072.tokens
CTXS="${CTXS:-8192 65536}"
GAMMAS="${GAMMAS:-1 2 3 4 5 7 0}"
GATE="${GATE:-1.25}"
[[ -x $BIN && -f $SRC ]] || { echo "缺 $BIN 或 $SRC"; exit 1; }

export QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1
export QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1
export QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 QWENOX_NOWARMUP=1
export QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1
export QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1
export QWENOX_KVSNAP=0 QWENOX_RCKPT_MAX=0 QWENOX_KV_PAGED=1 QWENOX_PARALLEL=1
export QWENOX_ROPE_FACTOR=1 QWENOX_ROPE_ORIGINAL_CTX=262144
export QWENOX_ROPE_BETA_FAST=32 QWENOX_ROPE_BETA_SLOW=1 QWENOX_ROPE_ATTN_SCALE=0

mkdir -p logs
head -n 1024 "$SRC" > logs/d2gw_tok1024.txt
for N in $CTXS; do head -n "$N" "$SRC" > "logs/d2gw_tok$N.txt"; done

PS_1K="1,2,3,4,5,6,7,8"
# T(R×B)：B=2 → p≤12；B=4 → p≤6；B=8 → p≤3（R≤24 新上限）
PS_BIG="1,2,3,4,5,6,7,8,9,10,12,14,16,20,24,4x2,6x2,8x2,10x2,16x2,8x4,12x4,16x4,24x4,16x8,24x8"

run() {  # run LABEL TOKFILE MAXCTX [ENV=...] [-- 引擎额外参数]
  local L=$1 T=$2 C=$3; shift 3
  local E=()
  while [[ $# -gt 0 && $1 != -- ]]; do E+=("$1"); shift; done
  [[ ${1:-} == -- ]] && shift
  echo "  .. $L $(date +%T)"
  env "${E[@]}" "$BIN" "$M" "$NG" "$MTP" --tokens-file "$T" --maxctx "$C" "$@" \
      > "logs/d2gw_$L.log" 2>&1 || { echo "  FAIL $L rc=$?"; tail -3 "logs/d2gw_$L.log"; }
}

echo "== d2_gate_win $(date '+%F %T') GATE=$GATE CTXS=$CTXS"
if [[ -z "${SKIP_VB:-}" ]]; then
run vb1024 logs/d2gw_tok1024.txt 9216 QWENOX_VBENCH=$PS_1K
for N in $CTXS; do
  C=$(( N + 8192 ))
  run "vb$N" "logs/d2gw_tok$N.txt" "$C" QWENOX_VBENCH=$PS_BIG
  grep -a '^vbench' "logs/d2gw_vb$N.log" | sed 's/^/     /'
done
fi
for N in $CTXS; do
  C=$(( N + 8192 ))
  for G in $GAMMAS; do
    run "g${N}_$G" "logs/d2gw_tok$N.txt" "$C" -- --spec-gen 256 --gamma "$G"
    grep -a '^spec:' "logs/d2gw_g${N}_$G.log" | tail -1 | sed "s/^/     γ=$G /"
  done
done

python3 - "$GATE" $CTXS <<'PY'
import re, sys
gate = float(sys.argv[1]); ctxs = [int(x) for x in sys.argv[2:]]
spec_re = re.compile(r'^spec: \d+ tokens in ([\d.]+) s = [\d.]+ tok/s \| rounds=(\d+) commit/round=([\d.]+)')

def vb(path):
    T = {}
    for ln in open(path, errors='replace'):
        m = re.match(r'vbench P=(\d+)(?: segs=(\d+))? median=([\d.]+)', ln)
        if m: T[(int(m[1]), int(m[2] or 1))] = float(m[3])
    return T

def spec(path):
    S = {}
    for ln in open(path, errors='replace'):
        m = spec_re.match(ln)
        if m: return float(m[1]) * 1000 / int(m[2]), float(m[3])  # Tr ms, E
    return None

T1K = vb('logs/d2gw_vb1024.log')
print(f'\nT_1K: ' + '  '.join(f'{p}:{v:.1f}' for (p, b), v in sorted(T1K.items())))
worst_est = 0.0
for N in ctxs:
    T = vb(f'logs/d2gw_vb{N}.log')
    G = {}
    for g in (1, 2, 3, 4, 5, 7, 0):
        s = spec(f'logs/d2gw_g{N}_{g}.log')
        if s: G[g] = s
    if not G: continue
    base_g = max(G, key=lambda g: G[g][1] / G[g][0])
    base_tps = G[base_g][1] / G[base_g][0] * 1000
    C = {p: max(0.0, T.get((p, 1), 0) - T1K.get((p, 1), 0)) for p in range(1, 9)}
    print(f'\n==== ctx {N} ====  分时基线 γ={base_g} {base_tps:.1f} tok/s')
    print('  T(P):   ' + '  '.join(f'{p}:{v:.1f}' for (p, b), v in sorted(T.items()) if b == 1))
    print('  T(RxB): ' + '  '.join(f'{p}x{b}:{v:.1f}' for (p, b), v in sorted(T.items()) if b > 1))
    print('  C(p):   ' + '  '.join(f'{p}:{v:.1f}' for p, v in sorted(C.items())))
    print(f'  {"B":>2} {"γ":>2} {"R":>3} {"上限":>6} {"估算":>6} {"理想":>6}')
    for B in (2, 4, 8):
        for g in sorted(x for x in G if x):
            p = g + 1; R = B * p
            if R > 24 or (p, 1) not in T or (R, 1) not in T or (R, B) not in T: continue
            Tr, E = G[g]
            up = B * Tr - (B * T[(p, 1)] - T[(R, 1)])
            est = B * Tr - (B * T[(p, 1)] - T[(R, B)]) + (B - 1) * C.get(p, 0)
            ideal = Tr + (T[(R, B)] - T[(p, 1)]) + (B - 1) * C.get(p, 0)
            f = lambda rd: B * E / rd * 1000 / base_tps
            r_up, r_est, r_id = f(up), f(est), f(ideal)
            worst_est = max(worst_est, r_est)
            mark = ' *' if r_est >= gate else ''
            print(f'  {B:>2} {g:>2} {R:>3} {r_up:>6.2f} {r_est:>6.2f} {r_id:>6.2f}{mark}')
print(f'\n估算最高 {worst_est:.2f}x vs 闸门 {gate}x ->',
      'D2 不值得做的结论仍成立' if worst_est < gate else '** 超闸：D2 值得重新评估 **')
PY
