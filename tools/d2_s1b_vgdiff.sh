#!/usr/bin/env bash
# d2_s1b_vgdiff.sh — D2-S1b plan B：QWENOX_VGDUMP_ROWS=1 逐行 hash 打点，
# γ=2(P=3) vs γ=3(P=4) 各跑一次 --spec-gen 2，按 tag 出现顺序配对，
# 找第一个共享行（row 0..2）hash 不同的埋点 = 首个 P 变源的下游。
#   bash tools/d2_s1b_vgdiff.sh
# 日志 logs/d2s1b_vgd_g2.log / _g3.log。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BIN=build/qwenox-engine-win.exe
M=./models/qwen38-flash-next-v2.hgn
NG=./models/qwen38-flash-next-ngram.hgn
MTP=./models/qwen38-flash-next-mtp.hgn
TOK=logs/d2gw_tok8192.txt
[[ -x $BIN && -f $TOK ]] || { echo "缺 $BIN 或 $TOK"; exit 1; }

run() {  # run GAMMA
  local G=$1
  echo "  .. γ=$G $(date +%T)" >&2
  ( export QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1
    export QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1
    export QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 QWENOX_NOWARMUP=1
    export QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1
    export QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1
    export QWENOX_KVSNAP=0 QWENOX_RCKPT_MAX=0 QWENOX_KV_PAGED=1 QWENOX_PARALLEL=1
    export QWENOX_ROPE_FACTOR=1 QWENOX_ROPE_ORIGINAL_CTX=262144
    export QWENOX_ROPE_BETA_FAST=32 QWENOX_ROPE_BETA_SLOW=1 QWENOX_ROPE_ATTN_SCALE=0
    export QWENOX_VGDUMP=1 QWENOX_VGDUMP_ROWS=1 QWENOX_FORK_DUMP=8194
    "$BIN" "$M" "$NG" "$MTP" --tokens-file "$TOK" --maxctx 16384 \
      --spec-gen "${SG:-2}" --gamma "$G" ) > "logs/d2s1b_vgd_g$G.log" 2>&1
}

run 2
run 3

python - <<'EOF'
import re, itertools
def tags(path):
    out = []
    for line in open(path, errors='replace'):
        m = re.match(r'\[(\w+)\] P=(\d+)(?: base=(-?\d+))? rows((?: [0-9a-f]{16})+)', line)
        if m:
            out.append((m.group(1), int(m.group(2)),
                        m.group(3), m.group(4).split()))
    return out
a = tags('logs/d2s1b_vgd_g2.log')
b = tags('logs/d2s1b_vgd_g3.log')
print(f'γ=2 埋点 {len(a)} 个，γ=3 {len(b)} 个')
first = None
for i, (ta, tb) in enumerate(itertools.zip_longest(a, b)):
    if ta is None or tb is None or ta[0] != tb[0]:
        print(f'序列在第 {i} 个埋点错位: {ta and ta[0]} vs {tb and tb[0]}')
        break
    tag, pa, base, ha = ta
    hb = tb[3]
    for r in range(min(pa, tb[1])):
        if ha[r] != hb[r]:
            print(f'首个分叉埋点 #{i}: [{tag}] row{r}: γ2={ha[r]} γ3={hb[r]}')
            first = i
            break
    if first is not None:
        for j in range(max(0, i - 2), min(len(a), i + 3)):
            ta2, tb2 = a[j], b[j]
            same = all(ta2[3][r] == tb2[3][r] for r in range(min(ta2[1], tb2[1])))
            print(f'  ctx #{j} [{ta2[0]}] P={ta2[1]}/{tb2[1]} base={ta2[2]} {"SAME" if same else "DIFF"}')
        break
else:
    print('所有共享行一致（分叉在埋点之外）')
EOF
