#!/usr/bin/env bash
# d2_s1b_probe.sh — 时序隐患开关二分：同一 base_env 下叠加额外 env，跑 γ=2/3，
# 报告首个 ids 分叉下标（全同则印 IDENTICAL）。
# 用法：bash tools/d2_s1b_probe.sh <tag> [EXTRA_ENV="K=V K=V"]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
TAG="$1"
EXTRA="${EXTRA_ENV:-}"
BIN=build/qwenox-engine-win.exe
M=./models/qwen38-flash-next-v2.hgn
NG=./models/qwen38-flash-next-ngram.hgn
MTP=./models/qwen38-flash-next-mtp.hgn
TOK=logs/d2gw_tok8192.txt

export QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1
export QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1
export QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 QWENOX_NOWARMUP=1
export QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1
export QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1
export QWENOX_KVSNAP=0 QWENOX_RCKPT_MAX=0 QWENOX_KV_PAGED=1 QWENOX_PARALLEL=1
export QWENOX_ROPE_FACTOR=1 QWENOX_ROPE_ORIGINAL_CTX=262144
export QWENOX_ROPE_BETA_FAST=32 QWENOX_ROPE_BETA_SLOW=1 QWENOX_ROPE_ATTN_SCALE=0
# 叠加实验开关
for kv in $EXTRA; do export "$kv"; done

for G in 2 3; do
  "$BIN" "$M" "$NG" "$MTP" --tokens-file "$TOK" --maxctx 16384 \
      --spec-gen 256 --gamma "$G" > "logs/d2s1b_${TAG}_g${G}.log" 2>&1 \
      || echo "  FAIL γ=$G rc=$?"
done
python - "$TAG" <<'EOF'
import sys, re
tag = sys.argv[1]
ids = {}
for g in (2, 3):
    last = None
    with open(f"logs/d2s1b_{tag}_g{g}.log", errors="replace") as f:
        for line in f:
            if line.startswith("ids:"):
                last = line
    ids[g] = [int(x) for x in last.split()[1:]] if last else []
a, b = ids[2], ids[3]
n = min(len(a), len(b))
fork = next((i for i in range(n) if a[i] != b[i]), None)
if fork is None and len(a) == len(b) and n:
    print(f"PROBE {tag}: IDENTICAL ({n} ids)")
else:
    va = a[fork] if fork is not None and fork < len(a) else "?"
    vb = b[fork] if fork is not None and fork < len(b) else "?"
    print(f"PROBE {tag}: FORK at idx {fork} (γ2={va} vs γ3={vb}), lens {len(a)}/{len(b)}")
EOF
