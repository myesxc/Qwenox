#!/usr/bin/env bash
# d2_s1b_sw.sh — 开关二分（敏感版）：eager + VGDUMP_ROWS + spec-gen 16，
# γ=2/3 各一跑后用 d2_s1b_vgcmp.py 对齐比较，报告首个分叉 (abspos, tag)。
# 用法: bash tools/d2_s1b_sw.sh <tag> [EXTRA_ENV="K=V K=V"]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
TAG="$1"
EXTRA="${EXTRA_ENV:-}"
BIN=build/qwenox-engine-win.exe
M=./models/qwen38-flash-next-v2.hgn
NG=./models/qwen38-flash-next-ngram.hgn
MTP=./models/qwen38-flash-next-mtp.hgn
TOK=logs/d2gw_tok8192.txt

BASE_ENV=(
  QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1
  QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1
  QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 QWENOX_NOWARMUP=1
  QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1
  QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1
  QWENOX_KVSNAP=0 QWENOX_RCKPT_MAX=0 QWENOX_KV_PAGED=1 QWENOX_PARALLEL=1
  QWENOX_ROPE_FACTOR=1 QWENOX_ROPE_ORIGINAL_CTX=262144
  QWENOX_ROPE_BETA_FAST=32 QWENOX_ROPE_BETA_SLOW=1 QWENOX_ROPE_ATTN_SCALE=0
  QWENOX_VGDUMP=1 QWENOX_VGDUMP_ROWS=1 QWENOX_VGRAPH=0
)

for G in 2 3; do
  ( export "${BASE_ENV[@]}"; for kv in $EXTRA; do export "$kv"; done
    "$BIN" "$M" "$NG" "$MTP" --tokens-file "$TOK" --maxctx 16384 \
      --spec-gen "${SG:-16}" --gamma "$G" ) > "logs/d2s1b_sw_${TAG}_g${G}.log" 2>&1 \
    || echo "  FAIL γ=$G rc=$?"
done
python tools/d2_s1b_vgcmp.py "logs/d2s1b_sw_${TAG}_g2.log" "logs/d2s1b_sw_${TAG}_g3.log" \
  | sed -n '1,3p;4,6p' | head -8
