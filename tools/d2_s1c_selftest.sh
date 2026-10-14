#!/usr/bin/env bash
# d2_s1c_selftest.sh — D2-S1c 验收②：多段 verify 一次调用 vs 各段串行逐位一致。
# QWENOX_VSEG_SELFTEST="3,4,5"：三段（P=3/4/5，base=2048/4097/6146，%4=0/1/2），
# 串行单段 verify_segments（生产路径）vs 一次多段 verify_segments，比 ids +
# 每槽 d_S/d_convst/d_ring/d_iraw/d_R/pos/chist 哈希。exit 0 = PASS。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BIN=build/qwenox-engine-win.exe
M=./models/qwen38-flash-next-v2.hgn
NG=./models/qwen38-flash-next-ngram.hgn
MTP=./models/qwen38-flash-next-mtp.hgn

export QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1
export QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1
export QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 QWENOX_NOWARMUP=1
export QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1
export QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1
export QWENOX_KVSNAP=0 QWENOX_RCKPT_MAX=0 QWENOX_KV_PAGED=1 QWENOX_PARALLEL=1
export QWENOX_ROPE_FACTOR=1 QWENOX_ROPE_ORIGINAL_CTX=262144
export QWENOX_ROPE_BETA_FAST=32 QWENOX_ROPE_BETA_SLOW=1 QWENOX_ROPE_ATTN_SCALE=0

SPEC="${1:-3,4,5}"
MAXCTX="${2:-16384}"  # 段数多/段落长时调大（共享 KV 池 = maxctx）
echo "== d2_s1c_selftest $(date '+%F %T')  QWENOX_VSEG_SELFTEST=$SPEC maxctx=$MAXCTX"
QWENOX_VSEG_SELFTEST="$SPEC" "$BIN" "$M" "$NG" "$MTP" --maxctx "$MAXCTX" \
    > logs/d2s1c_selftest.log 2>&1
rc=$?
tail -5 logs/d2s1c_selftest.log
if [ $rc = 0 ]; then echo "S1C SELFTEST: PASS"; else echo "S1C SELFTEST: FAIL rc=$rc"; fi
exit $rc
