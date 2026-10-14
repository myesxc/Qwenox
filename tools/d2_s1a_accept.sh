#!/usr/bin/env bash
# d2_s1a_accept.sh — D2-S1a 验收：verify_segments 单段版逐 bit。
# 三次 --spec-gen 256 --gamma 4（8192 ctx），ids 行须与 logs/s0_smoke.log 完全一致：
#   A 默认（verify graphs 开——走图路径，不触新代码，回归基线）
#   B QWENOX_VGRAPH=0（eager + verify_segments 单段，新代码真实执行）
#   C QWENOX_VGRAPH=0 QWENOX_NO_VSEG=1（eager 旧路径，隔离对照）
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
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
# 基线 logs/s0_smoke.log 是 fp32 dense 路径录的；本测验的是 verify 图/eager 逐位一致，与 dense WMMA 无关
export QWENOX_DENSE_WMMA=0
export QWENOX_KVSNAP=0 QWENOX_RCKPT_MAX=0 QWENOX_KV_PAGED=1 QWENOX_PARALLEL=1
export QWENOX_ROPE_FACTOR=1 QWENOX_ROPE_ORIGINAL_CTX=262144
export QWENOX_ROPE_BETA_FAST=32 QWENOX_ROPE_BETA_SLOW=1 QWENOX_ROPE_ATTN_SCALE=0

run() {  # run LABEL [ENV...]
  local L=$1; shift
  echo "  .. $L $(date +%T)"
  env "$@" "$BIN" "$M" "$NG" "$MTP" --tokens-file "$TOK" --maxctx 16384 \
      --spec-gen 256 --gamma 4 > "logs/d2s1a_$L.log" 2>&1 || echo "  FAIL $L rc=$?"
}
ids_tail() { grep -a '^ids:' "logs/d2s1a_$1.log" | tail -1 | cut -d' ' -f8194- | md5sum | cut -d' ' -f1; }

ref=$(grep -a '^ids:' logs/s0_smoke.log | tail -1 | cut -d' ' -f8194- | md5sum | cut -d' ' -f1)
echo "== d2_s1a_accept $(date '+%F %T')  ref(s0_smoke tail)=$ref"
run A_graphs
run B_vseg   QWENOX_VGRAPH=0
run C_oldpath QWENOX_VGRAPH=0 QWENOX_NO_VSEG=1
ok=1
for L in A_graphs B_vseg C_oldpath; do
  h=$(ids_tail $L)
  s=$(grep -a '^spec:' "logs/d2s1a_$L.log" | tail -1)
  if [[ "$h" == "$ref" && -n "$h" ]]; then r=BIT-IDENTICAL; else r=DIFF; ok=0; fi
  printf '  %-10s %s  %s\n' "$L" "$r" "$s"
done
echo "S1A ACCEPT: $([ $ok = 1 ] && echo PASS || echo FAIL)"
