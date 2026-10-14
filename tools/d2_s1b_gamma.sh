#!/usr/bin/env bash
# d2_s1b_gamma.sh — D2-S1b 验收：q6g64 split-K 门控 P 无关化之后，
# γ 扫描（默认 1 2 3 4 5 7 0）的 --spec-gen 256 ids 尾部必须两两逐 bit 全同。
# 用法：bash tools/d2_s1b_gamma.sh          # 全量（约 7 分钟）
#       GAMMAS="2 3" bash tools/d2_s1b_gamma.sh
# 日志 logs/d2s1b_g<G>.log；结论打 stdout。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BIN=build/qwenox-engine-win.exe
M=./models/qwen38-flash-next-v2.hgn
NG=./models/qwen38-flash-next-ngram.hgn
MTP=./models/qwen38-flash-next-mtp.hgn
TOK=logs/d2gw_tok8192.txt
GAMMAS="${GAMMAS:-1 2 3 4 5 7 0}"
[[ -x $BIN && -f $TOK ]] || { echo "缺 $BIN 或 $TOK"; exit 1; }

export QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1
export QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1
export QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 QWENOX_NOWARMUP=1
export QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1
export QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1
export QWENOX_KVSNAP=0 QWENOX_RCKPT_MAX=0 QWENOX_KV_PAGED=1 QWENOX_PARALLEL=1
export QWENOX_ROPE_FACTOR=1 QWENOX_ROPE_ORIGINAL_CTX=262144
export QWENOX_ROPE_BETA_FAST=32 QWENOX_ROPE_BETA_SLOW=1 QWENOX_ROPE_ATTN_SCALE=0

ids_tail() { grep -a '^ids:' "logs/d2s1b_g$1.log" | tail -1 | cut -d' ' -f8194-; }

echo "== d2_s1b_gamma $(date '+%F %T') γ=$GAMMAS"
ref=""
refg=""
ok=1
for G in $GAMMAS; do
  echo "  .. γ=$G $(date +%T)"
  "$BIN" "$M" "$NG" "$MTP" --tokens-file "$TOK" --maxctx 16384 \
      --spec-gen 256 --gamma "$G" > "logs/d2s1b_g$G.log" 2>&1 || echo "  FAIL γ=$G rc=$?"
  I=$(ids_tail "$G")
  S=$(grep -a '^spec:' "logs/d2s1b_g$G.log" | tail -1)
  if [[ -z "$ref" ]]; then ref="$I"; refg=$G; fi
  if [[ "$I" == "$ref" && -n "$I" ]]; then r="≡γ$refg"; else r="DIFF"; ok=0; fi
  printf '  γ=%-2s %-6s %s\n' "$G" "$r" "$S"
done
echo "S1B GAMMA-INVARIANCE: $([ $ok = 1 ] && echo PASS || echo FAIL)"
