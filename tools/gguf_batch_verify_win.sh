#!/usr/bin/env bash
# gguf_batch_verify_win.sh — Windows：GGUF 多行 dense（k_q8g32_wmma）+ 专家去重 GEMV ≤48 行
# 的一键检查（约 25 分钟）。新（默认）vs 旧（QWENOX_DENSE_WMMA_Q8=0 QWENOX_GG_GEMV=8）同一二进制。
#
#   bash tools/gguf_batch_verify_win.sh
#   ONLY="vb dec ppl vseg conc" bash tools/gguf_batch_verify_win.sh
#
# 门槛（结尾 GGUF_BATCH PASS / FAIL）：
#   kt    tools/ht_wmma_test.sh q8（dense kernel 精度 / 行无关 / 切片）+ 专家去重 GEMV P≤48 与普通 GEMV 逐位
#   vseg  QWENOX_VSEG_SELFTEST=3,4,5 与 5,5,5,5,5,5,5,5（批 verify 与逐段 verify 的 ids 和每路状态逐位相同）
#   ppl   2048.tokens PPL：新 vs 旧 相对差 < 0.5%；新的 chunk 2048 vs chunk 4 相对差 < 0.5%
#   conc  conc_verify：单路串行 vs 8 路并发，全部 flow 逐 token 相同（MATCH）+ 控制面
# 只报告：vbench P=1/5/8/16/40/40x8（ms）、--gen 128 decode、--spec-gen 256 γ4（2048/8192）。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$(pwd)"
GGUF="${GGUF:-/c/Users/Mark/Models/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}"
MTPG="${MTPG:-$(dirname "$GGUF")/mtp-Qwen3.8-Flash-Next-Q8_0.gguf}"
BIN=$ROOT/build/qwenox-engine-win.exe
L=logs/gbv; mkdir -p $L logs/conc
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
FAILS=()
fail() { FAILS+=("$1"); echo "  FAIL: $1"; }
pass() { echo "  PASS: $1"; }
info() { echo "  INFO: $1"; }
stamp() { echo; echo "=== [$(date +%H:%M:%S)] $*"; }
step() { [[ -z "${ONLY:-}" || " $ONLY " == *" $1 "* ]]; }
num() { sed -n "s#.*$1=\([0-9.e+-]*\).*#\1#p" | head -1; }
for f in "$GGUF" "$MTPG" "$BIN"; do [[ -f "$f" ]] || { echo "缺文件：$f"; echo "GGUF_BATCH FAIL"; exit 1; }; done
if tasklist 2>/dev/null | grep -qi "qwenox"; then
  echo "检测到 qwenox 进程在运行，请先停掉服务"; echo "GGUF_BATCH FAIL"; exit 1
fi
echo "head $(git log --oneline -1 | cut -c1-70)  dirty: $(git status --short | wc -l)"
GW=$(cygpath -m "$GGUF"); MW=$(cygpath -m "$MTPG")
OLD=(QWENOX_DENSE_WMMA_Q8=0 QWENOX_GG_GEMV=8)

prod_env() {
  for e in $(compgen -e | grep '^QWENOX_'); do unset "$e"; done
  export QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1
  export QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1
  export QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 QWENOX_NOWARMUP=1
  export QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1
  export QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1
  export QWENOX_KV_PAGED=1 QWENOX_KVSNAP=0 QWENOX_RCKPT_MAX=0
  export QWENOX_ROPE_FACTOR=1 QWENOX_ROPE_ORIGINAL_CTX=262144
  export QWENOX_ROPE_BETA_FAST=32 QWENOX_ROPE_BETA_SLOW=1 QWENOX_ROPE_ATTN_SCALE=0
}
# eng TAG [ENV=V ...] -- ARGS   -> $L/TAG.{out,err}
eng() {
  local tag=$1; shift
  local E=()
  while [[ $# -gt 0 && $1 != -- ]]; do E+=("$1"); shift; done
  shift
  prod_env
  local t0=$(date +%s)
  (cd build && env "${E[@]}" ./qwenox-engine-win.exe "$GW" "$@" > "$ROOT/$L/$tag.out" 2> "$ROOT/$L/$tag.err")
  local rc=$?
  echo "  [$tag] rc=$rc $(( $(date +%s) - t0 ))s"
  [[ $rc == 0 ]] || tr '\r' '\n' < "$L/$tag.err" | grep -vE "pload [0-9]" | tail -4
  return $rc
}
errl() { tr '\r' '\n' < "$L/$1.err"; }
TOK2K=$ROOT/data/qsa-oracle/2048.tokens

# ---------------------------------------------------------------- kt
if step kt; then
  stamp "kt：dense q8 kernel 单测 + 专家去重 GEMV P≤48"
  bash tools/ht_wmma_test.sh q8 > $L/q8_wmma_test.log 2>&1
  grep -E "per decode|TEST" $L/q8_wmma_test.log | sed 's/^/    /'
  grep -q "Q8_WMMA TEST: PASS" $L/q8_wmma_test.log && pass "q8_wmma_test" || fail "q8_wmma_test（见 $L/q8_wmma_test.log）"
  TR=/c/therock-dist-windows-gfx1151-10.1.0
  HIP_PATH="$(cygpath -w "$TR")" "$TR/bin/hipcc.exe" -O3 -std=c++17 --offload-arch=gfx1151 -D_CRT_SECURE_NO_WARNINGS \
    -D_CRT_NONSTDC_NO_DEPRECATE -I"$TR/include" tools/moe_gguf_gemv_test.cu -o build/iqt/moe_gguf_gemv_test.exe \
    > $L/gemvt_build.log 2>&1 || fail "编译 moe_gguf_gemv_test"
  for LY in 0 2 30; do for P in 2 9 24 48; do
    out=$(cd build && ./iqt/moe_gguf_gemv_test.exe "$GW" --layer $LY --P $P --iters 5 --share 0.3 2>&1)
    echo "$out" | tail -1 | grep -q "^PASS" && pass "专家 GEMV L$LY P=$P（$(echo "$out" | grep -o 'dedup vs plain: [a-z -]*, out [a-z-]*')）" \
      || { fail "专家 GEMV L$LY P=$P"; echo "$out" | tail -4; }
  done; done
fi

# ---------------------------------------------------------------- vb
if step vb; then
  stamp "vbench（同一步算 P 行，ms，1024 token 前缀）"
  head -n 1024 data/qsa-oracle/131072.tokens > $L/tok1024.txt
  for v in new old; do
    if [[ $v == new ]]; then E=(QWENOX_GGUF_MTP="$MW"); else E=(QWENOX_GGUF_MTP="$MW" "${OLD[@]}"); fi
    eng vb_$v "${E[@]}" QWENOX_PARALLEL=8 QWENOX_VBENCH=1,2,5,8,9,16,24,40,40x8,48 -- \
      --tokens-file "$ROOT/$L/tok1024.txt" --maxctx 9216
    echo "    $v: $(errl vb_$v | sed -n 's/^vbench P=\([0-9]*\)\( segs=\([0-9]*\)\)\? median=\([0-9.]*\).*/P\1\2=\4/p' | sed 's/ segs=/x/' | tr '\n' ' ')"
  done
fi

# ---------------------------------------------------------------- dec
if step dec; then
  stamp "decode：--gen 128（AR）与 --spec-gen 256 γ4（MTP），新 vs 旧"
  for v in new old; do
    if [[ $v == new ]]; then E=(); else E=("${OLD[@]}"); fi
    eng gen_$v "${E[@]}" QWENOX_GGUF_MTP=none -- --tokens-file "$TOK2K" --gen 128 --maxctx 4096
    info "AR $v: $(errl gen_$v | grep -m1 '^decode:')"
    for T in 2048 8192; do
      eng spec${T}_$v "${E[@]}" QWENOX_GGUF_MTP="$MW" -- --tokens-file "$ROOT/data/qsa-oracle/$T.tokens" \
        --spec-gen 256 --gamma 4 --maxctx 16384
      info "MTP $T $v: $(errl spec${T}_$v | grep -m1 '^spec:')"
    done
  done
  a=$(grep '^ids:' $L/gen_new.out); b=$(grep '^ids:' $L/gen_old.out)
  [[ "$a" == "$b" ]] && info "AR greedy 128：新旧 ids 完全相同" || info "AR greedy 128：新旧 ids 不同（数值口径变了，属预期；看 PPL 门槛）"
fi

# ---------------------------------------------------------------- ppl
if step ppl; then
  stamp "PPL（2048.tokens）：新 / 旧 / 新 chunk 4"
  eng ppl_new QWENOX_GGUF_MTP=none -- --tokens-file "$TOK2K" --ppl --maxctx 4096
  eng ppl_old QWENOX_GGUF_MTP=none "${OLD[@]}" -- --tokens-file "$TOK2K" --ppl --maxctx 4096
  eng ppl_c4 QWENOX_GGUF_MTP=none QWENOX_PREFILL_CHUNK=4 -- --tokens-file "$TOK2K" --ppl --maxctx 4096
  pn=$(grep ppl_summary $L/ppl_new.out | num ppl); po=$(grep ppl_summary $L/ppl_old.out | num ppl)
  pc=$(grep ppl_summary $L/ppl_c4.out | num ppl)
  echo "    PPL new $pn | old $po | new chunk4 $pc"
  rel() { awk -v a="$1" -v b="$2" 'BEGIN { d = (a > b ? a - b : b - a) / b; printf "%.4f", d * 100; exit !(d < 0.005) }'; }
  r=$(rel "$pn" "$po") && pass "新 vs 旧 PPL 相差 ${r}%" || fail "新 vs 旧 PPL 相差 ${r}%（≥0.5%）"
  r=$(rel "$pc" "$pn") && pass "chunk4 vs chunk2048 相差 ${r}%" || fail "chunk4 vs chunk2048 相差 ${r}%（≥0.5%）"
fi

# ---------------------------------------------------------------- vseg
if step vseg; then
  stamp "vseg 自检：批 verify == 逐段 verify（逐位）"
  for spec in 3,4,5 5,5,5,5,5,5,5,5; do
    tag=vseg_${spec//,/_}
    eng $tag QWENOX_GGUF_MTP="$MW" QWENOX_PARALLEL=8 QWENOX_VSEG_SELFTEST=$spec -- --maxctx 131072
    r=$(errl $tag | grep -m1 "VSEG SELFTEST")
    [[ "$r" == *PASS* ]] && pass "$r" || { fail "vseg $spec：${r:-无结果}"; errl $tag | grep "vseg-selftest" | head -5; }
  done
fi

# ---------------------------------------------------------------- conc
if step conc; then
  stamp "conc：单路串行 vs 8 路并发（生产 256K、MTP），逐 token"
  ONLY=conc GGUF="$GGUF" MTPG="$MTPG" bash tools/iq4xs_conc_win.sh > $L/conc.txt 2>&1
  grep -E "PASS:|FAIL:|MATCH|DIFF|requests" $L/conc.txt | sed 's/^/    /'
  grep -q "IQ4XS_CONC PASS" $L/conc.txt || fail "iq4xs_conc_win conc 步骤"
  grep -q "iq1 vs iq8: MATCH" $L/conc.txt && pass "8 路并发 == 单路串行（逐 token + spec 统计）" \
    || fail "8 路并发与单路不逐位相同"
fi

echo
if [[ ${#FAILS[@]} == 0 ]]; then echo "GGUF_BATCH PASS"
else printf '失败项：\n'; printf '  - %s\n' "${FAILS[@]}"; echo "GGUF_BATCH FAIL"; fi
