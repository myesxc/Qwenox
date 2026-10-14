#!/usr/bin/env bash
# d2_s2_ids.sh — D2-S2a/S2b 验收：会合调度（VerifyPool）下并发逐 bit 一致。
# QWENOX_PARALLEL=4 起 serve（仅引擎行协议），eng_cli 直连、固定 prompt、
# greedy、max_tokens=257：
#   每个 drafter（D2_DRAFTERS，默认 "1 4"：1=MTP-only spec_loop，4=chain）：
#     base  单跑基线 ids md5
#     c2    2 并发同 prompt，两客户端 ids md5 均须 ≡ base
#     c4    4 并发同 prompt，四客户端 ids md5 均须 ≡ base
#   mix   2×drafter4 + 2×drafter3 并发（drafter=3=ngram 轮，永不入会合）：
#         各客户端 ≡ 自己 drafter 的基线（base_d3 单跑基线）
# 硬门槛：drafter=1 与 drafter=4 的并发相位均须新增多段 [vr] batch
# （防假 PASS；chain 相位若退回 solo/串行说明会合接入失效）。
# 可选：D2_GAMMA=N 强制 γ=N。S3a 起行预算 48：PARALLEL=4 下 active=4 的
#       clamp 阈值为 48/4-1=11 > γ 上限 8，clamp/丢弃路径不再由本脚本触发
#       （executor 硬预算仍兜底；S2b 已验 fallbacks 路径逐位正确）。
#       注意 γ≤7：γ=8 输出偏离 γ≤7 基线（同配置 NO_VR 单跑亦偏离），属既有
#       引擎行为、与会合无关，超出本测试范围，故 γ>7 直接拒绝。
#       D2_NOVR=1 以 QWENOX_NO_VR=1 起引擎（分时旧路径对照，须 ≡ 基线且
#       不得出现任何 [vr] batch）；
#       QWENOX_VR_WAIT_MS=1 穿透给引擎（强制超时 executor 兜底路径）。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BIN=build/qwenox-engine-win.exe
CLI=build/eng_cli.exe
M=./models/qwen38-flash-next-v2.hgn
NG=./models/qwen38-flash-next-ngram.hgn
MTP=./models/qwen38-flash-next-mtp.hgn
TOKDIR=./models/tokenizer
EPORT=18730

export QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1
export QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1
export QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 QWENOX_NOWARMUP=1
export QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1
export QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1
export QWENOX_KVSNAP=0 QWENOX_RCKPT_MAX=0 QWENOX_KV_PAGED=1 QWENOX_PARALLEL=4
export QWENOX_ROPE_FACTOR=1 QWENOX_ROPE_ORIGINAL_CTX=262144
export QWENOX_ROPE_BETA_FAST=32 QWENOX_ROPE_BETA_SLOW=1 QWENOX_ROPE_ATTN_SCALE=0
export QWENOX_VR_DBG=1
if [[ "${D2_GAMMA:-0}" -gt 7 ]]; then
  echo "FAIL: D2_GAMMA>7 超出本测试范围（γ=8 既有引擎行为偏离基线，与会合无关）"; exit 1
fi
if [[ "${D2_GAMMA:-0}" -gt 0 ]]; then export QWENOX_SPEC_GAMMA="$D2_GAMMA"; fi
if [[ "${D2_NOVR:-0}" = 1 ]]; then export QWENOX_NO_VR=1; fi
# D2_SAMPLE=<temperature>：采样请求（固定 seed）。单跑/并发须逐 token 一致：会合
# verify 的 logits 行与串行逐 bit 相同、每请求 RNG 流独立，所以采样结果也应相同。
# 默认只测 drafter=4（chain，API 默认），跳过混合相位（纯 ngram 只支持 greedy）。
SMP=""
if [[ -n "${D2_SAMPLE:-}" ]]; then
  SMP=",\"temperature\":${D2_SAMPLE},\"seed\":4242"
  D2_DRAFTERS="${D2_DRAFTERS:-4}"
fi

# eng_cli：纯主机 C++，缺了就用 TheRock 自带 clang++ 现编（同 build.sh 的源列表）。
if [[ ! -x "$CLI" ]]; then
  TR="${THEROCK:-/c/therock-dist-windows-gfx1151-10.1.0}"
  CXX="$TR/lib/llvm/bin/clang++.exe"
  [[ -x "$CXX" ]] || { echo "FAIL: 缺 $CLI 且找不到 TheRock clang++: $CXX"; exit 1; }
  echo "[编译] $CLI"
  "$CXX" -O2 -std=c++17 -D_CRT_SECURE_NO_WARNINGS -Isrc/api -Ithird_party \
    src/api/engine_client.cpp src/api/tokenizer.cpp src/api/eng_cli.cpp \
    -lws2_32 -o "$CLI" || { echo "FAIL: eng_cli 编译失败"; exit 1; }
fi

ELOG=logs/d2s2_engine.log
epid=''
cleanup() {
  trap - EXIT INT TERM
  if [[ -n "$epid" ]]; then
    kill "$epid" 2>/dev/null || true
    taskkill //PID "$epid" //F //T >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

if netstat -an | grep -E "[:.]$EPORT\s+.*LISTENING" >/dev/null 2>&1; then
  echo "FAIL: 端口 $EPORT 已被占用"; exit 1
fi

echo "== d2_s2_ids $(date '+%F %T')  PARALLEL=4 NO_VR=${D2_NOVR:-0} GAMMA=${D2_GAMMA:-auto} DRAFTERS=${D2_DRAFTERS:-1 4} SAMPLE=${D2_SAMPLE:-off}"
"$BIN" "$M" "$NG" "$MTP" --serve --host 127.0.0.1 --port "$EPORT" \
    --maxctx 16384 >"$ELOG" 2>&1 &
epid=$!
begin=$SECONDS
until grep -q 'serve: listening' "$ELOG" 2>/dev/null; do
  kill -0 "$epid" 2>/dev/null || { tail -n 15 "$ELOG"; echo "FAIL: 引擎提前退出"; exit 1; }
  (( SECONDS - begin < 1800 )) || { echo "FAIL: 引擎启动超时"; exit 1; }
  sleep 2
done
echo "  .. engine ready ($((SECONDS - begin))s)"

PROMPT='Write a complete, production-quality quicksort implementation in Python. Requirements: include detailed comments explaining each step, handle edge cases (empty input, single element, duplicates), and include a small test harness (main function or equivalent) that sorts several example arrays and prints the results. Output only the code, no explanations.'

run_client() {  # $1=标签 $2=drafter
  "$CLI" --host "127.0.0.1:$EPORT" --tokenizer "$TOKDIR" \
      "{\"prompt\":\"$PROMPT\",\"max_tokens\":257,\"drafter\":$2$SMP}" \
      > "logs/d2s2_$1.json" 2> "logs/d2s2_$1.err"
}
ids_md5() { grep -o '"ids":\[[0-9,]*\]' "logs/d2s2_$1.json" | md5sum | cut -d' ' -f1; }

ok=1
check() {  # $1=标签 $2=期望 md5
  local h
  h=$(ids_md5 "$1")
  if [[ -n "$h" && "$h" == "$2" ]]; then
    printf '  %-10s md5=%s IDENTICAL\n' "$1" "$h"
  else
    printf '  %-10s md5=%s DIFF\n' "$1" "$h"; ok=0
  fi
}

multi_batches() {  # 引擎日志里多段 [vr] batch 的累计行数
  grep -a '\[vr\] batch' "$ELOG" 2>/dev/null | grep -avc 'batch: 1 segs' || true
}

for dr in ${D2_DRAFTERS:-1 4}; do
  echo "-- drafter=$dr"
  nm0=$(multi_batches)
  run_client "base_d$dr" "$dr" || { echo "FAIL: base_d$dr rc=$?"; tail -3 "logs/d2s2_base_d$dr.err"; exit 1; }
  b=$(ids_md5 "base_d$dr")
  [[ -n "$b" ]] || { echo "FAIL: base_d$dr 无 ids"; cat "logs/d2s2_base_d$dr.json"; exit 1; }
  echo "  base_d$dr   md5=$b"
  for conc in 2 4; do
    pids=()
    for ((i = 0; i < conc; i++)); do run_client "d${dr}c${conc}_$i" "$dr" & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p"; done
    for ((i = 0; i < conc; i++)); do check "d${dr}c${conc}_$i" "$b"; done
  done
  # 硬门槛（drafter 1 与 4 都适用）：本 drafter 的并发相位须形成新的多段
  # batch——chain 若退回 solo/串行说明会合接入失效（S2c 修复前即为 0）。
  if [[ "${D2_NOVR:-0}" != 1 && "$dr" != 3 ]]; then
    nm1=$(multi_batches)
    echo "  .. drafter=$dr 相位新增多段 batch: $((nm1 - nm0))"
    if [[ $((nm1 - nm0)) -eq 0 ]]; then
      echo "FAIL: drafter=$dr 并发相位未形成多段 batch（会合未生效？）"; ok=0
    fi
  fi
done

# 混合相位：2×chain(drafter=4) + 2×ngram(drafter=3) 并发。ngram 客户端永不
# 入会合（不入 active、不提交），chain 客户端照常会合；各自 ≡ 各自基线。
if [[ " ${D2_DRAFTERS:-1 4} " == *" 4 "* && -z "${D2_SAMPLE:-}" ]]; then
  echo "-- mix: 2x drafter=4 + 2x drafter=3"
  run_client base_d3 3 || { echo "FAIL: base_d3 rc=$?"; exit 1; }
  b3=$(ids_md5 base_d3)
  b4=$(ids_md5 base_d4)
  [[ -n "$b3" && -n "$b4" ]] || { echo "FAIL: 混合基线缺失"; exit 1; }
  echo "  base_d3   md5=$b3"
  pids=()
  for i in 0 1; do run_client "mix_c$i" 4 & pids+=($!); done
  for i in 2 3; do run_client "mix_c$i" 3 & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p"; done
  check mix_c0 "$b4"; check mix_c1 "$b4"
  check mix_c2 "$b3"; check mix_c3 "$b3"
fi

nb=$(grep -c '\[vr\] batch' "$ELOG" 2>/dev/null || true)
nm=$(multi_batches)
echo "  .. [vr] batch 行: $nb（其中多段: $nm）"
if [[ "${D2_NOVR:-0}" = 1 && "$nb" -ne 0 ]]; then
  echo "FAIL: NO_VR=1 下不应出现 [vr] batch"; ok=0
fi
echo "S2 IDS ACCEPT: $([ $ok = 1 ] && echo PASS || echo FAIL)"
[[ $ok = 1 ]]
