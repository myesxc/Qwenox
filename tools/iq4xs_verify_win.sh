#!/usr/bin/env bash
# iq4xs_verify_win.sh — Windows 一键验证：Unsloth UD-IQ4_XS GGUF（IQ3_S / IQ4_XS /
# IQ4_NL / Q8_0 专家 + Q8_0 dense + Q6_K lm_head）纯 GGUF 启动。
#
#   bash tools/iq4xs_verify_win.sh                 # 全部（约 30 分钟）
#   GGUF=/path/x-00001-of-00003.gguf bash tools/iq4xs_verify_win.sh
#   SKIP_BUILD=1 SKIP_BENCH=1 bash tools/iq4xs_verify_win.sh   # 只做正确性（约 12 分钟）
#   ONLY="svc bench" SKIP_BUILD=1 bash tools/iq4xs_verify_win.sh # 只跑指定步骤：build kt v2 gguf mtp svc bench
#   MTPG=/path/mtp-*.gguf …   # MTP sidecar（默认 GGUF 同目录的 mtp-Qwen3.8-Flash-Next-Q8_0.gguf；
#                             # 不存在则跳过 mtp 步骤，服务不带投机）
#
# 门槛（任何一项不过 => 末尾 FAIL）：
#   1 编译：engine / bench / api / launcher / settings 全部产物
#   2 kernel 对拍（真实权重，CPU 参考）：混合类型层 0（IQ3_S/IQ4_NL）、2（IQ4_XS/Q8_0）、
#     30/47（IQ3_S/Q8_0）；WMMA rel-L2 < 1e-2，GEMV < 1e-4，dedup 与普通路径逐 bit
#   3 v2 hgn 不受影响：新二进制 vs 基线 BASE_REV（默认 6eb7c9e）PPL 与 greedy ids 逐字相同
#   4 GGUF 一致性：prefill（chunk 2048，WMMA）与小 chunk（4，GEMV+dedup）PPL 相差 < 1%
#   5 GGUF 质量：同 tokens 上 PPL 低于 v2（BF16 KLD 64×512：UD-IQ4_XS .100 vs v2 .107）
#   M MTP（有 sidecar 时）：GGUF+sidecar 与 v2+mtp.hgn 同 prompt greedy γ=4 --spec-gen 256，
#     草稿头已加载，commit/round ≥ 0.9×v2（只报告：tok/s、与普通 greedy 的前缀一致长度）
#   6 服务端到端：临时 conf（weights=gguf，测试端口 18730/18731）起 qwenox-win，
#     /admin/engine/start 拉起 GGUF 引擎，/v1/chat/completions 返回非空回复
# 只报告不设门槛：decode tok/s（--gen 128，无 MTP）、prefill tok/s（qwenox-bench，≤128K）。
# 日志 logs/iq4xs_*.log；结尾 IQ4XS_VERIFY PASS / FAIL。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$(pwd)"
GGUF="${GGUF:-/c/Users/Mark/Models/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}"
MTPG="${MTPG:-$(dirname "$GGUF")/mtp-Qwen3.8-Flash-Next-Q8_0.gguf}"
V2MTP=./models/qwen38-flash-next-mtp.hgn
V2=./models/qwen38-flash-next-v2.hgn
NG=./models/qwen38-flash-next-ngram.hgn
TOK=data/qsa-oracle/2048.tokens
BASE_REV="${BASE_REV:-6eb7c9e}"
BASE_WT="${BASE_WT:-$HOME/aggrprobe/wt_base_$BASE_REV}"
TR="${THEROCK:-/c/therock-dist-windows-gfx1151-10.1.0}"
L=logs; mkdir -p $L
FAILS=()
step() { [[ -z "${ONLY:-}" || " $ONLY " == *" $1 "* ]]; }
fail() { FAILS+=("$1"); echo "  FAIL: $1"; }
pass() { echo "  PASS: $1"; }
info() { echo "  INFO: $1"; }
stamp() { echo; echo "=== [$(date +%H:%M:%S)] $*"; }
num() { sed -n "s#.*$1=\([0-9.e+-]*\).*#\1#p" | head -1; }

[[ -f "$GGUF" ]] || { echo "缺 GGUF 首分片：$GGUF"; echo "IQ4XS_VERIFY FAIL"; exit 1; }
if tasklist 2>/dev/null | grep -qiE "qwenox-(engine-)?win"; then
  echo "检测到 qwenox 服务/引擎在运行，请先停掉再跑（显存要整块用）"; echo "IQ4XS_VERIFY FAIL"; exit 1
fi
echo "head $(git log --oneline -1 | cut -c1-70)  dirty files: $(git status --short | wc -l)"
echo "GGUF $GGUF"

# 生产内核组合（同 start_win.sh），单路、无 kvsnap
export QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1
export QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1
export QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 QWENOX_NOWARMUP=1
export QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1
export QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1
export QWENOX_KVSNAP=0 QWENOX_RCKPT_MAX=0 QWENOX_KV_PAGED=1 QWENOX_PARALLEL=1
unset QWENOX_GGUF QWENOX_GGUF_DENSE QWENOX_GGUF_PLE QWENOX_GGUF_EXPERTS QWENOX_PREFILL_CHUNK
export QWENOX_GGUF_MTP=none   # 不投机（MTP sidecar 未就位时也能跑）

# ---------------------------------------------------------------- 1 编译
stamp "1 编译"
if [[ "${SKIP_BUILD:-0}" != 1 ]] && step build; then
  if bash build_win.sh > $L/iq4xs_build.log 2>&1; then pass "build_win.sh（全部产物）"
  else fail "build_win.sh"; grep -m15 -E "error|错误" $L/iq4xs_build.log; fi
fi
export HIP_PATH="$(cygpath -w "$TR")"
FL=(-O3 -std=c++17 --offload-arch=gfx1151 -D_CRT_SECURE_NO_WARNINGS -D_CRT_NONSTDC_NO_DEPRECATE -I"$TR/include")
mkdir -p build/iqt
step kt && for t in moe_gguf_test moe_gguf_gemv_test; do
  "$TR/bin/hipcc.exe" "${FL[@]}" tools/$t.cu -o build/iqt/$t.exe > $L/iq4xs_$t.build.log 2>&1 \
    || fail "编译 tools/$t.cu"
done
if step v2 && [[ ! -x "$BASE_WT/build/qwenox-engine-win.exe" ]]; then
  echo "  基线 $BASE_REV -> $BASE_WT（git worktree + build_win.sh engine）"
  [[ -d "$BASE_WT" ]] || git worktree add --detach "$BASE_WT" "$BASE_REV" > $L/iq4xs_base_wt.log 2>&1
  (cd "$BASE_WT" && bash build_win.sh engine > "$ROOT/$L/iq4xs_base_build.log" 2>&1) \
    || fail "基线 $BASE_REV 编译"
fi

# ---------------------------------------------------------------- 2 kernel 对拍
step kt && stamp "2 kernel 对拍（真实权重 vs CPU 参考）"
step kt && for LY in 0 2 30 47; do
  out=$(cd build && ./iqt/moe_gguf_test.exe "$GGUF" --layer $LY --P 2048 --iters 3 2>&1)
  echo "$out" > $L/iq4xs_wmma_l$LY.log
  echo "  L$LY $(echo "$out" | grep -m1 '^layer' | cut -c1-90)"
  echo "$out" | grep -q "RESULT PASS" && pass "WMMA layer $LY ($(echo "$out" | grep -o 'up rel-L2 max [0-9.e+-]*'))" \
    || fail "WMMA layer $LY"
  for P in 1 4 8; do
    out=$(cd build && ./iqt/moe_gguf_gemv_test.exe "$GGUF" --layer $LY --P $P --iters 10 --share 0.3 2>&1)
    echo "$out" > $L/iq4xs_gemv_l${LY}_p$P.log
    echo "$out" | tail -1 | grep -q "^PASS" && pass "GEMV layer $LY P=$P ($(echo "$out" | grep -o 'time [0-9.]* us/call' | head -1))" \
      || fail "GEMV layer $LY P=$P"
  done
done

# ---------------------------------------------------------------- 引擎运行器
# eng LABEL BIN [ENV=V ...] -- ARGS...   -> $L/iq4xs_LABEL.{out,err}，返回码
eng() {
  local lab=$1 bin=$2; shift 2
  local E=()
  while [[ $# -gt 0 && $1 != -- ]]; do E+=("$1"); shift; done
  shift
  local t0=$(date +%s)
  (cd "$(dirname "$bin")" && env "${E[@]}" "./$(basename "$bin")" "$@" \
     > "$ROOT/$L/iq4xs_$lab.out" 2> "$ROOT/$L/iq4xs_$lab.err")
  local rc=$?
  echo "  [$lab] rc=$rc $(( $(date +%s) - t0 ))s"
  [[ $rc == 0 ]] || tr '\r' '\n' < "$ROOT/$L/iq4xs_$lab.err" | grep -vE "pload [0-9]" | tail -5
  return $rc
}
ppl_of() { grep ppl_summary "$L/iq4xs_$1.out" | num ppl; }
ids_of() { grep '^ids:' "$L/iq4xs_$1.out"; }
tps_of() { tr '\r' '\n' < "$L/iq4xs_$1.err" | sed -n 's/^decode: .* = \([0-9.]*\) tok\/s.*/\1/p' | tail -1; }
NEW=$ROOT/build/qwenox-engine-win.exe
BASE=$BASE_WT/build/qwenox-engine-win.exe
V2A=("$ROOT/$V2" "$ROOT/$NG")
TOKA=$ROOT/$TOK

# ---------------------------------------------------------------- 3 v2 不受影响
if step v2; then
stamp "3 v2 hgn：新二进制 vs 基线 $BASE_REV（PPL + greedy ids 逐字）"
eng v2_new_ppl "$NEW" -- "${V2A[@]}" --tokens-file "$TOKA" --ppl --maxctx 4096
eng v2_base_ppl "$BASE" -- "${V2A[@]}" --tokens-file "$TOKA" --ppl --maxctx 4096
a=$(grep ppl_summary $L/iq4xs_v2_new_ppl.out); b=$(grep ppl_summary $L/iq4xs_v2_base_ppl.out)
echo "  new : $a"; echo "  base: $b"
[[ -n "$a" && "$a" == "$b" ]] && pass "v2 PPL 逐字相同" || fail "v2 PPL 与基线不同"
eng v2_new_gen "$NEW" -- "${V2A[@]}" --tokens-file "$TOKA" --gen 128 --maxctx 4096
eng v2_base_gen "$BASE" -- "${V2A[@]}" --tokens-file "$TOKA" --gen 128 --maxctx 4096
[[ -n "$(ids_of v2_new_gen)" && "$(ids_of v2_new_gen)" == "$(ids_of v2_base_gen)" ]] \
  && pass "v2 greedy 128 ids 逐字相同" || fail "v2 greedy ids 与基线不同"
fi

# ---------------------------------------------------------------- 4/5 GGUF
if step gguf; then
stamp "4 GGUF：prefill（WMMA）vs 小 chunk（GEMV+dedup）PPL；5 质量 vs v2"
eng gguf_ppl "$NEW" -- "$GGUF" --tokens-file "$TOKA" --ppl --maxctx 4096 \
  || fail "GGUF 启动/PPL（见 $L/iq4xs_gguf_ppl.err）"
eng gguf_ppl_c4 "$NEW" QWENOX_PREFILL_CHUNK=4 -- "$GGUF" --tokens-file "$TOKA" --ppl --maxctx 4096 \
  || fail "GGUF 小 chunk PPL"
tr '\r' '\n' < $L/iq4xs_gguf_ppl.err | grep -E "^gguf|requant|devarena:|weight arena|^ple:" | sed 's/^/    /'
pg=$(ppl_of gguf_ppl); pc=$(ppl_of gguf_ppl_c4); pv=$(ppl_of v2_new_ppl)
echo "  PPL  GGUF chunk2048 $pg | GGUF chunk4 $pc | v2 $pv"
if [[ -n "$pg" && -n "$pc" ]]; then
  awk -v a="$pg" -v b="$pc" 'BEGIN { d = (a > b ? a - b : b - a) / a; printf "  rel diff %.4f%%\n", d * 100; exit !(d < 0.01) }' \
    && pass "GGUF prefill/GEMV PPL 一致（<1%）" || fail "GGUF prefill 与 GEMV 路径 PPL 相差 ≥1%"
fi
[[ -z "$pv" ]] && { eng v2_new_ppl "$NEW" -- "${V2A[@]}" --tokens-file "$TOKA" --ppl --maxctx 4096; pv=$(ppl_of v2_new_ppl); }
[[ -n "$pg" && -n "$pv" ]] && awk -v g="$pg" -v v="$pv" 'BEGIN { exit !(g < v) }' \
  && pass "GGUF PPL $pg < v2 $pv" || fail "GGUF PPL 不低于 v2"

stamp "decode tok/s（--gen 128，2048 prompt，无 MTP）"
eng gguf_gen "$NEW" -- "$GGUF" --tokens-file "$TOKA" --gen 128 --maxctx 4096 || fail "GGUF --gen"
nu=$(ids_of gguf_gen | tr ' ' '\n' | tail -n +2 | sort -u | wc -l)
[[ $nu -ge 8 ]] && pass "GGUF greedy 输出非退化（$nu 个不同 token）" || fail "GGUF greedy 输出退化（$nu 个不同 token）"
[[ -f $L/iq4xs_v2_new_gen.err ]] || eng v2_new_gen "$NEW" -- "${V2A[@]}" --tokens-file "$TOKA" --gen 128 --maxctx 4096
info "decode（均无 MTP）GGUF $(tps_of gguf_gen) tok/s | v2 $(tps_of v2_new_gen) tok/s"
fi

# ---------------------------------------------------------------- M MTP
spec_of() { tr '\r' '\n' < "$L/iq4xs_$1.err" | grep -m1 '^spec:'; }
if step mtp; then
if [[ ! -f "$MTPG" ]]; then
  info "无 MTP sidecar（$MTPG），跳过 MTP 步骤"
else
  stamp "M MTP：GGUF + $(basename "$MTPG") vs v2 + mtp.hgn（greedy γ=4，--spec-gen 256）"
  MTPW=$(cygpath -m "$MTPG")
  for T in 2048 8192; do
    TF=$ROOT/data/qsa-oracle/$T.tokens
    eng gguf_spec$T "$NEW" QWENOX_GGUF_MTP="$MTPW" -- "$GGUF" --tokens-file "$TF" --spec-gen 256 --gamma 4 --maxctx 16384 \
      || fail "GGUF MTP spec-gen $T"
    eng v2_spec$T "$NEW" -- "${V2A[@]}" "$ROOT/$V2MTP" --tokens-file "$TF" --spec-gen 256 --gamma 4 --maxctx 16384 \
      || fail "v2 MTP spec-gen $T"
    tr '\r' '\n' < $L/iq4xs_gguf_spec$T.err | grep -qE "^mtp: draft head enabled" \
      || fail "GGUF $T：MTP 草稿头未启用（见 $L/iq4xs_gguf_spec$T.err）"
    sg=$(spec_of gguf_spec$T); sv=$(spec_of v2_spec$T)
    echo "  [$T] GGUF: ${sg:-（无 spec 行）}"
    echo "  [$T] v2  : ${sv:-（无 spec 行）}"
    cg=$(echo "$sg" | num commit/round); cv=$(echo "$sv" | num commit/round)
    [[ -n "$cg" && -n "$cv" ]] && awk -v g="$cg" -v v="$cv" 'BEGIN { exit !(g >= 0.9 * v) }' \
      && pass "[$T] GGUF commit/round $cg ≥ 0.9×v2 $cv" || fail "[$T] GGUF commit/round ${cg:-?} < 0.9×v2 ${cv:-?}"
  done
  # 与普通 greedy 的一致性（只报告）：同 2048 prompt --gen 256
  eng gguf_gen256 "$NEW" -- "$GGUF" --tokens-file "$TOKA" --gen 256 --maxctx 16384
  a=($(ids_of gguf_gen256)); b=($(grep '^ids:' $L/iq4xs_gguf_spec2048.out))
  n=0; while [[ $n -lt ${#a[@]} && $n -lt ${#b[@]} && "${a[$n]}" == "${b[$n]}" ]]; do n=$((n+1)); done
  plen=$(wc -w < "$TOKA")
  info "GGUF spec 与普通 greedy：生成部分前 $(( n - plen - 1 )) 个 token 一致（共比 $(( ${#a[@]} - plen - 1 )) 个）"
  info "GGUF 普通 decode $(tps_of gguf_gen256) tok/s（--gen 256）"
fi
fi

# ---------------------------------------------------------------- 6 服务端到端
GWIN=$(cygpath -m "$GGUF")
if step svc; then
stamp "6 服务：qwenox-win + weights=gguf 临时 conf → /admin/engine/start → chat"
CONF=$ROOT/service.iq4xs-test.conf
cat > "$CONF" <<EOF
# iq4xs_verify_win.sh 临时配置（跑完删除）
MODEL_DIR="./models"
TOKENIZER_DIR="./models/tokenizer"
GGUF_FILE="$GWIN"
GGUF_MTP_FILE="$([[ -f "$MTPG" ]] && cygpath -m "$MTPG")"
GGUF_VISION_FILE=""
ENGINE_HOST="127.0.0.1"
ENGINE_PORT="18730"
API_HOST="127.0.0.1"
API_PORT="18731"
MAX_CONTEXT="32768"
PARALLEL="1"
KV_PAGED="1"
KVSNAP_MAX_GB="0"
MTP_GAMMA="0"
# start_win: weights=gguf
EOF
# 从项目根启动：--service-conf 指定时 qwenox-win 不 chdir，相对路径（tokenizer）按 CWD
./build/qwenox-win.exe --console --service-conf "$(cygpath -w "$CONF")" > "$ROOT/$L/iq4xs_api.log" 2>&1 &
# 只杀本测试拉起的进程（命令行含临时 conf 名 / 测试端口 18730），不碰正式服务
kill_test() {
  powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { (\$_.Name -like 'qwenox*') -and (\$_.CommandLine -match 'iq4xs-test|18730') } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" > /dev/null 2>&1
}
API=http://127.0.0.1:18731
ok_api=0
for i in $(seq 60); do curl -s -m 2 $API/admin/engine/status > /dev/null && { ok_api=1; break; }; sleep 1; done
if [[ $ok_api == 1 ]]; then
  curl -s -m 10 -X POST $API/admin/engine/start > $L/iq4xs_svc_start.json; echo "  start: $(cut -c1-160 $L/iq4xs_svc_start.json)"
  st=""
  for i in $(seq 120); do
    st=$(curl -s -m 3 $API/admin/engine/status | sed -n 's/.*"state":"\([a-z]*\)".*/\1/p')
    [[ $st == running || $st == stopped || $st == exited || $st == failed ]] && break
    sleep 3
  done
  echo "  engine state: $st"
  elog=$(ls -t logs/engine-api-*.log 2>/dev/null | head -1)
  [[ -n "$elog" ]] && tr '\r' '\n' < "$elog" | grep -E "^gguf: pure|^mtp: " | sed 's/^/    /' 
  if [[ $st == running ]]; then
    # 请求体先落成 UTF-8 文件再 --data-binary：中文直接放进 curl 命令行参数会被
    # Windows 按 ANSI 代码页（GBK）转码，API 收到的就不是合法 UTF-8（JSON decode error）
    printf '%s' '{"model":"qwenox","messages":[{"role":"user","content":"用一句话解释为什么天空是蓝色的。"}],"max_tokens":96,"temperature":0}' \
      > $L/iq4xs_svc_req.json
    curl -s -m 180 $API/v1/chat/completions -H 'Content-Type: application/json; charset=utf-8' \
      --data-binary @$L/iq4xs_svc_req.json > $L/iq4xs_svc_chat.json
    reply=$(PYTHONIOENCODING=utf-8 python -c "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8')); print(d['choices'][0]['message']['content'])" \
      $L/iq4xs_svc_chat.json 2>/dev/null)
    [[ -n "$reply" ]] || echo "  response: $(cut -c1-300 $L/iq4xs_svc_chat.json)"
    echo "  reply: ${reply:0:200}"
    [[ ${#reply} -ge 4 ]] && pass "服务端到端（weights=gguf）对话有回复" || fail "服务对话无回复（见 $L/iq4xs_svc_chat.json）"
  else
    fail "服务未能拉起 GGUF 引擎（state=$st，见 logs/engine-api-*.log）"
  fi
  curl -s -m 30 -X POST $API/admin/engine/stop > /dev/null
  sleep 5
else
  fail "qwenox-win 未在 18731 就绪（见 $L/iq4xs_api.log）"
fi
kill_test; sleep 2; kill_test
rm -f "$CONF" "$CONF.bak"
fi

# ---------------------------------------------------------------- prefill bench
if [[ "${SKIP_BENCH:-0}" != 1 ]] && step bench; then
  # 只取 prefill：qwenox-bench 没有 MTP 时跳过 TG 并返回 1（设计如此，GGUF 这里不带
  # sidecar）；hgn 的 MTP 会从 service.conf 读到（Windows 上空环境变量等于未设），
  # 所以 TG 两边不可比，decode 对比看上面 --gen 那组。门槛 = prefill 汇总行出现。
  stamp "prefill tok/s（qwenox-bench，MAX_CONTEXT=131072，1 遍）"
  for fmt in gguf hgn; do
    (cd build && env QWENOX_BENCH_FORMAT=$fmt GGUF_FILE="$GWIN" GGUF_MTP_FILE= \
      MAX_CONTEXT=131072 BENCH_PREFILL_REPEATS=1 PARALLEL=1 KVSNAP_MAX_GB=0 \
      ./qwenox-bench.exe run > "$ROOT/$L/iq4xs_bench_$fmt.log" 2>&1)
    grep -q "^Prefill average" "$L/iq4xs_bench_$fmt.log" || fail "qwenox-bench $fmt prefill 未完成（见 $L/iq4xs_bench_$fmt.log）"
    echo "  [$fmt]"; grep -E "^ +[0-9]+ +[0-9]+ tokens|^Prefill average" "$L/iq4xs_bench_$fmt.log" | sed 's/^/   /'
  done
fi

echo
if [[ ${#FAILS[@]} == 0 ]]; then echo "IQ4XS_VERIFY PASS"
else printf '失败项：\n'; printf '  - %s\n' "${FAILS[@]}"; echo "IQ4XS_VERIFY FAIL"; fi
