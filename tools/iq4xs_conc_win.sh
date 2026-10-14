#!/usr/bin/env bash
# iq4xs_conc_win.sh — Windows：UD-IQ4_XS GGUF 的并发、显存、prefill 分段一键检查（约 12 分钟）。
#
#   bash tools/iq4xs_conc_win.sh
#   GGUF=/c/.../x-00001-of-00003.gguf MTPG=/c/.../mtp-*.gguf bash tools/iq4xs_conc_win.sh
#   ONLY="conc mem pp" …   # 只跑其中几步
#
# 1 conc：tools/conc_verify.py（端口 8732）
#   单路（PARALLEL=1）串行跑 8 组 flow → 生产配置（256K、PARALLEL=8、MTP、视觉塔）8 路并发跑同一组
#   → ctl（INFO 槽数、负载下 PING、取消、断连）。门槛：所有请求正常结束 + ctl PASS。
#   并发 vs 单路逐 token 对比只报告（GGUF dense / 多行专家 kernel 不是行无关的，批大小不同
#   舍入就不同，可能在某处分叉）：打印每个 flow 的分叉位置。
# 2 mem：生产配置（kvsnap 开）chunk 8192 / 16384 各起一次，读 devarena 与 memory[ready]。
#   门槛：两种都能起到 listening。
# 3 pp：32K prompt 单请求，QWENOX_PREFILL_CHUNK=4096 / 8192 / 16384（只报告 tok/s）。
# 结尾 IQ4XS_CONC PASS / FAIL；日志 logs/iq4xs_conc_*.{log,txt}。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$(pwd)"
GGUF="${GGUF:-/c/Users/Mark/Models/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}"
MTPG="${MTPG:-$(dirname "$GGUF")/mtp-Qwen3.8-Flash-Next-Q8_0.gguf}"
VIS="${VIS:-$ROOT/models/qwen38-flash-next-vision.hgn}"
BIN=$ROOT/build/qwenox-engine-win.exe
L=logs; mkdir -p $L logs/conc
FAILS=()
fail() { FAILS+=("$1"); echo "  FAIL: $1"; }
pass() { echo "  PASS: $1"; }
info() { echo "  INFO: $1"; }
stamp() { echo; echo "=== [$(date +%H:%M:%S)] $*"; }
step() { [[ -z "${ONLY:-}" || " $ONLY " == *" $1 "* ]]; }

for f in "$GGUF" "$MTPG" "$VIS" "$BIN"; do
  [[ -f "$f" ]] || { echo "缺文件：$f"; echo "IQ4XS_CONC FAIL"; exit 1; }
done
if tasklist 2>/dev/null | grep -qi "qwenox"; then
  echo "检测到 qwenox 进程在运行，请先停掉服务（显存要整块用）"; echo "IQ4XS_CONC FAIL"; exit 1
fi
echo "head $(git log --oneline -1 | cut -c1-70)"
GW=$(cygpath -m "$GGUF"); MW=$(cygpath -m "$MTPG")
# Windows 上 Python 默认按 ANSI 代码页（GBK）读写文本：工具读 UTF-8 的 json/tokenizer 会报错
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8

prod_env() {  # start_win.sh 同款生产内核组合
  for e in $(compgen -e | grep '^QWENOX_'); do unset "$e"; done
  export QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1
  export QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1
  export QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 QWENOX_NOWARMUP=1
  export QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1
  export QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1
  export QWENOX_KV_PAGED=1 QWENOX_ROPE_FACTOR=1 QWENOX_ROPE_ORIGINAL_CTX=262144
  export QWENOX_ROPE_BETA_FAST=32 QWENOX_ROPE_BETA_SLOW=1 QWENOX_ROPE_ATTN_SCALE=0
  export QWENOX_GGUF_MTP="$MW"
}

ENG_PID=""
# serve TAG MAXCTX [ENV=V ...] : 起引擎到 listening（最多 10 分钟），日志 $L/iq4xs_conc_TAG.log
serve() {
  local tag=$1 mc=$2; shift 2
  for e in "$@"; do export "$e"; done
  # EXTRA_ENV="QWENOX_X=1 QWENOX_Y=2" (debug): extra engine env on every engine
  # this script starts (prod_env clears QWENOX_* first, so plain exports are lost)
  for e in ${EXTRA_ENV:-}; do export "$e"; done
  local log=$ROOT/$L/iq4xs_conc_$tag.log
  # MODEL_ARGS="<hgn args>" (debug): serve another model with the same harness
  local margs=("$GW")
  [[ -n "${MODEL_ARGS:-}" ]] && read -r -a margs <<< "$MODEL_ARGS"
  (cd build && exec ./qwenox-engine-win.exe "${margs[@]}" --serve --host 127.0.0.1 --port 8732 \
      --maxctx "$mc" --vision-tower "$(cygpath -m "$VIS")" > "$log" 2>&1) &
  ENG_PID=$!
  local t0=$(date +%s)
  while :; do
    if tr '\r' '\n' < "$log" | grep -q "serve: listening"; then
      echo "  [$tag] listening after $(( $(date +%s) - t0 ))s"; return 0
    fi
    if ! kill -0 $ENG_PID 2>/dev/null; then
      echo "  [$tag] engine exited before listening"; tr '\r' '\n' < "$log" | grep -v "pload [0-9]" | tail -8; return 1
    fi
    (( $(date +%s) - t0 > 600 )) && { echo "  [$tag] timeout"; return 1; }
    sleep 2
  done
}
stop_eng() {
  [[ -n "$ENG_PID" ]] && kill $ENG_PID 2>/dev/null
  powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { (\$_.Name -like 'qwenox-engine*') -and (\$_.CommandLine -match '8732') } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" > /dev/null 2>&1
  sleep 4; ENG_PID=""
}
trap stop_eng EXIT
memline() {  # 显存相关日志行
  tr '\r' '\n' < "$ROOT/$L/iq4xs_conc_$1.log" | grep -E "^devarena:|^weight arena: [0-9]|^memory\[ready\]|^\[kvpage\]|^mtp: draft|arena: estimate" | sed 's/^/    /'
}

# ---------------------------------------------------------------- 1 并发
if step conc; then
  stamp "1 并发：单路串行 → 8 路并发（同一组 flow）→ 控制面"
  rm -f logs/conc/iq1.json logs/conc/iq8.json
  prod_env
  # SEQ_PAR=8: run the serial reference on an 8-slot engine too (separates
  # concurrency effects from PARALLEL-dependent engine configuration)
  SP=${SEQ_PAR:-1}
  # SEQ_MAXCTX=262144: same --maxctx as the concurrent engine (debug)
  if serve single ${SEQ_MAXCTX:-32768} QWENOX_PARALLEL=$SP QWENOX_PREFILL_CHUNK=8192 QWENOX_CONC_PREFILL_CHUNK=8192 QWENOX_KVSNAP=0 QWENOX_NGRAM_CHUNK=16; then
    python tools/conc_verify.py seq --tag iq1 > $L/iq4xs_conc_seq.txt 2>&1 \
      && pass "单路串行 8 组 flow 正常结束" || { fail "单路串行"; tail -5 $L/iq4xs_conc_seq.txt; }
    grep -E "requests|FAIL" $L/iq4xs_conc_seq.txt | sed 's/^/    /'
  else fail "单路引擎没起来"; fi
  stop_eng
  prod_env
  CP=${CPAR:-8}  # concurrent slots / client threads (debug)
  if serve par8 262144 QWENOX_PARALLEL=$CP QWENOX_CONC_PREFILL_CHUNK=8192 QWENOX_KVSNAP=0 QWENOX_NGRAM_CHUNK=16; then
    memline par8
    python tools/conc_verify.py conc --tag iq8 --slots $CP > $L/iq4xs_conc_conc.txt 2>&1 \
      && pass "8 路并发 8 组 flow 正常结束" || { fail "8 路并发"; tail -5 $L/iq4xs_conc_conc.txt; }
    grep -E "requests|FAIL" $L/iq4xs_conc_conc.txt | sed 's/^/    /'
    if [[ -f logs/conc/iq1.json && -f logs/conc/iq8.json ]]; then
      python tools/conc_verify.py cmp --a iq1 --b iq8 > $L/iq4xs_conc_cmp.txt 2>&1
      info "并发 vs 单路逐 token（只报告）："; sed 's/^/    /' $L/iq4xs_conc_cmp.txt
      PYTHONIOENCODING=utf-8 python - <<'PY' 2>/dev/null | sed 's/^/    /'
import json, sys
sys.path.insert(0, 'tools')
from qwentok import Tokenizer
tk = Tokenizer()
A = json.load(open('logs/conc/iq1.json'))['res']; B = json.load(open('logs/conc/iq8.json'))['res']
for k in ('nat-d0', 'code-d3', 'conv-T3'):
    if k in A and k in B:
        print(f'[{k}] 单路: ' + tk.decode(A[k]['tokens'])[:110].replace('\n', ' '))
        print(f'[{k}] 并发: ' + tk.decode(B[k]['tokens'])[:110].replace('\n', ' '))
PY
    fi
    python tools/conc_verify.py ctl --slots $CP > $L/iq4xs_conc_ctl.txt 2>&1 \
      && pass "8 路控制面（INFO/PING/取消/断连）" || { fail "8 路控制面"; tail -8 $L/iq4xs_conc_ctl.txt; }
    tr '\r' '\n' < $L/iq4xs_conc_par8.log | grep -E "hipError|HIP error|Exception Code|FATAL|abort" | head -3 \
      | while read -r ln; do fail "引擎日志异常：$ln"; done
  else fail "8 路生产引擎没起来"; fi
  stop_eng
fi

# ---------------------------------------------------------------- 2 显存
if step mem; then
  stamp "2 显存：生产配置（256K、PARALLEL=8、MTP、视觉塔、kvsnap 开）"
  for C in 8192 16384; do
    prod_env
    if serve mem$C 262144 QWENOX_PARALLEL=8 QWENOX_PREFILL_CHUNK=$C QWENOX_CONC_PREFILL_CHUNK=$C \
         QWENOX_KVSNAP=1 QWENOX_KVSNAP_MAX_GB=64 QWENOX_RCKPT_MAX=8; then
      pass "chunk $C 起到 listening"; memline mem$C
    else fail "chunk $C 起不来"; memline mem$C; fi
    stop_eng
  done
fi

# ---------------------------------------------------------------- 3 prefill 分段
if step pp; then
  stamp "3 prefill：32K prompt 单请求，chunk 4096 / 8192 / 16384（含预热）"
  for C in 4096 8192 16384; do
    prod_env; unset QWENOX_NOWARMUP
    (cd build && env QWENOX_PREFILL_CHUNK=$C ./qwenox-engine-win.exe "$GW" \
       --tokens-file "$ROOT/data/qsa-oracle/32768.tokens" --gen 2 --maxctx 40960 \
       > "$ROOT/$L/iq4xs_conc_pp$C.out" 2> "$ROOT/$L/iq4xs_conc_pp$C.log") || fail "pp chunk $C"
    r=$(tr '\r' '\n' < $L/iq4xs_conc_pp$C.log | grep -E "prefill: [0-9]+ tokens in" | tail -1)
    info "chunk $C: ${r:-（无 prefill 行）}"
  done
fi

echo
if [[ ${#FAILS[@]} == 0 ]]; then echo "IQ4XS_CONC PASS"
else printf '失败项：\n'; printf '  - %s\n' "${FAILS[@]}"; echo "IQ4XS_CONC FAIL"; fi
