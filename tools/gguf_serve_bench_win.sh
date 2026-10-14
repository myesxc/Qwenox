#!/usr/bin/env bash
# gguf_serve_bench_win.sh — Windows：经 API 实测并发吞吐（tools/conc_bench.py，MTP 友好的代码题，
# greedy，每 worker 2 个请求 × 1200 token），对比三种配置（约 30 分钟）：
#   gguf_new  UD-IQ4_XS GGUF，当前默认（dense q8g32 WMMA ≤48 行 + 专家去重 GEMV ≤48 行）
#   gguf_old  同一二进制，QWENOX_DENSE_WMMA_Q8=0 QWENOX_GG_GEMV=8（改动前的 GGUF 路径）
#   v2        生产 v2 hgn（参照）
# 每种都用临时 conf（测试端口 18730/18731，PARALLEL=8，MAX_CONTEXT=65536，MTP 开，kvsnap 关）
# 起 qwenox-win → /admin/engine/start → conc_bench --levels 1,4,8。结尾打印汇总表。
#   CFGS="gguf_new v2" LEVELS=1,8 bash tools/gguf_serve_bench_win.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$(pwd)"
GGUF="${GGUF:-/c/Users/Mark/Models/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}"
MTPG="${MTPG:-$(dirname "$GGUF")/mtp-Qwen3.8-Flash-Next-Q8_0.gguf}"
CFGS="${CFGS:-gguf_new gguf_old v2}"
LEVELS="${LEVELS:-1,4,8}"
L=logs/gsb; mkdir -p $L
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
if tasklist 2>/dev/null | grep -qi "qwenox"; then
  echo "检测到 qwenox 进程在运行，请先停掉服务"; echo "GGUF_SERVE_BENCH FAIL"; exit 1
fi
CONF=$ROOT/service.gsb-test.conf
API=http://127.0.0.1:18731
kill_test() {
  powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { (\$_.Name -like 'qwenox*') -and (\$_.CommandLine -match 'gsb-test|18730') } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" > /dev/null 2>&1
}
trap 'kill_test; rm -f "$CONF" "$CONF.bak"' EXIT
FAIL=0
for cfg in $CFGS; do
  echo; echo "=== [$(date +%H:%M:%S)] $cfg"
  for e in $(compgen -e | grep '^QWENOX_'); do unset "$e"; done
  {
    echo '# gguf_serve_bench_win.sh 临时配置（跑完删除）'
    echo 'MODEL_DIR="./models"'
    echo 'TOKENIZER_DIR="./models/tokenizer"'
    if [[ $cfg == v2 ]]; then
      echo 'MODEL_FILE="./models/qwen38-flash-next-v2.hgn"'
      echo 'NGRAM_FILE="./models/qwen38-flash-next-ngram.hgn"'
      echo 'OVERLAY_FILE=""'
      echo 'MTP_FILE="./models/qwen38-flash-next-mtp.hgn"'
      echo 'VISION_FILE=""'
    else
      echo "GGUF_FILE=\"$(cygpath -m "$GGUF")\""
      echo "GGUF_MTP_FILE=\"$(cygpath -m "$MTPG")\""
      echo 'GGUF_VISION_FILE=""'
    fi
    echo 'ENGINE_HOST="127.0.0.1"'
    echo 'ENGINE_PORT="18730"'
    echo 'API_HOST="127.0.0.1"'
    echo 'API_PORT="18731"'
    echo 'MAX_CONTEXT="65536"'
    echo 'PARALLEL="8"'
    echo 'KV_PAGED="1"'
    echo 'KVSNAP_MAX_GB="0"'
    echo 'MTP_GAMMA="0"'
    [[ $cfg == v2 ]] && echo '# start_win: weights=v2' || echo '# start_win: weights=gguf'
  } > "$CONF"
  # 引擎子进程继承 qwenox-win 的环境
  [[ $cfg == gguf_old ]] && export QWENOX_DENSE_WMMA_Q8=0 QWENOX_GG_GEMV=8
  # BENCH_ENV="QWENOX_X=0 ..." (A/B): extra engine env for every config (the loop
  # clears QWENOX_* above, so plain exports from the caller are lost)
  for e in ${BENCH_ENV:-}; do export "$e"; done
  ./build/qwenox-win.exe --console --service-conf "$(cygpath -w "$CONF")" > "$L/api_$cfg.log" 2>&1 &
  ok=0
  for i in $(seq 60); do curl -s -m 2 $API/admin/engine/status > /dev/null && { ok=1; break; }; sleep 1; done
  [[ $ok == 1 ]] || { echo "  API 没起来"; FAIL=1; kill_test; continue; }
  curl -s -m 10 -X POST $API/admin/engine/start > /dev/null
  st=""
  for i in $(seq 200); do
    st=$(curl -s -m 3 $API/admin/engine/status | sed -n 's/.*"state":"\([a-z]*\)".*/\1/p')
    [[ $st == running || $st == stopped || $st == exited || $st == failed ]] && break
    sleep 3
  done
  echo "  engine: $st"
  if [[ $st == running ]]; then
    python tools/conc_bench.py --base-url $API --levels "$LEVELS" > "$L/bench_$cfg.txt" 2>&1 || FAIL=1
    grep -E "^ *[0-9]+ +\||level|并发|agg|tok/s" "$L/bench_$cfg.txt" | tail -12 | sed 's/^/    /'
  else
    FAIL=1
  fi
  curl -s -m 30 -X POST $API/admin/engine/stop > /dev/null
  sleep 5
  kill_test; sleep 3
  unset QWENOX_DENSE_WMMA_Q8 QWENOX_GG_GEMV
  for e in ${BENCH_ENV:-}; do unset "${e%%=*}"; done
done
echo
for cfg in $CFGS; do echo "--- $cfg"; tail -14 "$L/bench_$cfg.txt" 2>/dev/null; done
[[ $FAIL == 0 ]] && echo "GGUF_SERVE_BENCH DONE" || echo "GGUF_SERVE_BENCH FAIL"
