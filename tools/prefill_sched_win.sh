#!/usr/bin/env bash
# prefill_sched_win.sh - one-shot check of the 10-13 prefill scheduling fix
# (PLE prefetch owner check + light-only lending + shortest-remaining-first).
# Windows Git Bash (Linux works too: give BIN_OLD/BIN_NEW). v2 hgn, 2 slots.
#
#   bash tools/prefill_sched_win.sh
#   BIN_OLD=<old engine exe> BIN_NEW=<new engine exe> bash tools/prefill_sched_win.sh
#
# 1. old engine: tools/prefill_sched.py run (T1-T4)       ~6 min
# 2. new engine: same, then conc_verify seq + conc (2 slots), tokens compared
# 3. gate: new concurrent == solo (24K, 64K), 2x64K TTFT vs solo and vs old,
#    decode stall not worse, 8K prompt not starved by a 64K prefill.
# Needs the service stopped (whole-GPU arena). Last line PREFILL_SCHED PASS|FAIL.
# Logs: logs/psched_<old|new>.log (engine), logs/psched_*.txt, logs/psched/*.json
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PSCHED_ROOT:-$(cd "$HERE/.." && pwd)}"
cd "$ROOT"
export PSCHED_ROOT="$ROOT"
PY="$HERE/prefill_sched.py"
L=logs; mkdir -p $L logs/psched logs/conc
win() { if command -v cygpath > /dev/null; then cygpath -m "$1"; else echo "$1"; fi; }
if [[ "$(uname -s)" == MINGW* || "$(uname -s)" == MSYS* ]]; then EXE=qwenox-engine-win.exe; else EXE=qwenox-engine; fi
BIN_OLD=${BIN_OLD:-$ROOT/build/$EXE}
BIN_NEW=${BIN_NEW:-$HOME/Workspace/gfx1151-srpt/build/$EXE}
M=$ROOT/models
MODEL_ARGS="${MODEL_ARGS:-$(win $M/qwen38-flash-next-v2.hgn) $(win $M/qwen38-flash-next-ngram.hgn) $(win $M/qwen38-flash-next-mtp.hgn)}"
MAXCTX=${MAXCTX:-262144}
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
FAILS=()
fail() { FAILS+=("$1"); echo "  FAIL: $1"; }
stamp() { echo; echo "=== [$(date +%H:%M:%S)] $*"; }

for b in "$BIN_OLD" "$BIN_NEW"; do
  [[ -f "$b" ]] || { echo "missing engine: $b"; echo "PREFILL_SCHED FAIL"; exit 1; }
done
for f in $MODEL_ARGS; do
  [[ -f "$f" || -f "$(cygpath -u "$f" 2>/dev/null)" ]] || { echo "missing model: $f"; echo "PREFILL_SCHED FAIL"; exit 1; }
done
if command -v tasklist > /dev/null && tasklist 2>/dev/null | grep -qi "qwenox"; then
  echo "a qwenox process is running - stop the service first (the test needs the whole GPU)"
  echo "PREFILL_SCHED FAIL"; exit 1
fi
echo "root $ROOT  head $(git log --oneline -1 | cut -c1-60)"
echo "old  $BIN_OLD ($(ls -l --time-style=+%F_%T "$BIN_OLD" | awk '{print $6}'))"
echo "new  $BIN_NEW ($(ls -l --time-style=+%F_%T "$BIN_NEW" | awk '{print $6}'))"

prod_env() {  # start_win.sh production kernel set (same as conc_diverge_win.sh), 2 slots
  for e in $(compgen -e | grep '^QWENOX_'); do unset "$e"; done
  export QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1
  export QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1
  export QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 QWENOX_NOWARMUP=1
  export QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1
  export QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1
  export QWENOX_KV_PAGED=1 QWENOX_ROPE_FACTOR=1 QWENOX_ROPE_ORIGINAL_CTX=262144
  export QWENOX_ROPE_BETA_FAST=32 QWENOX_ROPE_BETA_SLOW=1 QWENOX_ROPE_ATTN_SCALE=0
  export QWENOX_PARALLEL=2 QWENOX_PREFILL_CHUNK=8192 QWENOX_CONC_PREFILL_CHUNK=8192
  export QWENOX_KVSNAP=0 QWENOX_RCKPT=0 QWENOX_NGRAM_CHUNK=16 QWENOX_PLE_MISS_LOG=1
  return 0
}

ENG_PID=""
serve() {  # serve TAG BIN
  local tag=$1 bin=$2
  prod_env
  local log=$ROOT/$L/psched_$tag.log
  local margs; read -r -a margs <<< "$MODEL_ARGS"
  (cd "$(dirname "$bin")" && exec "$bin" "${margs[@]}" --serve --host 127.0.0.1 --port 8732 \
      --maxctx "$MAXCTX" > "$log" 2>&1) &
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
  if command -v powershell > /dev/null; then
    powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { (\$_.Name -like 'qwenox-engine*') -and (\$_.CommandLine -match '8732') } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" > /dev/null 2>&1
  fi
  sleep 5; ENG_PID=""
}
trap stop_eng EXIT

engine_errors() {
  tr '\r' '\n' < $L/psched_$1.log | grep -E "hipError|HIP error|Exception Code|FATAL|abort|KV pool exhausted|cannot fit|rejected" | head -3 \
    | while read -r ln; do echo "  engine log: $ln"; done
}

if [[ "${SMOKE:-0}" == 1 ]]; then  # quick liveness check of the new build only
  stamp "smoke: new engine"
  serve smoke "$BIN_NEW" || { echo "PREFILL_SCHED FAIL"; exit 1; }
  timeout 600 python "$PY" smoke 2>&1 | tee $L/psched_smoke.txt
  rc=${PIPESTATUS[0]}
  tr '\r' '\n' < $L/psched_smoke.log | grep -E "prefill scheduling|ple: drop|lends GPU" | sort | uniq -c | sort -rn | head -6
  engine_errors smoke
  stop_eng
  [[ $rc == 0 ]] && echo "PREFILL_SCHED SMOKE PASS" || echo "PREFILL_SCHED SMOKE FAIL"
  exit $rc
fi

stamp "1/3 old engine"
if serve old "$BIN_OLD"; then
  timeout 2400 python "$PY" run --tag old 2>&1 | tee $L/psched_old.txt
  [[ ${PIPESTATUS[0]} == 0 ]] || fail "old run had errors"
  engine_errors old
else
  fail "old engine did not start"
fi
stop_eng

stamp "2/3 new engine"
if serve new "$BIN_NEW"; then
  timeout 2400 python "$PY" run --tag new 2>&1 | tee $L/psched_new.txt
  [[ ${PIPESTATUS[0]} == 0 ]] || fail "new run had errors"
  echo "  conc_verify flows: seq then 2-way concurrent (tokens must match)"
  rm -f logs/conc/psched_seq.json logs/conc/psched_conc.json
  if timeout 1200 python tools/conc_verify.py seq --tag psched_seq > $L/psched_cv_seq.txt 2>&1 &&
     timeout 1200 python tools/conc_verify.py conc --tag psched_conc --slots 2 > $L/psched_cv_conc.txt 2>&1; then
    grep -h "requests" $L/psched_cv_seq.txt $L/psched_cv_conc.txt | sed 's/^/    /'
  else
    fail "conc_verify flows had errors"; tail -4 $L/psched_cv_conc.txt | sed 's/^/    /'
  fi
  engine_errors new
else
  fail "new engine did not start"
fi
stop_eng

stamp "3/3 gate"
python - <<'PY' | tee $L/psched_cv_cmp.txt
import json
from pathlib import Path
C = Path('logs/conc')
try:
    A = json.loads((C / 'psched_seq.json').read_text())['res']
    B = json.loads((C / 'psched_conc.json').read_text())['res']
except Exception as e:
    print('conc_verify compare: missing results', e); print('CV FAIL'); raise SystemExit
bad = [k for k in A if k not in B or A[k]['tokens'] != B[k]['tokens']]
greedy_bad = [k for k in bad if not k.startswith('S-')]
print(f'conc_verify tokens seq vs 2-way: {len(A) - len(bad)}/{len(A)} identical'
      + (f'; differ: {bad}' if bad else '') + ' (S-* = unseeded sampling, info only)')
print('CV', 'PASS' if not greedy_bad else 'FAIL')
PY
grep -q "^CV PASS" $L/psched_cv_cmp.txt || fail "conc_verify greedy flows differ between seq and concurrent"
python "$PY" gate --old old --new new --olog $L/psched_old.log --nlog $L/psched_new.log | tee $L/psched_gate.txt
[[ ${PIPESTATUS[0]} == 0 ]] || fail "gate"

echo
if (( ${#FAILS[@]} )); then
  printf '  - %s\n' "${FAILS[@]}"
  echo "PREFILL_SCHED FAIL"; exit 1
fi
echo "PREFILL_SCHED PASS"
