#!/usr/bin/env bash
# cutspan_ab_win.sh — A/B the QWENOX_CUT_MIN_SPAN change on the real weights.
# A = new engine exe with QWENOX_CUT_MIN_SPAN=0 (legacy: every cut splits),
# B = same exe, default (64). Stops the production engine first, verifies the
# GPU memory is actually released before every engine start, isolates the
# test engine's kvsnap dir and the test API's data files from production,
# and restarts the production engine at the end.
set -u
R=/d/llama/gfx1151-engine/qwenox
S=/d/llama/gfx1151-engine/qwenox-code/Qwenox
OUT=$R/logs/perf_ab
API_PORT=8731
TE_PORT=8732
TA_PORT=8733
TE_EXE=qwenox-engine-win-new.exe
KVDIR_WIN="D:/llama/gfx1151-engine/qwenox/data/kvsnap_abtest"
APITMP=/tmp/apitest_cutspan

log() { echo "[$(date +%H:%M:%S)] $*"; }
gpu() {
    powershell -NoProfile -Command '
        $c = (Get-Counter "\GPU Adapter Memory(*)\Dedicated Usage","\GPU Adapter Memory(*)\Shared Usage" -ErrorAction SilentlyContinue).CounterSamples
        $g = @{}
        foreach ($s in $c) { $g[$s.InstanceName] = [double]$g[$s.InstanceName] + $s.CookedValue }
        ($g.Values | Measure-Object -Maximum).Maximum
    ' 2>/dev/null | tr -d '\r' | awk '{printf "%.1f", $1/1073741824}'
}
eprocs() {
    { tasklist //FI "IMAGENAME eq qwenox-engine-win.exe"; tasklist //FI "IMAGENAME eq $TE_EXE"; } 2>/dev/null \
        | grep -c "qwenox-engine-win" || true
}
wait_gpu_free() {  # engine processes gone AND gpu <= 10 GiB
    for _ in $(seq 1 120); do [[ "$(eprocs)" == 0 ]] && break; sleep 1; done
    [[ "$(eprocs)" == 0 ]] || { log "FATAL: engine process still alive"; exit 1; }
    for _ in $(seq 1 60); do g=$(gpu); awk -v g="$g" 'BEGIN{exit !(g+0 <= 10)}' && break; sleep 2; done
    g=$(gpu); awk -v g="$g" 'BEGIN{exit !(g+0 <= 10)}' || { log "FATAL: GPU still $g GiB"; exit 1; }
    log "gpu free: $g GiB, engine procs 0"
}
kill_test_api() {
    local pid
    pid=$(netstat -ano 2>/dev/null | grep -E "TCP +127.0.0.1:$TA_PORT .*LISTENING" | awk '{print $NF}' | head -1 | tr -d '\r')
    [[ -n "$pid" ]] && taskkill //PID "$pid" //F >/dev/null 2>&1
}
restore_prod() {
    log "restoring production engine"
    curl -s -m 10 -X POST "http://127.0.0.1:$API_PORT/admin/engine/start" | head -c 200; echo
}
trap 'kill_test_api; taskkill //IM '"$TE_EXE"' //F >/dev/null 2>&1; restore_prod' EXIT

cp -f "$S/build/qwenox-engine-win.exe" "$R/build/$TE_EXE" || exit 1
mkdir -p "$APITMP/data" "$OUT"

log "stopping production engine"
curl -s -m 10 -X POST "http://127.0.0.1:$API_PORT/admin/engine/stop" | head -c 200; echo
wait_gpu_free

for cfg in A B; do
    rm -rf "$R/data/kvsnap_abtest"; mkdir -p "$R/data/kvsnap_abtest"
    rm -f "$APITMP/data/tcache.bin"
    export QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1 \
           QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1 \
           QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 QWENOX_NOWARMUP=1 \
           QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1 QWENOX_INDEX_FUSED2=1 \
           QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1 QWENOX_KV_PAGED=1 \
           QWENOX_KVSNAP=1 QWENOX_KVSNAP_MAX_GB=5 QWENOX_KVSNAP_DIR="$KVDIR_WIN" \
           QWENOX_RCKPT_MAX=8 QWENOX_PROF=1
    if [[ $cfg == A ]]; then export QWENOX_CUT_MIN_SPAN=0; else unset QWENOX_CUT_MIN_SPAN; fi
    log "== $cfg: starting test engine (CUT_MIN_SPAN=${QWENOX_CUT_MIN_SPAN:-default})"
    EL=$OUT/cutspan_$cfg.engine.log
    ( cd "$R" && exec "./build/$TE_EXE" models/qwen38-flash-next-v2.hgn models/qwen38-flash-next-ngram.hgn \
        models/qwen38-flash-next-mtp.hgn --serve --host 127.0.0.1 --port $TE_PORT --maxctx 32768 ) > "$EL" 2>&1 &
    for _ in $(seq 1 60); do grep -q "serve: listening" "$EL" 2>/dev/null && break; sleep 5; done
    grep -q "serve: listening" "$EL" || { log "FATAL: engine did not start"; tail -5 "$EL"; exit 1; }
    log "engine up (gpu $(gpu) GiB); starting test API"
    ( cd "$APITMP" && exec "$R/build/qwenox-win.exe" --console --tokenizer "$R/models/tokenizer" \
        --engine 127.0.0.1:$TE_PORT --host 127.0.0.1 --port $TA_PORT --context 32768 ) > "$OUT/cutspan_$cfg.api.log" 2>&1 &
    for _ in $(seq 1 30); do curl -s -m 3 "http://127.0.0.1:$TA_PORT/v1/models" >/dev/null 2>&1 && break; sleep 2; done
    log "running turns"
    python -I "$S/tools/chat_turns_replay.py" $TA_PORT $cfg 2>&1 | tee "$OUT/cutspan_$cfg.turns"
    log "== $cfg done; stopping test API + engine"
    kill_test_api; sleep 1
    taskkill //IM "$TE_EXE" //F >/dev/null 2>&1
    wait_gpu_free
done
rm -rf "$R/data/kvsnap_abtest"
log "A/B finished"
