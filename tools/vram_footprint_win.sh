#!/usr/bin/env bash
# vram_footprint_win.sh — Windows：各权重 × 上下文/并发配置的显存（arena）实际用量，约 2 分钟/配置。
#
#   bash tools/vram_footprint_win.sh                      # 默认全部配置
#   CASES="v2:262144:0:4:1 v2:262144:524288:8:1" bash tools/vram_footprint_win.sh
#   GGUF=/c/.../UD-IQ4_XS-00001-of-00003.gguf GGUF_MTP=/c/.../mtp-*.gguf bash tools/vram_footprint_win.sh
#
# 每个配置 = 权重:MAX_CONTEXT:KV_POOL_TOKENS:PARALLEL:ROPE_FACTOR，权重为
#   v2（v2 + ngram + mtp.hgn）、w4b（w4b + overlay + mtp.hgn）、iq4（GGUF + MTP sidecar）、
#   iq4nomtp（GGUF，不投机）。所有配置都带视觉塔、prefill chunk 8192（Windows 默认），
#   环境变量与 start_win.sh / 托盘程序的生产组合相同。
# 每个配置起一次引擎（端口 8732，需先停掉生产服务），记录：
#   - devarena 预估 / 预留、arena 实际用量（QWENOX_ARENA_DEBUG 的最大偏移）、hipMalloc 回退次数
#   - memory[ready]、驱动报告的独显占用、引擎工作集
#   - 一条 8K prompt + 64 token 的冒烟请求（tools/vram_smoke.py）
# 文件缺失的配置跳过。结尾 VRAM_FOOTPRINT DONE；汇总在 logs/vram_footprint.txt。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$(pwd)"
L=logs; mkdir -p $L
OUT=$L/vram_footprint.txt; : > $OUT
GGUF="${GGUF:-/c/Users/Mark/Models/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}"
GGUF_MTP="${GGUF_MTP:-$(dirname "$GGUF")/mtp-Qwen3.8-Flash-Next-Q8_0.gguf}"
M=./models
VIS=$M/qwen38-flash-next-vision.hgn
CASES="${CASES:-v2:262144:0:4:1 v2:262144:524288:2:1 v2:262144:524288:8:1 v2:524288:0:2:2
  iq4:262144:0:4:1 iq4:262144:524288:2:1 iq4:262144:524288:8:1 iq4:524288:0:2:2 iq4nomtp:262144:0:4:1
  w4b:262144:0:4:1 w4b:262144:524288:2:1}"
log() { echo "$*" | tee -a $OUT; }
log "head $(git log --oneline -1 | cut -c1-60)  $(date '+%F %T')"

prod_env() {
  export QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1
  export QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1
  export QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 QWENOX_NOWARMUP=1
  export QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1
  export QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1
  export QWENOX_KVSNAP=1 QWENOX_KVSNAP_MAX_GB=64 QWENOX_RCKPT_MAX=8 QWENOX_KV_PAGED=1
  export QWENOX_ROPE_ORIGINAL_CTX=262144 QWENOX_ROPE_BETA_FAST=32 QWENOX_ROPE_BETA_SLOW=1 QWENOX_ROPE_ATTN_SCALE=0
  export QWENOX_ARENA_DEBUG=1
  unset QWENOX_SPEC_GAMMA QWENOX_PREFILL_CHUNK QWENOX_KV_POOL_TOKENS
  unset QWENOX_GGUF QWENOX_GGUF_DENSE QWENOX_GGUF_PLE QWENOX_GGUF_EXPERTS QWENOX_GGUF_MTP
}
killeng() {
  powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { (\$_.Name -like 'qwenox-engine*') -and (\$_.CommandLine -match '8732') } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" > /dev/null 2>&1
  sleep 25   # 显存释放有延迟，紧接着申请 90 GiB 会 OOM
}
vram() { powershell -NoProfile -Command "\$s=(Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage').CounterSamples | Measure-Object CookedValue -Sum; '{0:N2}' -f (\$s.Sum/1GB)"; }
wset() { powershell -NoProfile -Command "\$p=Get-Process qwenox-engine-win -ErrorAction SilentlyContinue | Select-Object -First 1; if(\$p){'{0:N2}' -f (\$p.WorkingSet64/1GB)}else{'-'}"; }
have() { for f in "$@"; do [[ -r "$f" ]] || { log "  SKIP: missing $f"; return 1; }; done; }

run() {  # run <w> <maxctx> <pool> <parallel> <rope_factor>
  local w=$1 mc=$2 pool=$3 par=$4 rf=$5 tag="$1_$2_$3_p$4_rf$5" a=()
  local lg=$L/vram_footprint_$tag.log
  log "=== $tag  weights=$w MAX_CONTEXT=$mc KV_POOL_TOKENS=$pool PARALLEL=$par ROPE_FACTOR=$rf"
  prod_env
  export QWENOX_PARALLEL=$par QWENOX_ROPE_FACTOR=$rf
  (( pool )) && export QWENOX_KV_POOL_TOKENS=$pool
  case $w in
    v2)  have $M/qwen38-flash-next-v2.hgn $M/qwen38-flash-next-ngram.hgn $M/qwen38-flash-next-mtp.hgn || return
         a=($M/qwen38-flash-next-v2.hgn $M/qwen38-flash-next-ngram.hgn $M/qwen38-flash-next-mtp.hgn) ;;
    w4b) have $M/qwen38-flash-next-w4b.hgn $M/qwen38-flash-next-w4b.overlay.hgn $M/qwen38-flash-next-mtp.hgn || return
         a=($M/qwen38-flash-next-w4b.hgn $M/qwen38-flash-next-w4b.overlay.hgn $M/qwen38-flash-next-mtp.hgn) ;;
    iq4) have "$GGUF" "$GGUF_MTP" || return
         export QWENOX_GGUF_MTP="$(cygpath -m "$GGUF_MTP")"; a=("$(cygpath -m "$GGUF")") ;;
    iq4nomtp) have "$GGUF" || return
         export QWENOX_GGUF_MTP=none; a=("$(cygpath -m "$GGUF")") ;;
    *) log "  SKIP: unknown weights $w"; return ;;
  esac
  local v0 t0=$SECONDS pid ok=0
  v0=$(vram)
  build/qwenox-engine-win.exe "${a[@]}" --serve --host 127.0.0.1 --port 8732 --maxctx $mc \
      --vision-tower $VIS > $lg 2>&1 &
  pid=$!
  while (( SECONDS - t0 < 900 )); do
    grep -q 'serve: listening' $lg 2>/dev/null && { ok=1; break; }
    kill -0 $pid 2>/dev/null || break
    sleep 3
  done
  if (( ! ok )); then
    log "  START FAIL after $((SECONDS - t0)) s"
    tr '\r' '\n' < $lg | grep -v '^\s*$' | tail -8 | sed 's/^/    /' | tee -a $OUT
    killeng; return
  fi
  log "  ready in $((SECONDS - t0)) s; driver dedicated VRAM $v0 -> $(vram) GiB; engine working set $(wset) GiB"
  tr '\r' '\n' < $lg | grep -E "^devarena:|^arena: |^weight arena: [0-9]|^memory\[ready\]|^\[kvpage\] (paged|pool)|^RoPE" \
    | sed 's/^/    /' | tee -a $OUT
  # arena-debug: "<name> <MiB> MiB @ <GiB> GiB" 或 "... @ n/a (hipMalloc fallback)"
  tr '\r' '\n' < $lg | awk '/^arena-debug:/ && /@ [0-9.]+ GiB/ { e = $6 + $3 / 1024; if (e > m) m = e }
      /hipMalloc fallback/ { fb++; fbm += $3 }
      END { printf "    ARENA USED %.2f GiB; hipMalloc fallbacks %d (%.1f MiB)\n", m, fb, fbm }' | tee -a $OUT
  python tools/vram_smoke.py data/qsa-oracle/32768.tokens 8192 64 2>&1 | sed 's/^/    /' | tee -a $OUT
  log "  after smoke: driver dedicated VRAM $(vram) GiB; engine working set $(wset) GiB"
  killeng
}

killeng
for c in $CASES; do IFS=: read -r w mc pool par rf <<< "$c"; run "$w" "$mc" "$pool" "$par" "$rf"; done
log ""
sum=$(awk '/^=== / { t = $2 } /^ +ARENA USED / { printf "  %-34s %s GiB, hipMalloc fallbacks %s\n", t, $3, $7 }
     /^ +(START FAIL|SKIP:)/ { sub(/^ +/, ""); printf "  %-34s %s\n", t, $0 }' $OUT)
log "summary (ARENA USED, cap 95 GiB):"
log "$sum"
log "VRAM_FOOTPRINT DONE $(date '+%F %T')"
