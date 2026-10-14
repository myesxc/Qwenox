#!/usr/bin/env bash
# conc_diverge_win.sh — 并发 vs 单跑分叉排查（dev-docs/HANDOFF-CONC-DIVERGE.md §4 E1–E3）一键跑。
# Windows Git Bash（Linux 也能跑：BIN/模型路径自己给）。默认 v2 hgn，不带视觉塔。
#
#   bash tools/conc_diverge_win.sh
#   CONFIGS="base|QWENOX_OCC_FORCE=4" PAR=4 bash tools/conc_diverge_win.sh
#
# 流程：seq 参考跑 NREF 次（默认 2，每次新起引擎，PARALLEL=$PAR，单连接逐个跑 conc_verify 的
# 10 个请求），再按 CONFIGS（'|' 分隔，每项是空格分隔的 ENV=V，"base" = 不加）每项新起一个
# 引擎、$PAR 路并发跑同一组请求。所有引擎同一 --maxctx / chunk / 生产内核组合，seq 引擎也带
# 该项的 ENV（greedy 单跑时 OCC_FORCE=4 不起作用：B=1 本来就是 γ4）。
# 只比 token（并发时 rounds/proposed 本来就不同）：每个请求打印 = 或第一个不同的位置。
# 采样流（S-*）默认没带 SEEDED，只报告不计入 greedy 判定；CONC_SEEDED=1 让它们带 SEEDED
# （此时看 all 列）。GGUF：MODEL_ARGS=<x-00001-of-0000N.gguf> GGUF_MTP=<mtp sidecar>。
# 表后每个 run 一行 "RESULT <tag> greedy=MATCH|DIFF all=MATCH|DIFF"（给回归脚本用）。
# 结尾 CONC_DIVERGE PASS = 每步都跑完（结论看表）；FAIL = 有引擎/请求出错。
# 日志 logs/<PFX>_*.log（PFX 默认 cdiv），结果 logs/conc/<PFX>_*.json，表 logs/<PFX>_summary.txt。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$(pwd)"
L=logs; mkdir -p $L logs/conc
win() { if command -v cygpath > /dev/null; then cygpath -m "$1"; else echo "$1"; fi; }
if [[ "$(uname -s)" == MINGW* || "$(uname -s)" == MSYS* ]]; then
  BIN=${BIN:-$ROOT/build/qwenox-engine-win.exe}
else
  BIN=${BIN:-$ROOT/build/qwenox-engine}
fi
M=$ROOT/models
MODEL_ARGS="${MODEL_ARGS:-$(win $M/qwen38-flash-next-v2.hgn) $(win $M/qwen38-flash-next-ngram.hgn) $(win $M/qwen38-flash-next-mtp.hgn)}"
PAR=${PAR:-4}
NREF=${NREF:-2}
PFX=${PFX:-cdiv}
GGUF_MTP=${GGUF_MTP:-}
export CONC_SEEDED=${CONC_SEEDED:-0}
MAXCTX=${MAXCTX:-262144}
CONFIGS="${CONFIGS:-base|base|QWENOX_OCC_FORCE=4|QWENOX_OCC_FORCE=4 QWENOX_NO_DR=1|QWENOX_OCC_FORCE=4 QWENOX_NO_DR=1 QWENOX_NO_VR=1}"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
FAILS=()
fail() { FAILS+=("$1"); echo "  FAIL: $1"; }
stamp() { echo; echo "=== [$(date +%H:%M:%S)] $*"; }

[[ -f "$BIN" ]] || { echo "缺引擎：$BIN"; echo "CONC_DIVERGE FAIL"; exit 1; }
for f in $MODEL_ARGS; do
  [[ -f "$f" || -f "$(cygpath -u "$f" 2>/dev/null)" ]] || { echo "缺模型：$f"; echo "CONC_DIVERGE FAIL"; exit 1; }
done
if command -v tasklist > /dev/null && tasklist 2>/dev/null | grep -qi "qwenox"; then
  echo "检测到 qwenox 进程在运行，请先停掉服务（显存要整块用）"; echo "CONC_DIVERGE FAIL"; exit 1
fi
echo "head $(git log --oneline -1 | cut -c1-70)"
echo "bin  $(ls -l --time-style=+%F_%T "$BIN" | awk '{print $6}')  PAR=$PAR MAXCTX=$MAXCTX NREF=$NREF"

prod_env() {  # start_win.sh 同款生产内核组合（与 iq4xs_conc_win.sh 相同；GGUF_MTP 仅 GGUF 用）
  for e in $(compgen -e | grep '^QWENOX_'); do unset "$e"; done
  export QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1
  export QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1
  export QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 QWENOX_NOWARMUP=1
  export QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1
  export QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1
  export QWENOX_KV_PAGED=1 QWENOX_ROPE_FACTOR=1 QWENOX_ROPE_ORIGINAL_CTX=262144
  export QWENOX_ROPE_BETA_FAST=32 QWENOX_ROPE_BETA_SLOW=1 QWENOX_ROPE_ATTN_SCALE=0
  export QWENOX_PARALLEL=$PAR QWENOX_PREFILL_CHUNK=8192 QWENOX_CONC_PREFILL_CHUNK=8192
  export QWENOX_KVSNAP=0 QWENOX_NGRAM_CHUNK=16
  [[ -n "$GGUF_MTP" ]] && export QWENOX_GGUF_MTP="$GGUF_MTP"
  return 0
}

ENG_PID=""
serve() {  # serve TAG "ENV=V ..." : 起引擎到 listening（最多 10 分钟）
  local tag=$1 envs=$2
  prod_env
  [[ "$envs" != base ]] && for e in $envs; do export "$e"; done
  local log=$ROOT/$L/${PFX}_$tag.log
  local margs; read -r -a margs <<< "$MODEL_ARGS"
  (cd build && exec "$BIN" "${margs[@]}" --serve --host 127.0.0.1 --port 8732 \
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
  sleep 4; ENG_PID=""
}
trap stop_eng EXIT

run() {  # run TAG MODE(seq|conc) "ENV..."
  local tag=$1 mode=$2 envs=$3
  rm -f logs/conc/${PFX}_$tag.json
  if ! serve "$tag" "$envs"; then fail "$tag 引擎没起来"; stop_eng; return; fi
  local extra=(); [[ $mode == conc ]] && extra=(--slots $PAR)
  if timeout 1200 python tools/conc_verify.py $mode --tag ${PFX}_$tag "${extra[@]}" > $L/${PFX}_$tag.txt 2>&1; then
    grep -E "requests" $L/${PFX}_$tag.txt | sed 's/^/    /'
  else
    fail "$tag 请求出错"; tail -5 $L/${PFX}_$tag.txt | sed 's/^/    /'
  fi
  tr '\r' '\n' < $L/${PFX}_$tag.log | grep -E "hipError|HIP error|Exception Code|FATAL|abort" | head -3 \
    | while read -r ln; do echo "  引擎日志异常：$ln"; done
  stop_eng
}

TAGS=()
for i in $(seq 1 $NREF); do
  stamp "seq 参考 #$i（PARALLEL=$PAR，单连接逐个跑）"
  run ref$i seq base; TAGS+=("ref$i")
done
IFS='|' read -r -a CFG <<< "$CONFIGS"
n=0
for c in "${CFG[@]}"; do
  n=$((n + 1))
  stamp "conc #$n：$PAR 路并发，ENV=[$c]"
  run c$n conc "$c"; TAGS+=("c$n")
  printf '%s\n' "$c" > logs/conc/${PFX}_c$n.env
done

stamp "对比（基准 ref1，只比 token；= 相同，数字 = 第一个不同的生成 token 位置）"
PFX=$PFX python - "${TAGS[@]}" <<'PY' | tee $L/${PFX}_summary.txt
import json, re, sys
from pathlib import Path
import os
PFX = os.environ['PFX']
SEEDED = os.environ.get('CONC_SEEDED') == '1'
tags = sys.argv[1:]
C = Path('logs/conc'); Lg = Path('logs')
def load(t):
    p = C / f'{PFX}_{t}.json'
    return json.loads(p.read_text())['res'] if p.exists() else None
ref = load('ref1')
if ref is None:
    print('ref1 没有结果'); sys.exit(0)
keys = list(ref)
def fd(a, b):
    i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if i is None:
        return '=' if len(a) == len(b) else f'len{len(a)}/{len(b)}'
    return str(i)
def occ(t):  # occ-gamma histogram + QWENOX_VR_CHECK clobbers over the engine log
    p = Lg / f'{PFX}_{t}.log'
    if not p.exists():
        return ''
    h, clob, refused, last = {}, {}, 0, ''
    for ln in p.read_text(errors='replace').replace('\r', '\n').split('\n'):
        m = re.search(r'occ-gamma: \d+ rounds at B>1, γ(.*?)\|', ln)
        if m:
            for g, k in re.findall(r'(\d+):(\d+)', m.group(1)):
                h[int(g)] = h.get(int(g), 0) + int(k)
        m = re.search(r'CLOBBERED: (\S+)', ln)
        if m:
            for b in m.group(1).split(','):
                clob[b] = clob.get(b, 0) + 1
        m = re.search(r'\[vr-check\] ok (\d+) bad (\d+)', ln)
        if m:
            last = f'{m.group(1)}/{m.group(2)}'
        refused += 'batched verify refused' in ln
    s = ' '.join(f'{g}:{h[g]}' for g in sorted(h)) or '-'
    if clob or last:
        s += '  clobber[' + ' '.join(f'{b}:{n}' for b, n in clob.items()) + (f'] ok/bad~{last}' if last else ']')
    if refused:
        s += f'  refused:{refused}'
    return s
res = []
short = [k.replace('conv-', '') for k in keys]
print(f'{"run":5s} {"env":44s} ' + ' '.join(f'{s:>7s}' for s in short) + '  greedy  occ γ(rounds)')
for t in tags:
    r = load(t)
    envp = C / f'{PFX}_{t}.env'
    env = envp.read_text().strip() if envp.exists() else ('seq' if t.startswith('ref') else '')
    if r is None:
        print(f'{t:5s} {env:44s} (no result)'); continue
    cells = [fd(ref[k]['tokens'], r[k]['tokens']) if k in r else 'miss' for k in keys]
    g = all(c == '=' for k, c in zip(keys, cells) if not k.startswith('S-'))
    a = all(c == '=' for c in cells)
    print(f'{t:5s} {env[:44]:44s} ' + ' '.join(f'{c:>7s}' for c in cells)
          + f'  {"MATCH" if g else "DIFF ":5s}   {occ(t)}')
    res.append(f'RESULT {t} greedy={"MATCH" if g else "DIFF"} all={"MATCH" if a else "DIFF"}')
print('greedy = 除 S-*（采样）外全部请求与 ref1 逐 token 相同'
      + ('；本轮采样带 SEEDED，all = 连 S-* 一起比' if SEEDED else '；S-* 没带 SEEDED，只报告'))
print('\n'.join(res))
PY

echo
if [[ ${#FAILS[@]} == 0 ]]; then echo "CONC_DIVERGE PASS"
else printf '失败项：\n'; printf '  - %s\n' "${FAILS[@]}"; echo "CONC_DIVERGE FAIL"; fi
