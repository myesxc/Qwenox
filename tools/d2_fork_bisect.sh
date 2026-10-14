#!/usr/bin/env bash
# d2_fork_bisect.sh — D2-S1a 任务2：γ=2(P=3) 与 γ=3(P=4) 的 greedy ids 在第 1 个
# spec token 处分叉（8192 上下文；γ=1/2 互相一致到 idx 122，γ≥3 全相同）。
# 逐开关二分：同一开关下 γ=2 与 γ=3 各跑一次（--spec-gen 16 足够覆盖分叉点），
# 若某开关下两者 16 个 id 完全一致，则该开关控制的 kernel 就是随 P 变化的源。
#   bash tools/d2_fork_bisect.sh           # 全量（约 25 分钟）
#   ONLY=VGRAPH,BROW_R bash tools/d2_fork_bisect.sh
# 日志 logs/d2fb_*.log。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BIN=build/qwenox-engine-win.exe
M=./models/qwen38-flash-next-v2.hgn
NG=./models/qwen38-flash-next-ngram.hgn
MTP=./models/qwen38-flash-next-mtp.hgn
TOK=logs/d2gw_tok8192.txt
NS="${NS:-16}"
[[ -x $BIN && -f $TOK ]] || { echo "缺 $BIN 或 $TOK（先跑 tools/d2_gate_win.sh 或 head -8192 data/qsa-oracle/131072.tokens）"; exit 1; }

base_env() {
  export QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1
  export QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1
  export QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 QWENOX_NOWARMUP=1
  export QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1
  export QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1
  export QWENOX_KVSNAP=0 QWENOX_RCKPT_MAX=0 QWENOX_KV_PAGED=1 QWENOX_PARALLEL=1
  export QWENOX_ROPE_FACTOR=1 QWENOX_ROPE_ORIGINAL_CTX=262144
  export QWENOX_ROPE_BETA_FAST=32 QWENOX_ROPE_BETA_SLOW=1 QWENOX_ROPE_ATTN_SCALE=0
}

spec_ids() {  # ids 行第 8194 个数起（8192 prompt + next_id）
  grep -a '^ids:' "logs/d2fb_$1.log" | tail -1 | cut -d' ' -f8194- | tr ' ' '\n' | grep -v '^$' | head -n "$NS" | tr '\n' ' '
}

run() {  # run LABEL GAMMA [ENV=...]
  local L=$1 G=$2; shift 2
  echo "  .. $L γ=$G $(date +%T)" >&2
  ( base_env; [[ $# -gt 0 ]] && export "$@";
    "$BIN" "$M" "$NG" "$MTP" --tokens-file "$TOK" --maxctx 16384 \
      --spec-gen "$NS" --gamma "$G" ) > "logs/d2fb_$L.log" 2>&1
  [[ -s "logs/d2fb_$L.log" ]] || echo "  FAIL $L" >&2
}

# 开关清单：名字|γ=3 时相对生产环境的变化（env 覆盖，-u NAME 表示取消）
SWITCHES=(
  "VGRAPH|QWENOX_VGRAPH=0"
  "BROW_R|QWENOX_BROW_R=0"
  "GEMVMR|QWENOX_NO_GEMVMR=1"
  "CONVL2|QWENOX_GDN_CONVL2=0"
  "GDNFUSED|-u QWENOX_GDN_FUSED"
  "IDXF2|-u QWENOX_INDEX_FUSED2"
  "IDXOLD|QWENOX_INDEX_OLDSEL=1"
  "MOELT|-u QWENOX_MOE_LT -u QWENOX_MOE_LT_BF16"
  "PPMOEOUT|-u QWENOX_PP_MOE_OUT"
  "GRBF16|-u QWENOX_GR_BF16"
  "GEMMWMMA|-u QWENOX_GEMM_WMMA"
  "GDNWAVE|-u QWENOX_GDN_WAVE -u QWENOX_GDN_STREAM"
)

mkdir -p logs
echo "== d2_fork_bisect $(date '+%F %T') NS=$NS"

# 0) 基线 + fork 位置 margin（QWENOX_FORK_DUMP 打分叉点 ±4 的 top1/top2）
run base2 2 QWENOX_FORK_DUMP=8194
run base3 3 QWENOX_FORK_DUMP=8194
I2=$(spec_ids base2); I3=$(spec_ids base3)
echo "base γ=2: $I2"
echo "base γ=3: $I3"
echo "base fork margin:"
grep -a '\[fork\]' logs/d2fb_base2.log | head -6 | sed 's/^/  γ2 /'
grep -a '\[fork\]' logs/d2fb_base3.log | head -6 | sed 's/^/  γ3 /'

ONLY="${ONLY:-}"
for ent in "${SWITCHES[@]}"; do
  name=${ent%%|*}; sw=${ent#*|}
  if [[ -n $ONLY && ",$ONLY," != *",$name,"* ]]; then continue; fi
  read -ra SW <<<"$sw"
  run "s${name}2" 2 "${SW[@]}"
  run "s${name}3" 3 "${SW[@]}"
  A=$(spec_ids "s${name}2"); B=$(spec_ids "s${name}3")
  if [[ "$A" == "$B" && -n "$A" ]]; then
    echo "  ** $name: γ=2 与 γ=3 在该开关下完全一致 -> 嫌疑人"
  else
    D=$(python -c "
a='$A'.split(); b='$B'.split()
d=next((i for i in range(min(len(a),len(b))) if a[i]!=b[i]), None)
print('idx', d)" 2>/dev/null || echo '?')
    echo "     $name: 仍分叉（$D）"
  fi
done
echo "== 结束 $(date +%T)"
