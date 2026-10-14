#!/usr/bin/env bash
# conc_fix_regress_win.sh — 会合 verify 行窗口备份（QWENOX_VR_RESTORE，默认开）的一键回归。
# Windows Git Bash，约 70 分钟。先 bash build_win.sh engine。
#
#   bash tools/conc_fix_regress_win.sh
#   ONLY="v2c4 ids" bash tools/conc_fix_regress_win.sh
#
# 背景：合批 verify 之后执行者交出 GPU，成员回来前，共享回滚行（d_Rb / d_grow / d_csnap /
# d_iproj_snap）可能被别的请求改写 → 并发输出与单跑分叉。修法：执行者按槽备份，成员回来先拷回。
#
# 门槛（结尾 CONC_FIX PASS / FAIL）：
#   v2c4    v2 4 路：默认 ×3 + VR_CHECK=1 ×1，greedy 全 MATCH；VR_RESTORE=0 对照 ×1（只报告，预期 DIFF）
#   v2c8    v2 8 路 ×3：greedy 全 MATCH
#   seeded  v2 4 路 ×2，采样流带 SEEDED：连采样在内全 MATCH
#   gguf    GGUF UD-IQ4_XS 4 路、8 路各 OCC_FORCE=4 ×2：greedy 全 MATCH（base 按人数选 γ 只报告，
#           GGUF 对 γ 不严格不变，见 docs/CONCURRENCY.md）
#   ids     tools/d2_s2_ids.sh（同 prompt 2/4 并发 + ngram 混跑，逐 bit）：S2 IDS ACCEPT: PASS
#   vseg    v2 与 GGUF 的 QWENOX_VSEG_SELFTEST=3,4,5 / 5,5,5,5,5,5,5,5：PASS
# 只报告：bench  API 吞吐 C4/C8（tools/gguf_serve_bench_win.sh）：v2 默认 vs QWENOX_VR_RESTORE=0，GGUF 默认。
# 日志 logs/cfix/*，各阶段对比表 logs/<阶段>_summary.txt。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$(pwd)"
L=logs/cfix; mkdir -p $L
GGUF="${GGUF:-/c/Users/Mark/Models/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}"
MTPG="${MTPG:-$(dirname "$GGUF")/mtp-Qwen3.8-Flash-Next-Q8_0.gguf}"
BIN=$ROOT/build/qwenox-engine-win.exe
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
FAILS=()
fail() { FAILS+=("$1"); echo "  FAIL: $1"; }
pass() { echo "  PASS: $1"; }
info() { echo "  INFO: $1"; }
stamp() { echo; echo "=== [$(date +%H:%M:%S)] $*"; }
step() { [[ -z "${ONLY:-}" || " $ONLY " == *" $1 "* ]]; }
for f in "$BIN" "$GGUF" "$MTPG"; do [[ -f "$f" ]] || { echo "缺文件：$f"; echo "CONC_FIX FAIL"; exit 1; }; done
if tasklist 2>/dev/null | grep -qi "qwenox"; then
  echo "检测到 qwenox 进程在运行，请先停掉服务（显存要整块用）"; echo "CONC_FIX FAIL"; exit 1
fi
echo "head $(git log --oneline -1 | cut -c1-70)  dirty: $(git status --short | wc -l)"
GW=$(cygpath -m "$GGUF"); MW=$(cygpath -m "$MTPG")
V2ARGS="$(cygpath -m "$ROOT/models/qwen38-flash-next-v2.hgn") $(cygpath -m "$ROOT/models/qwen38-flash-next-ngram.hgn") $(cygpath -m "$ROOT/models/qwen38-flash-next-mtp.hgn")"

# cdiv PFX "tags that must MATCH" field(greedy|all) [ENV=V ...] : run conc_diverge_win.sh, gate on RESULT lines
cdiv() {
  local pfx=$1 want=$2 field=$3; shift 3
  env "$@" PFX=$pfx bash tools/conc_diverge_win.sh > $L/$pfx.out 2>&1
  local rc=$?
  sed -n '/^run /,/^RESULT/p' logs/${pfx}_summary.txt 2>/dev/null | grep -v '^RESULT' | sed 's/^/    /'
  [[ $rc == 0 ]] || { fail "$pfx：脚本出错（$L/$pfx.out）"; grep -E "FAIL|失败" $L/$pfx.out | head -5 | sed 's/^/    /'; }
  local t
  for t in $want; do
    if grep -q "^RESULT $t .*$field=MATCH" logs/${pfx}_summary.txt 2>/dev/null; then :; else
      fail "$pfx $t $field 不一致"; fi
  done
}

# ---------------------------------------------------------------- v2 4 路
if step v2c4; then
  stamp "v2c4：v2 4 路（默认 ×3、VR_CHECK ×1、VR_RESTORE=0 对照 ×1）"
  cdiv cfix_v2c4 "c1 c2 c3 c4" greedy PAR=4 NREF=1 MODEL_ARGS="$V2ARGS" \
    CONFIGS="base|base|base|QWENOX_VR_CHECK=1|QWENOX_VR_RESTORE=0"
  grep -q "^RESULT c5 greedy=DIFF" logs/cfix_v2c4_summary.txt 2>/dev/null \
    && info "VR_RESTORE=0 对照分叉（与修复前一致）" || info "VR_RESTORE=0 对照这次没分叉（竞态靠时序，可能碰巧没撞上）"
fi
# ---------------------------------------------------------------- v2 8 路
if step v2c8; then
  stamp "v2c8：v2 8 路 ×3"
  cdiv cfix_v2c8 "c1 c2 c3" greedy PAR=8 NREF=1 MODEL_ARGS="$V2ARGS" CONFIGS="base|base|base"
fi
# ---------------------------------------------------------------- SEEDED 采样
if step seeded; then
  stamp "seeded：v2 4 路 ×2，采样流带 SEEDED（连采样一起比）"
  cdiv cfix_seed "c1 c2" all PAR=4 NREF=1 MODEL_ARGS="$V2ARGS" CONC_SEEDED=1 CONFIGS="base|base"
fi
# ---------------------------------------------------------------- GGUF
if step gguf; then
  # GGUF 的输出对 γ 不严格不变（docs/CONCURRENCY.md“与单跑是否一致”）：判定用 OCC_FORCE=4
  # 钉住 γ；最后一项 base（按人数选 γ）只报告。
  stamp "gguf：UD-IQ4_XS 4 路（OCC_FORCE=4 ×2 + base 对照）"
  cdiv cfix_ggc4 "c1 c2" greedy PAR=4 NREF=1 MODEL_ARGS="$GW" GGUF_MTP="$MW" \
    CONFIGS="QWENOX_OCC_FORCE=4|QWENOX_OCC_FORCE=4|base"
  stamp "gguf：UD-IQ4_XS 8 路（OCC_FORCE=4 ×2 + base 对照）"
  cdiv cfix_ggc8 "c1 c2" greedy PAR=8 NREF=1 MODEL_ARGS="$GW" GGUF_MTP="$MW" \
    CONFIGS="QWENOX_OCC_FORCE=4|QWENOX_OCC_FORCE=4|base"
  for p in ggc4 ggc8; do
    grep -q "^RESULT c3 greedy=MATCH" logs/cfix_${p}_summary.txt 2>/dev/null \
      && info "$p base（按人数选 γ）这次也一致" || info "$p base（按人数选 γ）分叉：已知的 GGUF γ 不变性问题，只报告"
  done
fi
# ---------------------------------------------------------------- d2_s2_ids
if step ids; then
  stamp "ids：tools/d2_s2_ids.sh（同 prompt 2/4 并发 + ngram 混跑）"
  bash tools/d2_s2_ids.sh > $L/ids.out 2>&1
  r=$(grep -a "S2 IDS ACCEPT" $L/ids.out | tail -1)
  [[ "$r" == *PASS* ]] && pass "$r" || { fail "d2_s2_ids：${r:-无结果}"; grep -aE "FAIL|≠|mismatch" $L/ids.out | head -5 | sed 's/^/    /'; }
fi
# ---------------------------------------------------------------- vseg 自检
if step vseg; then
  stamp "vseg：v2 批 verify == 逐段 verify"
  for spec in 3,4,5 5,5,5,5,5,5,5,5; do
    tag=vseg_v2_${spec//,/_}
    (
      for e in $(compgen -e | grep '^QWENOX_'); do unset "$e"; done
      export QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1
      export QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1
      export QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 QWENOX_NOWARMUP=1
      export QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1
      export QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1
      export QWENOX_KV_PAGED=1 QWENOX_KVSNAP=0 QWENOX_RCKPT_MAX=0
      export QWENOX_ROPE_FACTOR=1 QWENOX_ROPE_ORIGINAL_CTX=262144
      export QWENOX_ROPE_BETA_FAST=32 QWENOX_ROPE_BETA_SLOW=1 QWENOX_ROPE_ATTN_SCALE=0
      export QWENOX_PARALLEL=8 QWENOX_VSEG_SELFTEST=$spec
      read -r -a a <<< "$V2ARGS"
      cd build && ./qwenox-engine-win.exe "${a[@]}" --maxctx 131072 > "$ROOT/$L/$tag.out" 2> "$ROOT/$L/$tag.err"
    )
    r=$(tr '\r' '\n' < $L/$tag.err | grep -a -m1 "VSEG SELFTEST")
    [[ "$r" == *PASS* ]] && pass "v2 $spec: $r" || { fail "v2 vseg $spec：${r:-无结果}"; tr '\r' '\n' < $L/$tag.err | grep -a "vseg-selftest" | head -5 | sed 's/^/    /'; }
  done
  stamp "vseg：GGUF（tools/gguf_batch_verify_win.sh ONLY=vseg）"
  ONLY=vseg bash tools/gguf_batch_verify_win.sh > $L/vseg_gguf.out 2>&1
  grep -aE "PASS:|FAIL:" $L/vseg_gguf.out | sed 's/^/  /'
  grep -aq "GGUF_BATCH PASS" $L/vseg_gguf.out || fail "GGUF vseg 自检"
fi
# ---------------------------------------------------------------- 吞吐（只报告）
if step bench; then
  stamp "bench：API 吞吐 C4/C8（v2 默认 / v2 VR_RESTORE=0 / GGUF 默认）"
  CFGS=v2 LEVELS=4,8 bash tools/gguf_serve_bench_win.sh > $L/bench_v2_on.out 2>&1
  cp -f logs/gsb/bench_v2.txt $L/bench_v2_on.txt 2>/dev/null
  BENCH_ENV=QWENOX_VR_RESTORE=0 CFGS=v2 LEVELS=4,8 bash tools/gguf_serve_bench_win.sh > $L/bench_v2_off.out 2>&1
  cp -f logs/gsb/bench_v2.txt $L/bench_v2_off.txt 2>/dev/null
  CFGS=gguf_new LEVELS=4,8 bash tools/gguf_serve_bench_win.sh > $L/bench_gguf_on.out 2>&1
  cp -f logs/gsb/bench_gguf_new.txt $L/bench_gguf_on.txt 2>/dev/null
  for c in v2_on v2_off gguf_on; do
    echo "  --- $c"; tail -8 $L/bench_$c.txt 2>/dev/null | sed 's/^/    /'
  done
  info "此前参考：v2 C4/C8 83.2/97.0，GGUF 70.7/82.4 tok/s"
fi

echo
if [[ ${#FAILS[@]} == 0 ]]; then echo "CONC_FIX PASS"
else printf '失败项：\n'; printf '  - %s\n' "${FAILS[@]}"; echo "CONC_FIX FAIL"; fi
