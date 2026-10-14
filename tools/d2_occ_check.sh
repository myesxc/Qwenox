#!/usr/bin/env bash
# d2_occ_check.sh — D2 阶段 3（按会合路数选 γ，Model::OccGamma）验收，结尾 PASS/FAIL。
#   T1 单序列：QWENOX_OCC_FORCE=0（每轮 γ=0：只 verify cur 一行）的 ids 必须是
#      γ=1 输出的前缀（γ=0 每轮只出 1 个 token，不会像 γ>0 的末轮那样多带一个）
#   T2 服务端 greedy，策略默认开：d2_s2_ids 并发 ≡ 单跑（γ 随实时人数变化）
#   T3 服务端采样 0.7，策略默认开（采样按 nslots 固定 B）：并发 ≡ 单跑
#   T4 服务端 greedy 强制 γ=0：并发 ≡ 单跑，且 base md5 ≡ T2 的 base（γ 不变性）
#   T5 服务端采样强制 γ=0：并发 ≡ 单跑
# 用法：bash tools/d2_occ_check.sh      # 约 11 分钟，需先 bash build_win.sh engine
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
L=logs/d2_occ_check.log
mkdir -p logs
ok=1
port_free() {
  for _ in $(seq 1 60); do
    netstat -an | grep -qE "[:.]18730\s+.*LISTENING" || break
    sleep 2
  done
  sleep 5  # 显存回收
}
ids() { grep -a '^ids:' "$1" | tail -1 | cut -d' ' -f8194-; }
base_md5() { grep -a "base_d$2 " "$1" | head -1 | grep -o 'md5=[0-9a-f]*'; }

echo "== d2_occ_check $(date '+%F %T')" | tee "$L"
echo "-- T1 单序列 γ=0 前缀对拍" | tee -a "$L"
GAMMAS="1" bash tools/d2_s1b_gamma.sh >> "$L" 2>&1
cp logs/d2s1b_g1.log logs/d2occ_g1.log
QWENOX_OCC_FORCE=0 GAMMAS="4" bash tools/d2_s1b_gamma.sh >> "$L" 2>&1
cp logs/d2s1b_g4.log logs/d2occ_f0.log
a=$(ids logs/d2occ_g1.log)
b=$(ids logs/d2occ_f0.log)
if [[ -n "$b" && "$a" == "$b"* ]]; then
  echo "  T1 PASS（γ=0 $(echo $b | wc -w) 个 id 为 γ=1 前缀）" | tee -a "$L"
else
  echo "  T1 FAIL" | tee -a "$L"; ok=0
fi

run_ids() {  # $1=标签 其余=env
  local tag=$1; shift
  port_free
  echo "-- $tag" | tee -a "$L"
  env "$@" bash tools/d2_s2_ids.sh > "logs/d2occ_$tag.log" 2>&1
  if grep -aq 'S2 IDS ACCEPT: PASS' "logs/d2occ_$tag.log"; then
    echo "  $tag PASS $(grep -a 'occ-gamma' logs/d2s2_engine.log | tail -1)" | tee -a "$L"
  else
    echo "  $tag FAIL（见 logs/d2occ_$tag.log）" | tee -a "$L"; ok=0
  fi
}
run_ids T2 QWENOX_OCC=1
run_ids T3 QWENOX_OCC=1 D2_SAMPLE=0.7
run_ids T4 QWENOX_OCC_FORCE=0
run_ids T5 QWENOX_OCC_FORCE=0 D2_SAMPLE=0.7
m2=$(base_md5 logs/d2occ_T2.log 4)
m4=$(base_md5 logs/d2occ_T4.log 4)
if [[ -n "$m2" && "$m2" == "$m4" ]]; then
  echo "  T4 γ=0 base ≡ T2 base（$m2）" | tee -a "$L"
else
  echo "  T4 base 与 T2 不同：$m4 vs $m2 FAIL" | tee -a "$L"; ok=0
fi
echo "D2 OCC CHECK: $([ $ok = 1 ] && echo PASS || echo FAIL)" | tee -a "$L"
