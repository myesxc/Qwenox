#!/bin/bash
# gguf_vision_mem_win.sh — Windows：UD-IQ4_XS GGUF vs v2 的主机内存曲线 + 多模态一键检查（约 12 分钟）
#   bash tools/gguf_vision_mem_win.sh                    # GGUF + v2
#   CFGS=gguf SKIP_FETCH=1 bash tools/gguf_vision_mem_win.sh
# A 若 Models 下没有 mmproj-BF16.gguf，尝试从本机 Linux 盘（WSL 挂载，需要管理员）拷一份；
# B 有 mmproj 时：同一份随机 patches 分别用 vision.hgn / mmproj 跑视觉塔前向，逐层 dump 逐字节对比；
# C 每种配置用临时 conf（端口 18730/18731，256K、8 路、MTP、kvsnap、视觉塔）起服务，发一张图片（media/qwen-charactor.jpg）
#   和一条 ~42K token 的长文本，tools/win_memsampler.ps1 每 2 s 记录引擎工作集 / 系统可用 / standby / GPU 专用与共享。
# 日志 logs/mm/。结尾 MM_DONE；判读看各阶段的 engine WS 与图片回复内容。
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT=$PWD; O=$ROOT/logs/mm; mkdir -p $O; rm -f $O/*.csv
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
GGUF="${GGUF:-/c/Users/Mark/Models/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}"
MTPG="${MTPG:-$(dirname "$GGUF")/mtp-Qwen3.8-Flash-Next-Q8_0.gguf}"
MMPROJ="${MMPROJ:-$(dirname "$GGUF")/mmproj-BF16.gguf}"
VHGN=$ROOT/models/qwen38-flash-next-vision.hgn
bash build_win.sh engine > $O/build.log 2>&1 && echo "build ok" || { echo BUILD FAIL; exit 1; }
echo "== A. mmproj-BF16.gguf"
if [[ ! -f $MMPROJ && -z "${SKIP_FETCH:-}" ]]; then
  cat > /c/Users/Mark/AppData/Local/Temp/mmfind.sh <<'X'
set -u
for d in $(lsblk -bnpo NAME,SIZE,TYPE | awk '$3=="part" && $2>3000000000000 {print $1}'); do
  mkdir -p /mnt/lx3; mount -t ext4 -o ro,noload "$d" /mnt/lx3 2>/dev/null || continue
  f=$(find /mnt/lx3 -maxdepth 8 -name 'mmproj*BF16*.gguf' -size +100M 2>/dev/null | grep -i flash | head -1)
  [ -z "$f" ] && f=$(find /mnt/lx3 -maxdepth 8 -name 'mmproj*.gguf' -size +100M 2>/dev/null | head -1)
  echo "found: $f"
  [ -n "$f" ] && cp "$f" /mnt/c/Users/Mark/Models/mmproj-BF16.gguf && echo copied
  umount /mnt/lx3
done
X
  MSYS_NO_PATHCONV=1 wsl.exe --mount '\\.\PHYSICALDRIVE1' --bare 2>&1 | tr -d '\0' | head -2
  MSYS_NO_PATHCONV=1 wsl.exe -u root -e bash /mnt/c/Users/Mark/AppData/Local/Temp/mmfind.sh 2>&1 | tr -d '\0'
  MSYS_NO_PATHCONV=1 wsl.exe --unmount '\\.\PHYSICALDRIVE1' 2>&1 | tr -d '\0' | head -1
fi
ls -la $MMPROJ 2>/dev/null || echo "no mmproj: GGUF run uses vision.hgn"
[[ -f $MMPROJ ]] && GVIS=$MMPROJ || GVIS=$VHGN

echo "== B. vision tower forward: vision.hgn vs mmproj (--vision-test, bit-exact dumps)"
if [[ -f $MMPROJ ]]; then
  python - $O/patches.npy <<'P'
import sys, array, random
random.seed(1234)
P, C = 256, 1536
a = array.array('f', (random.gauss(0.0, 1.0) for _ in range(P * C)))
hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }" % (P, C)
hdr += ' ' * (63 - (10 + len(hdr)) % 64) + '\n'
with open(sys.argv[1], 'wb') as f:
    f.write(b'\x93NUMPY\x01\x00' + len(hdr).to_bytes(2, 'little') + hdr.encode()); a.tofile(f)
P
  rm -rf $O/vt_hgn* $O/vt_gguf*
  for k in hgn gguf; do
    [[ $k == hgn ]] && V=$VHGN || V=$MMPROJ
    (cd build && ./qwenox-engine-win.exe - --vision-test "$(cygpath -m $O/vt_$k)" --vision-tower "$(cygpath -m $V)" \
       --patches "$(cygpath -m $O/patches.npy)" --mrope-grid 1,16,16 > $O/vt_$k.log 2>&1); echo "  $k rc=$?: $(tr '\r' '\n' < $O/vt_$k.log | grep -E 'vision(-test)?:' | tail -1)"
  done
  n=$(ls $O/vt_hgn.layers 2>/dev/null | wc -l)
  if [[ $n -gt 0 ]] && diff -rq $O/vt_hgn.layers $O/vt_gguf.layers > $O/vt_diff.txt; then echo "  VISION_TOWER PASS: $n dump files byte-identical"; else echo "  VISION_TOWER FAIL ($n files)"; head -5 $O/vt_diff.txt; fi
fi

echo "== C. service: memory profile + image + long text"
python - $ROOT/media/qwen-charactor.jpg $O/req_img.json $O/req_long.json <<'P'
import sys, base64, json, pathlib, glob
img = base64.b64encode(open(sys.argv[1], 'rb').read()).decode()
json.dump({"model": "qwenox", "max_tokens": 160, "temperature": 0, "messages": [{"role": "user", "content": [
  {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img}},
  {"type": "text", "text": "请用两三句话描述这张图片：画的是什么、什么风格、有哪些显著特征。"}]}]},
  open(sys.argv[2], 'w', encoding='utf-8'), ensure_ascii=False)
txt = ''.join(open(f, encoding='utf-8').read() for f in sorted(glob.glob('docs/*.md')))[:90000]
json.dump({"model": "qwenox", "max_tokens": 48, "temperature": 0, "messages": [{"role": "user", "content": txt + "\n\n用一句话说明上面这些文档是关于什么的。"}]},
  open(sys.argv[3], 'w', encoding='utf-8'), ensure_ascii=False)
P
CONF=$ROOT/service.mm-test.conf; API=http://127.0.0.1:18731
kill_test() { powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { (\$_.Name -like 'qwenox*') -and (\$_.CommandLine -match 'mm-test|18730') } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" > /dev/null 2>&1; }
trap 'kill_test; rm -f "$CONF" "$CONF.bak" $O/stop' EXIT
for cfg in ${CFGS:-gguf v2}; do
  echo "--- $cfg"
  for e in $(compgen -e | grep '^QWENOX_'); do unset "$e"; done
  {
    echo 'MODEL_DIR="./models"'; echo 'TOKENIZER_DIR="./models/tokenizer"'
    if [[ $cfg == v2 ]]; then
      echo 'MODEL_FILE="./models/qwen38-flash-next-v2.hgn"'; echo 'NGRAM_FILE="./models/qwen38-flash-next-ngram.hgn"'
      echo 'OVERLAY_FILE=""'; echo 'MTP_FILE="./models/qwen38-flash-next-mtp.hgn"'; echo 'VISION_FILE="./models/qwen38-flash-next-vision.hgn"'
    else
      echo "GGUF_FILE=\"$(cygpath -m $GGUF)\""; echo "GGUF_MTP_FILE=\"$(cygpath -m $MTPG)\""; echo "GGUF_VISION_FILE=\"$(cygpath -m $GVIS)\""
    fi
    echo 'ENGINE_HOST="127.0.0.1"'; echo 'ENGINE_PORT="18730"'; echo 'API_HOST="127.0.0.1"'; echo 'API_PORT="18731"'
    echo 'MAX_CONTEXT="262144"'; echo 'PARALLEL="8"'; echo 'KV_PAGED="1"'; echo 'KVSNAP_MAX_GB="64"'; echo 'RCKPT_MAX="8"'; echo 'MTP_GAMMA="0"'; echo 'MAX_IMAGES="32"'
    [[ $cfg == v2 ]] && echo '# start_win: weights=v2' || echo '# start_win: weights=gguf'
  } > "$CONF"
  rm -f $O/stop; echo idle > $O/phase
  powershell -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w $ROOT/tools/win_memsampler.ps1)" -Csv "$(cygpath -w $O/mem_$cfg.csv)" -PhaseFile "$(cygpath -w $O/phase)" -StopFile "$(cygpath -w $O/stop)" &
  sp=$!; sleep 6
  echo loading > $O/phase
  ./build/qwenox-win.exe --console --service-conf "$(cygpath -w "$CONF")" > $O/api_$cfg.log 2>&1 &
  for i in $(seq 60); do curl -s -m 2 $API/admin/engine/status > /dev/null && break; sleep 1; done
  curl -s -m 10 -X POST $API/admin/engine/start > /dev/null
  for i in $(seq 200); do st=$(curl -s -m 3 $API/admin/engine/status | sed -n 's/.*"state":"\([a-z]*\)".*/\1/p'); [[ $st == running || $st == exited || $st == failed ]] && break; sleep 3; done
  echo "  engine: $st"; echo ready > $O/phase; sleep 12
  echo image > $O/phase
  curl -s -m 300 $API/v1/chat/completions -H 'Content-Type: application/json; charset=utf-8' --data-binary @$O/req_img.json > $O/img_$cfg.json
  echo "  image reply: $(python -c "import json,sys;d=json.load(open(sys.argv[1],encoding='utf-8'));print(d['choices'][0]['message']['content'].replace(chr(10),' ')[-400:], '| usage', d.get('usage'))" $O/img_$cfg.json 2>&1 | cut -c1-600)"
  sleep 6; echo long > $O/phase
  t0=$(date +%s)
  curl -s -m 600 $API/v1/chat/completions -H 'Content-Type: application/json; charset=utf-8' --data-binary @$O/req_long.json > $O/long_$cfg.json
  echo "  long reply ($(( $(date +%s) - t0 ))s): $(python -c "import json,sys;d=json.load(open(sys.argv[1],encoding='utf-8'));print(d['choices'][0]['message']['content'].replace(chr(10),' ')[-200:], '| usage', d.get('usage'))" $O/long_$cfg.json 2>&1 | cut -c1-400)"
  echo after > $O/phase; sleep 20
  elog=$(ls -t logs/engine-api-*.log | head -1); tr '\r' '\n' < $elog | grep -E "^memory\[|^devarena:|^gguf dense: released|^vision|rckpt|^weight arena: [0-9]" | head -12 | sed 's/^/    /'
  curl -s -m 30 -X POST $API/admin/engine/stop > /dev/null; sleep 5; kill_test; sleep 3
  touch $O/stop; wait $sp 2>/dev/null
  python - $O/mem_$cfg.csv <<'P'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
f = lambda r, k: float(r[k].replace(',', ''))
base = rows[0]
print(f"    idle before start: avail {f(base,'avail'):.1f} standby {f(base,'standby'):.1f} GiB")
for ph in ['loading', 'ready', 'image', 'long', 'after']:
    rs = [r for r in rows if r['phase'] == ph]
    if not rs: continue
    mx = lambda k: max(f(r, k) for r in rs)
    last = rs[-1]
    print(f"    {ph:8s} engine WS max {mx('eng_ws'):6.2f} (end {f(last,'eng_ws'):6.2f}) private max {mx('eng_priv'):6.2f} | api WS {f(last,'api_ws'):5.2f} | "
          f"avail min {min(f(r,'avail') for r in rs):5.1f} standby end {f(last,'standby'):5.1f} | GPU ded {f(last,'gpu_ded'):5.1f} shared {f(last,'gpu_shared'):5.2f}")
P
done
echo MM_DONE
