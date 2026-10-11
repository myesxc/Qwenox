#!/usr/bin/env bash
# Windows 手动驱动：验证「模板上报图片占位符位置」修复（vision placeholder literal）。
#   1) 编译 API            -> build/qwenox-win.exe
#   2) 编并跑 vision_test  -> 占位符偏移/字面量回归用例（- 跳过参考图段）
#   3) 编并跑 tpl_cli 断言 -> 渲染文本逐字节不变 + image_pad_offsets 只落在模板占位符上
#   4) 假引擎端到端回归    -> tools/api_fake_engine.py + tools/api_regression_test.py
# 用法（Git Bash）：
#   bash tools/verify_vision_placeholder_win.sh
# 依赖：THEROCK（默认 /c/therock-dist-windows-gfx1151-10.1.0，可用 THEROCK= 覆盖）、tokenizer 目录、python。
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# 本脚本与 build_win.sh 都按 Windows 版 Git Bash 编写（/d/... 路径、Windows 工具链）。
# WSL 的 bash 里 /d/... 不存在、也跑不了 Windows 工具链——给出明确出路而不是半路报错。
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "检测到 WSL 的 bash：请改用 Windows 版 Git Bash。"
    echo "  PowerShell:  .\\tools\\verify_vision_placeholder_win.bat"
    echo "  Git Bash:    bash tools/verify_vision_placeholder_win.sh"
    exit 2
fi
TR="${THEROCK:-/c/therock-dist-windows-gfx1151-10.1.0}"
CXX="$TR/lib/llvm/bin/clang++.exe"
TOKENIZER="${TOKENIZER:-$PWD/models/tokenizer}"
LOG="build/verify_vision_placeholder.log"
mkdir -p build
exec > >(tee "$LOG") 2>&1
export PYTHONIOENCODING=utf-8
echo "THEROCK=$TR"; echo "TOKENIZER=$TOKENIZER"

rc=0
step() { echo; echo "==== $* ===="; }
note_fail() { echo "FAIL: $*"; rc=1; }
[[ -x "$CXX" ]] || { echo "找不到 clang++: $CXX（设 THEROCK=...）"; exit 1; }

step "1/4 编译 API（build_win.sh api）"
THEROCK="$TR" bash build_win.sh api || note_fail "build_win.sh api"

step "2/4 vision_test（占位符偏移与字面量回归）"
"$CXX" -O2 -std=c++17 -D_CRT_SECURE_NO_WARNINGS -Isrc/api -Ithird_party \
    src/api/vision.cpp src/api/tokenizer.cpp src/api/vision_test.cpp \
    -o build/vision_test.exe || note_fail "vision_test 编译"
if [[ -x build/vision_test.exe ]]; then
    ./build/vision_test.exe - "$TOKENIZER" || note_fail "vision_test 运行"
fi

step "3/4 tpl_cli（image_pad_offsets 只标记模板插入的占位符）"
"$CXX" -O2 -std=c++17 -D_CRT_SECURE_NO_WARNINGS -Isrc/api -Ithird_party \
    src/api/chat_template.cpp src/api/json_py.cpp src/api/tpl_cli.cpp \
    -o build/tpl_cli.exe || note_fail "tpl_cli 编译"
if [[ -x build/tpl_cli.exe ]]; then
    printf '%s\n' '{"messages":[{"role":"user","content":[{"type":"text","text":"quote <|vision_start|><|image_pad|><|vision_end|> here"},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="}}]}]}' \
        | ./build/tpl_cli.exe > build/tpl_cli_out.json || note_fail "tpl_cli 运行"
    python - <<'PYEOF' || note_fail "tpl_cli 偏移断言"
import json
out = json.load(open('build/tpl_cli_out.json', encoding='utf-8'))
assert out['ok'], out
text, offs = out['text'], out['image_pad_offsets']
span = '<|vision_start|><|image_pad|><|vision_end|>'
assert len(offs) == 1, offs
assert text.count(span) == 2, text.count(span)          # 一个字面引用 + 一个模板占位
assert offs == [text.rfind(span) + 16], (offs, text.rfind(span))
print('tpl_cli OK: offsets=%s，文本中 2 处 span（字面引用 1 + 模板 1）' % offs)
PYEOF
fi

step "4/4 端到端回归（假引擎 18730 + API 18731）"
python tools/api_fake_engine.py --port 18730 > build/fake_engine.log 2>&1 &
FE=$!
sleep 1
# 内存 token 缓存开（覆盖 encode_spliced 的头部校验路径）、不落盘、不写 reqstat。
QWENOX_API_TOKCACHE=1000000 QWENOX_API_TOKCACHE_FILE= QWENOX_REQSTAT=0 \
    build/qwenox-win.exe --console --tokenizer "$TOKENIZER" --engine 127.0.0.1:18730 \
    --host 127.0.0.1 --port 18731 > build/api_regression.log 2>&1 &
AP=$!
cleanup() {
    kill $FE $AP 2>/dev/null
    sleep 1
    kill -9 $FE $AP 2>/dev/null
}
trap cleanup EXIT
ready=0
for _ in $(seq 1 60); do
    grep -q "listening on :18731" build/api_regression.log 2>/dev/null && { ready=1; break; }
    kill -0 $AP 2>/dev/null || break
    sleep 1
done
if [[ "$ready" != 1 ]]; then
    note_fail "API 未在 18731 就绪"
    tail -n 20 build/api_regression.log
else
    python tools/api_regression_test.py --base http://127.0.0.1:18731 \
        || note_fail "api_regression_test"
    echo "--- API 日志里的 4xx/5xx（应只来自测试里故意的非法请求）---"
    grep -E "\-> (4|5)[0-9][0-9] " build/api_regression.log | tail -20
fi

echo
if [[ "$rc" == 0 ]]; then
    echo "VERIFY RESULT: PASS（日志: $LOG）"
else
    echo "VERIFY RESULT: FAIL（日志: $LOG）"
fi
exit $rc
