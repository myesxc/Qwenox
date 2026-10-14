#!/usr/bin/env python3
"""Compile and exercise the production admission code with a host-only model."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default=os.environ.get("CXX"))
    parser.add_argument("--tsan", action="store_true")
    args = parser.parse_args()
    compiler = args.cxx or shutil.which("clang++") or shutil.which("g++")
    if not compiler and os.name == "nt":
        candidate = Path("C:/Program Files/LLVM/bin/clang++.exe")
        if candidate.is_file():
            compiler = str(candidate)
    if not compiler:
        parser.error("a C++17 compiler is required; pass --cxx PATH")
    if args.tsan and os.name == "nt":
        parser.error("--tsan requires a compiler with ThreadSanitizer on Linux")

    root = Path(__file__).resolve().parent.parent
    source = (root / "src/gpu/parts/51_host_cfg.inc").read_text(encoding="utf-8")
    start = source.index("struct Turnstile {")
    end = source.index("// GpuModel::yield_fn:", start)
    build = root / "build"
    build.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kv-admission-", dir=build) as directory:
        work = Path(directory).resolve()
        work.relative_to(build.resolve())
        (work / "kv_admission_serve.inc").write_text(source[start:end], encoding="utf-8")
        binary = work / ("kv_admission_test.exe" if os.name == "nt" else "kv_admission_test")
        command = [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                   # The slice is a partial view of 51_host_cfg.inc: helpers the
                   # test doesn't drive must not fail the build.
                   "-Wno-unused-function",
                   "-D_CRT_SECURE_NO_WARNINGS", "-I", str(work),
                   str(root / "tools/kv_admission_test.cpp"), "-o", str(binary)]
        if os.name != "nt":
            command.append("-pthread")
        if args.tsan:
            command.extend(["-fsanitize=thread", "-g", "-O1"])
        subprocess.run(command, check=True, timeout=120)
        result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
        print(result.stdout, end="", flush=True)
        if result.returncode:
            print(result.stderr, end="", flush=True)
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
