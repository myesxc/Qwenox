#!/usr/bin/env python3
# conc_bench.py — OpenAI 兼容接口并发吞吐测试（MTP 友好场景）。
#
# 用法：
#   python tools/conc_bench.py                          # 默认 1/2/4/8 并发各跑一轮
#   python tools/conc_bench.py --levels 2,4,8 --rpw 2   # 每 worker 2 个请求
#   python tools/conc_bench.py --base-url http://127.0.0.1:8731
#
# 注意：8 并发需要引擎以 PARALLEL=8 启动（service.conf 默认 PARALLEL=2）。
#
# 口径说明：
#   客户端：wall time + usage.completion_tokens。
#   服务端：/reqstat/tail 记录（req_seq 递增），按 level 前后的 req_seq 水位切分，
#   decode_tps = output_tokens / decode_ms；acceptance = (commit-rounds)/proposed。
# 纯 stdlib，无第三方依赖。

import argparse
import json
import sys
import threading
import time
import urllib.request
import urllib.error

# 8 个 MTP 友好任务：算法实现类，输出可预测、重复结构多，接受率高。
# 默认贪心（temperature=0）+ 长输出，让 decode 阶段主导耗时；
# --temperature >0 测采样路径（API 不传 temperature 时默认就是采样）。
TASKS = [
    ("Python",     "quicksort"),
    ("C",          "quicksort"),
    ("Java",       "quicksort"),
    ("Go",         "quicksort"),
    ("JavaScript", "quicksort"),
    ("Rust",       "quicksort"),
    ("C++",        "quicksort"),
    ("TypeScript", "quicksort"),
]


def make_prompt(lang: str, algo: str) -> str:
    return (
        f"Write a complete, production-quality {algo} implementation in {lang}. "
        "Requirements: include detailed comments explaining each step, handle edge "
        "cases (empty input, single element, duplicates), and include a small test "
        "harness (main function or equivalent) that sorts several example arrays "
        "and prints the results. Output only the code, no explanations."
    )


class Api:
    def __init__(self, base_url: str, timeout: float, temperature: float = 0.0,
                 top_p=None, top_k=None):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature, self.top_p, self.top_k = temperature, top_p, top_k

    def _req(self, method: str, path: str, body=None):
        url = self.base + path
        data = None
        headers = {}
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def get(self, path):
        return self._req("GET", path)

    def chat(self, model, prompt, max_tokens):
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self.temperature > 0:
            if self.top_p is not None:
                body["top_p"] = self.top_p
            if self.top_k is not None:
                body["top_k"] = self.top_k
        return self._req("POST", "/v1/chat/completions", body)

    def model_id(self):
        try:
            j = self.get("/v1/models")
            return j["data"][0]["id"]
        except Exception:
            return "default"

    def reqstat_watermark(self):
        """当前最大 req_seq；无记录时返回 0。"""
        try:
            j = self.get("/reqstat/tail?n=1")
            recs = j.get("records", [])
            return recs[-1]["req_seq"] if recs else 0
        except Exception:
            return None

    def reqstat_since(self, watermark):
        """req_seq > watermark 的全部记录（上限 1000 条）。"""
        if watermark is None:
            return None
        try:
            j = self.get("/reqstat/tail?n=1000")
            return [r for r in j.get("records", []) if r["req_seq"] > watermark]
        except Exception:
            return None


def worker(api, model, tasks, max_tokens, barrier, results, wid):
    """每个 worker 串行跑 tasks；barrier 保证所有 worker 同时出发。"""
    barrier.wait()
    for lang, algo in tasks:
        t0 = time.perf_counter()
        err = None
        completion = 0
        try:
            resp = api.chat(model, make_prompt(lang, algo), max_tokens)
            usage = resp.get("usage") or {}
            completion = usage.get("completion_tokens", 0)
        except Exception as e:  # noqa: BLE001
            err = str(e)
        t1 = time.perf_counter()
        results.append({
            "worker": wid, "task": f"{lang}/{algo}", "t0": t0, "t1": t1,
            "wall_s": t1 - t0, "completion_tokens": completion, "error": err,
        })


def run_level(api, model, level, rpw, max_tokens):
    wm = api.reqstat_watermark()
    results = []
    barrier = threading.Barrier(level)
    threads = []
    for w in range(level):
        tasks = [TASKS[(w * rpw + r) % len(TASKS)] for r in range(rpw)]
        t = threading.Thread(target=worker,
                             args=(api, model, tasks, max_tokens, barrier,
                                   results, w))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    results.sort(key=lambda r: r["t0"])
    recs = api.reqstat_since(wm)
    return results, recs


def summarize(level, results, recs):
    ok = [r for r in results if not r["error"]]
    errs = [r for r in results if r["error"]]
    total_tokens = sum(r["completion_tokens"] for r in ok)
    t_start = min(r["t0"] for r in ok) if ok else 0
    t_end = max(r["t1"] for r in ok) if ok else 0
    span = t_end - t_start if ok else 0
    agg_tps = total_tokens / span if span > 0 else 0
    per_client = [r["completion_tokens"] / r["wall_s"]
                  for r in ok if r["wall_s"] > 0 and r["completion_tokens"] > 0]
    avg_client_tps = sum(per_client) / len(per_client) if per_client else 0

    row = {
        "level": level, "requests": len(ok), "errors": len(errs),
        "total_tokens": total_tokens, "span_s": span,
        "agg_tps": agg_tps, "avg_client_tps": avg_client_tps,
    }

    if recs:
        dps = [r["output_tokens"] / (r["decode_ms"] / 1000.0)
               for r in recs if r.get("decode_ms") and r.get("output_tokens")]
        accs = [r["acceptance"] for r in recs
                if r.get("acceptance") is not None]
        ttfts = [r["ttft_ms"] for r in recs if r.get("ttft_ms")]
        drafters = {}
        for r in recs:
            drafters[r.get("drafter", "?")] = drafters.get(r.get("drafter", "?"), 0) + 1
        row["server_records"] = len(recs)
        row["server_decode_tps"] = (sum(dps) / len(dps)) if dps else None
        row["acceptance"] = (sum(accs) / len(accs)) if accs else None
        row["ttft_ms"] = (sum(ttfts) / len(ttfts)) if ttfts else None
        row["drafters"] = drafters
    if errs:
        row["error_sample"] = errs[0]["error"][:200]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8731")
    ap.add_argument("--levels", default="1,2,4,8",
                    help="并发档位，逗号分隔（默认 1,2,4,8；1 是基线）")
    ap.add_argument("--rpw", type=int, default=2,
                    help="每个 worker 的请求数（默认 2）")
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="采样温度（默认 0 = 贪心；>0 走引擎采样路径）")
    ap.add_argument("--top-p", type=float, default=None,
                    help="仅 temperature>0 时发送（默认不发，用模型默认）")
    ap.add_argument("--top-k", type=int, default=None,
                    help="仅 temperature>0 时发送（默认不发，用模型默认）")
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    api = Api(args.base_url, args.timeout, args.temperature, args.top_p, args.top_k)

    print(f"目标: {args.base_url}")
    print(f"采样: temperature={args.temperature}"
          + (f" top_p={args.top_p}" if args.top_p is not None else "")
          + (f" top_k={args.top_k}" if args.top_k is not None else ""))
    print("提醒: 8 并发需要引擎以 PARALLEL=8 启动（service.conf 默认 PARALLEL=2）")
    try:
        model = api.model_id()
    except Exception as e:
        print(f"连不上 API: {e}")
        sys.exit(1)
    print(f"模型: {model}")

    if not args.no_warmup:
        print("预热中 ...", flush=True)
        api.chat(model, make_prompt(*TASKS[0]), 256)

    rows = []
    for lv in levels:
        print(f"\n=== 并发 {lv} ({lv * args.rpw} 个请求) ===", flush=True)
        results, recs = run_level(api, model, lv, args.rpw, args.max_tokens)
        row = summarize(lv, results, recs)
        rows.append(row)
        for r in sorted(results, key=lambda x: (x["worker"], x["t0"])):
            if r["error"]:
                print(f"  w{r['worker']} {r['task']:22s} ERROR {r['error'][:120]}")
            else:
                tps = (r["completion_tokens"] / r["wall_s"]) if r["wall_s"] else 0
                print(f"  w{r['worker']} {r['task']:22s} "
                      f"{r['completion_tokens']:5d} tok  {r['wall_s']:6.1f}s  "
                      f"{tps:6.1f} tok/s")
        print(f"  -> 聚合 {row['agg_tps']:.1f} tok/s | "
              f"单客户端均值 {row['avg_client_tps']:.1f} tok/s | "
              f"span {row['span_s']:.1f}s", flush=True)
        if row.get("server_decode_tps"):
            print(f"  -> 服务端 decode {row['server_decode_tps']:.1f} tok/s | "
                  f"接受率 {row['acceptance']:.3f} | "
                  f"TTFT {row['ttft_ms']:.0f}ms | drafter {row['drafters']}",
                  flush=True)

    print("\n=== 汇总 ===")
    hdr = f"{'并发':>4} {'请求':>4} {'聚合tok/s':>10} {'单客户端tok/s':>13} " \
          f"{'服务端decode':>12} {'接受率':>7} {'TTFTms':>8}"
    print(hdr)
    base = rows[0]["agg_tps"] if rows and rows[0]["agg_tps"] else None
    for row in rows:
        sd = f"{row['server_decode_tps']:.1f}" if row.get("server_decode_tps") else "-"
        acc = f"{row['acceptance']:.3f}" if row.get("acceptance") is not None else "-"
        ttft = f"{row['ttft_ms']:.0f}" if row.get("ttft_ms") else "-"
        print(f"{row['level']:>4} {row['requests']:>4} "
              f"{row['agg_tps']:>10.1f} {row['avg_client_tps']:>13.1f} "
              f"{sd:>12} {acc:>7} {ttft:>8}")
    if base and len(rows) > 1:
        print("\n聚合吞吐相对基线(并发%d):" % rows[0]["level"])
        for row in rows:
            print(f"  并发 {row['level']}: {row['agg_tps'] / base:.2f}x")


if __name__ == "__main__":
    main()
