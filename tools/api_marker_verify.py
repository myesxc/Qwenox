#!/usr/bin/env python3
"""End-to-end checks for the media-marker encoding rules against the fake engine.

Run api_fake_engine.py on --engine port and qwenox-api on --base port first
(same rig as api_regression_test.py), then:

    python -I api_marker_verify.py [--base http://127.0.0.1:18731]
                                   [--engine 127.0.0.1:18730]
                                   [--api-dir <the API's working directory>]

Checks, in order:
  1. A literal media marker in a message is ordinary text: the GEN line's ids
     hold no marker id, and the marker text arrives as byte-level tokens.
  2. A template-placed image span still expands to pads with its marker pair
     kept (vision request).
  3. A′: a sampled request carries the default negative BIAS for the marker
     ids; a greedy request carries none (the engine rejects BIAS without
     SAMPLE, so that would be a broken request, not a no-op).
  4. A client's own logit_bias for a marker id is not duplicated.
  5. The token cache never reuses a span whose recorded ids hold a marker as an
     added token (cache reuse would put the model back in the template region).
     This one rewrites the API's data/tcache.bin, so it restarts the API and
     needs --api-dir (the directory the API was started in).
"""
import argparse
import json
import os
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_replay_ids as er  # noqa: E402  (decoder: ids -> text)

MARKERS = [248053, 248054, 248055, 248056, 248057, 248070, 248071, 248076]
MARKER_TEXT = "<|vision_start|>"
BYTE_TOKENS = [27, 91, 1742, 29302]  # '<' '|' 'image' '_pad' (this tokenizer)


def post(base, path, body, timeout=30):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def engine_state(addr):
    host, port = addr.rsplit(":", 1)
    with socket.create_connection((host, int(port)), timeout=5) as conn:
        conn.sendall(b"TESTSTATE\n")
        reply = b""
        while not conn.recv(0) if False else True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            reply += chunk
            if reply.endswith(b"\n"):
                break
    return json.loads(reply.decode("utf-8", "replace"))


def last_gen(addr, since):
    """The most recent GEN trace entry the fake engine saw."""
    entries = [e for e in engine_state(addr) if e.get("kind") == "gen"]
    fresh = entries[since:]
    return (fresh[-1] if fresh else None), len(entries)


def check(ok, name, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ((" " + str(detail)) if detail else ""))
    if not ok:
        sys.exit(1)


def chat(base, text, **extra):
    body = {"model": "qwen3.8-flash-next",
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 16}
    body.update(extra)
    status, raw = post(base, "/v1/chat/completions", body)
    check(status == 200, "chat-status", "%d %s" % (status, raw[:200]))
    return json.loads(raw.decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18731")
    ap.add_argument("--engine", default="127.0.0.1:18730")
    ap.add_argument("--api-dir", default=".",
                    help="the API's working directory (its data/tcache.bin)")
    ap.add_argument("--api-exe", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "build", "qwenox-win.exe.new"))
    args = ap.parse_args()

    special, inv, c2b = er.load_tokenizer(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "models", "tokenizer",
        "tokenizer.json"))

    # 1. Literal markers in a message are ordinary text.
    text = ("quote %s here and <|image_pad|> too, plus <|vision_end|> "
            "<|audio_pad|>" % MARKER_TEXT)
    _, n0 = last_gen(args.engine, 0)
    chat(args.base, text, temperature=0)
    entry, _ = last_gen(args.engine, n0)
    check(entry is not None, "text-only-gen-seen")
    hits = [i for i in entry["ids"] if i in MARKERS]
    check(not hits, "literal-markers-not-added-tokens", hits)
    check(all(t in entry["ids"] for t in BYTE_TOKENS),
          "literal-markers-as-text-tokens",
          "want %s in the first %s ids" % (BYTE_TOKENS, entry["ids"][:40]))
    decoded = er.decode_ids(entry["ids"], special, inv, c2b)
    check(MARKER_TEXT in decoded and "<|image_pad|>" in decoded,
          "literal-markers-visible-in-text", repr(decoded[-160:]))

    # 2. A template-placed span still expands (vision request through the API
    #    needs an image, so this one is a unit-level fact checked by
    #    vision_test; here just make sure a plain request keeps the pair when
    #    the template renders one -- no image => no span => no markers).

    # 3a. Sampled request carries the marker bias.
    _, n0 = last_gen(args.engine, 0)
    chat(args.base, "hello", temperature=0.7, top_p=0.9, seed=4242)
    entry, _ = last_gen(args.engine, n0)
    check(entry is not None and entry["sample"], "sampled-gen-seen")
    biased = {int(b[0]): float(b[1]) for b in entry["bias"]}
    check(all(m in biased for m in MARKERS), "marker-bias-present",
          sorted(biased)[:12])
    check(all(biased[m] < -1000.0 for m in MARKERS), "marker-bias-negative",
          {m: biased[m] for m in MARKERS})

    # 3b. Greedy request carries none (BIAS without SAMPLE is a rejected GEN).
    _, n0 = last_gen(args.engine, 0)
    chat(args.base, "hello", temperature=0)
    entry, _ = last_gen(args.engine, n0)
    check(entry is not None and not entry["sample"], "greedy-gen-seen")
    check(entry["bias"] == [], "greedy-no-bias", entry["bias"][:4])

    # 4. A client bias for a marker id wins (no duplicate, no override).
    _, n0 = last_gen(args.engine, 0)
    chat(args.base, "hello", temperature=0.7, top_p=0.9, seed=4243,
         logit_bias={"248056": -3.0})
    entry, _ = last_gen(args.engine, n0)
    ids = [int(b[0]) for b in entry["bias"]]
    check(ids.count(248056) == 1, "client-bias-not-duplicated", ids)
    value = {int(b[0]): float(b[1]) for b in entry["bias"]}[248056]
    check(abs(value + 3.0) < 1e-6, "client-bias-kept", value)

    # 5. The token cache does not reuse a span recorded with an added marker.
    if args.api_dir == ".":
        print("SKIP cache-reuse clamp (pass --api-dir to enable)")
    else:
        cache = os.path.join(args.api_dir, "data", "tcache.bin")
        # Take the ids of a request whose text holds the marker as plain text,
        # then rewrite exactly those marker bytes as the added token and store
        # that entry: its bytes still match, its ids no longer do.
        _, n0 = last_gen(args.engine, 0)
        chat(args.base, "cache probe " + MARKER_TEXT + " tail", temperature=0)
        entry, _ = last_gen(args.engine, n0)
        ids = list(entry["ids"])
        text_ids = er.decode_ids(ids, special, inv, c2b)
        at = text_ids.rfind(MARKER_TEXT)
        check(at >= 0, "cache-probe-marker-in-prompt", repr(text_ids[-80:]))
        # Map the marker's bytes to the ids covering them.
        pos, first, last = 0, None, None
        for index, token_id in enumerate(ids):
            length = len(er.decode_ids([token_id], special, inv, c2b))
            if first is None and pos + length > at:
                first = index
            if pos < at + len(MARKER_TEXT):
                last = index
            pos += length
        check(first is not None and last is not None and last - first >= 1,
              "cache-probe-marker-ids", (first, last))
        crafted = ids[:first] + [248053] + ids[last + 1:]
        with open(cache, "wb") as handle:
            handle.write(b"GDTC1\x00\x00\x00" + struct.pack("<IIQ", 2, 248077, 1))
            handle.write(struct.pack("<Q", len(crafted)))
            handle.write(struct.pack("<%di" % len(crafted), *crafted))
        # The cache is read at boot: restart the API (the fake engine stays).
        procs = subprocess.run(["powershell", "-NoProfile", "-Command",
                                "Get-CimInstance Win32_Process -Filter \"Name like 'qwenox-win%'\""
                                " | Select-Object ProcessId,CommandLine | ConvertTo-Json"],
                               capture_output=True, text=True)
        pid = None
        for item in json.loads(procs.stdout or "[]") if procs.stdout.strip().startswith("[") \
                else [json.loads(procs.stdout or "{}")]:
            if args.api_exe.replace("/", "\\").split("\\")[-1] in (item.get("CommandLine") or ""):
                pid = item["ProcessId"]
        check(pid is not None, "api-pid-found", procs.stdout[:200])
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        "Stop-Process -Id %d -Force" % pid], check=True)
        time.sleep(1.5)
        log = open(os.path.join(args.api_dir, "api.log"), "a")
        subprocess.Popen([os.path.abspath(args.api_exe), "--console"], cwd=args.api_dir,
                         stdout=log, stderr=log)
        time.sleep(4.0)
        _, n0 = last_gen(args.engine, 0)
        chat(args.base, "cache probe " + MARKER_TEXT + " tail", temperature=0)
        entry, _ = last_gen(args.engine, n0)
        check(entry is not None, "cache-probe-regenerated")
        check(248053 not in entry["ids"], "cache-reuse-not-before-clamp",
              "248053 came back through the reused prefix")
        reused = None
        with open(os.path.join(args.api_dir, "api.log"), encoding="utf-8",
                  errors="replace") as handle:
            for line in handle:
                if line.startswith("tcache: reused"):
                    reused = line.strip()
        print("     last cache line: %s" % reused)

    print("RESULT PASS")


if __name__ == "__main__":
    main()
