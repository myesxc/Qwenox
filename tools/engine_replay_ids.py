#!/usr/bin/env python3
"""Replay an exact token-id prompt against a running engine (no tokenizer round trip).

The API's token cache (data/tcache.bin) records, for every request it served, the
ids the engine actually saw (prompt + generated).  Those ids are the only exact
record of a prompt, so replaying them isolates engine behaviour from the API's
encoding, the chat template and the tokenizer: if the engine stops at the same
place with the same ids, the encoding/template layer is not the cause.

    python -I engine_replay_ids.py --list
    python -I engine_replay_ids.py --entry-len 41753 --cut-generated 436
    python -I engine_replay_ids.py --entry-len 6538 --cut-generated 366 \
        --ckpt 248058,248069

The API's GEN line carries CKPT hints (the ids it checkpoints RAM state at
mid-generation); --ckpt replays with the same hints, which is what makes the
engine pause at </tool_call>/</think> like a real request.  --tail keeps only
the last N ids (after --cut-generated), for replaying the tail of a long entry
as a standalone prompt.

The tool speaks the engine's line protocol directly:

    GEN <req> <max_tokens> <n_eos> <eos...> <n_ids> <ids...> [drafter]
    ->  T <req> <tok> [logprob]  ...  and finally  D <req> <reason> ...

Use QWENOX_DRAFTER=<0|3|4> on the engine to compare serial (0) against the
speculative paths: the same prompt that dies inside a marker under the default
chain should continue under serial if the speculative path is at fault.

Read-only apart from the generation itself (it occupies the engine's slot like
any other request, so run it when nobody is using the service).
"""
import argparse
import json
import os
import socket
import struct
import sys

EOS = [248046, 248044]


# --------------------------------------------------------------- token cache --
def read_tcache(path):
    raw = open(path, "rb").read()
    if raw[:5] != b"GDTC1":
        raise SystemExit("not a token cache: %s" % path)
    _, vocab, count = struct.unpack_from("<IIQ", raw, 8)
    off, entries = 24, []
    for _ in range(count):
        (n,) = struct.unpack_from("<Q", raw, off)
        off += 8
        entries.append(list(struct.unpack_from("<%di" % n, raw, off)))
        off += 4 * n
    return vocab, entries


# ------------------------------------------------------------------ decoding --
def load_tokenizer(path):
    tk = json.load(open(path, encoding="utf-8"))
    special = {t["id"]: t["content"] for t in tk["added_tokens"]}
    inv = {v: k for k, v in tk["model"]["vocab"].items()}
    # GPT-2 byte <-> unicode table (the vocab's tokens are byte-level BPE).
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    c2b = {chr(c): b for b, c in zip(bs, cs)}
    return special, inv, c2b


def decode_ids(ids, special, inv, c2b):
    out = bytearray()
    for i in ids:
        if i in special:
            out += special[i].encode("utf-8")
            continue
        tok = inv.get(i)
        if tok is None:
            out += ("<%d?>" % i).encode("utf-8")
            continue
        for ch in tok:
            b = c2b.get(ch)
            out += bytes([b]) if b is not None else ch.encode("utf-8")
    return out.decode("utf-8", errors="replace")


def describe(ids, special, inv, c2b):
    parts = []
    for i in ids:
        if i in special:
            parts.append("%d:%s[SP]" % (i, special[i]))
        else:
            parts.append("%d:%r" % (i, decode_ids([i], special, inv, c2b)))
    return " ".join(parts)


# -------------------------------------------------------------------- engine --
def generate(host, port, ids, max_tokens, drafter, timeout, ckpt=()):
    req = 1
    line = "GEN %d %d %d" % (req, max_tokens, len(EOS))
    for e in EOS:
        line += " %d" % e
    line += " %d" % len(ids)
    for i in ids:
        line += " %d" % i
    if ckpt:
        line += " CKPT %d" % len(ckpt)
        for c in ckpt:
            line += " %d" % c
    if drafter >= 0:
        line += " %d" % drafter
    line += "\n"

    sock = socket.create_connection((host, port), timeout=20)
    sock.settimeout(timeout)
    sock.sendall(line.encode())
    buf = b""
    tokens, dline = [], None
    while dline is None:
        chunk = sock.recv(65536)
        if not chunk:
            raise SystemExit("engine closed the connection (no D line)")
        buf += chunk
        while b"\n" in buf:
            raw, buf = buf.split(b"\n", 1)
            parts = raw.decode(errors="replace").split()
            if not parts:
                continue
            if parts[0] == "T" and len(parts) >= 3:
                tokens.append(int(parts[2]))
            elif parts[0] == "D":
                dline = raw.decode(errors="replace")
    sock.close()
    return tokens, dline


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tcache", default="data/tcache.bin")
    ap.add_argument("--tokenizer", default="models/tokenizer/tokenizer.json")
    ap.add_argument("--list", action="store_true", help="list cached entries and exit")
    ap.add_argument("--entry-len", type=int, help="pick the entry with this id count")
    ap.add_argument("--entry-index", type=int, help="pick the entry by index in the file")
    ap.add_argument("--cut-generated", type=int, default=0,
                    help="an entry is prompt+generated; drop this many ids from the end "
                         "first (the API log's gen= field) to replay the prompt alone")
    ap.add_argument("--tail", type=int, default=0,
                    help="keep only the last N ids (after --cut-generated); 0 = all")
    ap.add_argument("--ckpt", default="",
                    help="comma-separated CKPT hint ids (the API sends "
                         "<tool_call>,</think>); empty = none")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--drafter", type=int, default=-1,
                    help="-1 engine default, 0 serial, 3 ngram, 4 chain")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8730)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    vocab, entries = read_tcache(args.tcache)
    special, inv, c2b = load_tokenizer(args.tokenizer)
    print("tcache %s: vocab %d, %d entries" % (args.tcache, vocab, len(entries)))

    if args.list:
        for k, e in enumerate(entries):
            print("  [%2d] %6d ids (prompt+generated) | tail: %s"
                  % (k, len(e), describe(e[-10:], special, inv, c2b)))
        return

    if args.entry_index is not None:
        ids = entries[args.entry_index]
    elif args.entry_len is not None:
        match = [e for e in entries if len(e) == args.entry_len]
        if not match:
            raise SystemExit("no entry with %d ids; use --list" % args.entry_len)
        ids = match[0]
    else:
        raise SystemExit("pick an entry with --entry-len/--entry-index (or --list)")

    if args.cut_generated:
        ids = ids[:len(ids) - args.cut_generated]
    if args.tail:
        ids = ids[-args.tail:]
    ckpt = [int(v) for v in args.ckpt.split(",") if v]
    print("replaying %d ids, max_tokens %d, drafter %s, ckpt %s, engine %s:%d"
          % (len(ids), args.max_tokens, args.drafter, ckpt or "-", args.host, args.port))
    print("prompt tail: %s" % describe(ids[-12:], special, inv, c2b))

    tokens, dline = generate(args.host, args.port, ids, args.max_tokens,
                             args.drafter, args.timeout, ckpt)
    print("D line: %s" % dline)
    print("generated %d tokens" % len(tokens))
    print("ids:  %s" % " ".join(str(t) for t in tokens))
    print("desc: %s" % describe(tokens, special, inv, c2b))
    print("text: %r" % decode_ids(tokens, special, inv, c2b))


if __name__ == "__main__":
    sys.exit(main())
