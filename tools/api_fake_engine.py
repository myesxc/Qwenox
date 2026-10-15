#!/usr/bin/env python3
"""Deterministic line-protocol engine used by api_regression_test.py.

Run this on port 18730, start qwenox-api with --engine 127.0.0.1:18730 and
--port 18731, then run api_regression_test.py against that API port.
"""
import argparse
import json
import select
import socket
import struct
import threading
import time


def send_line(conn, line):
    conn.sendall((line + "\n").encode("ascii"))


SLOTS = 1
POOL = 0
TRACE = []
TRACE_LOCK = threading.Lock()
CONNS = set()  # every live client connection, for RESTART
CONNS_LOCK = threading.Lock()


def drop_other_connections(keep):
    """Stand in for the engine process going away under the front-end.

    Every connection but `keep` is shut down while it sits idle, which is what
    a restart leaves behind: the front-end's pooled sockets still look
    connected to it until it writes. Only shutdown here -- the owning handler
    thread wakes from recv and closes its own socket, so a fd is never freed
    from under a thread still using it.
    """
    with CONNS_LOCK:
        others = [conn for conn in CONNS if conn is not keep]
    for conn in others:
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
    return len(others)


# Think-guard scenarios (api_think_guard_test.py), keyed by max_tokens rounded
# up to a multiple of 100 so a resume (max_tokens minus what the first run
# generated) maps back to its scenario. A resume prompt ends with the closer
# the API appends, "</think>\n\n" = [248069, 271]. GUARDLOG returns every
# such GEN.
GUARD_THINK = [9764, 728, 1716, 279, 8831, 13]  # "Let me check the weather."
GUARD_CALL = [248058, 198, 27, 1628, 27362, 67017, 29, 198, 27, 15704, 28, 8656, 29,
              198, 98116, 198, 510, 15704, 29, 198, 510, 1628, 29, 198,
              248059]  # <tool_call> get_weather(city=北京) </tool_call>
GUARD_HI = [12675]
GUARD_LOG = []
GUARD_RESUMES = {}
# Every GEN the engine saw, as {"req", "sample", "bias"} (bias = [[id, value]]).
# GENLOG returns and clears it — used to check the API's default media-marker
# bias without a real engine.
GEN_LOG = []


def guard_scenario(key, resumed, max_tokens):
    """(tokens, reason) for a think-guard scenario GEN."""
    if key == 7100:  # A: the turn ends inside the unclosed think block
        return (GUARD_HI if resumed else GUARD_THINK), "done"
    if key == 7200:  # B: "</think>\n\n" and no answer
        return (GUARD_HI if resumed else [248069, 271]), "done"
    if key == 7300:  # B: "</think> " (whitespace-only answer)
        return (GUARD_HI if resumed else [248069, 220]), "done"
    if key == 7400:  # A, and the first resume ends at once
        if not resumed:
            GUARD_RESUMES[key] = 0
            return GUARD_THINK, "done"
        count = GUARD_RESUMES.get(key, 0)
        GUARD_RESUMES[key] = count + 1
        return ([] if count == 0 else GUARD_HI), "done"
    if key == 7500:  # C: a complete tool call at the end of the reasoning
        return (GUARD_HI if resumed else GUARD_THINK + [271] + GUARD_CALL), "done"
    if key == 7600:  # think-only, but cut by max_tokens: not the guard's case
        return GUARD_THINK, "length"
    if key == 7700:  # think-only text; the request is not in a think block
        return GUARD_THINK, "done"
    if key == 7800:  # honours max_tokens (think_guard_test_cut)
        if resumed:
            return GUARD_HI, "done"
        tokens = (GUARD_THINK * 1300)[:max_tokens]
        return tokens, ("length" if len(tokens) == max_tokens else "done")
    if key == 7900:  # think-only text that a client stop string cuts
        return GUARD_THINK, "done"
    return None


def handle(conn):
    with CONNS_LOCK:
        CONNS.add(conn)
    try:
        serve_connection(conn)
    except OSError:
        pass  # peer reset, or RESTART shut this socket down under us
    finally:
        with CONNS_LOCK:
            CONNS.discard(conn)


def serve_connection(conn):
    pending = b""

    def record_cancel(req):
        with TRACE_LOCK:
            for entry in TRACE:
                if entry["req"] == req:
                    entry["cancel"] += 1

    def cancellable(req, seed):
        nonlocal pending
        entry = {"req": req, "seed": seed, "cancel": 0, "tokens": 0, "done": False}
        with TRACE_LOCK:
            TRACE.append(entry)
        deadline = time.monotonic() + (2.2 if seed == 616163 else 6.0)
        cancelled = False
        while time.monotonic() < deadline:
            if seed != 616162:
                conn.sendall((f"T {req} 12675 -0.125\n" * 32).encode("ascii"))
                with TRACE_LOCK:
                    entry["tokens"] += 32
            if select.select([conn], [], [], 0.01)[0]:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                pending += chunk
            while b"\n" in pending:
                command, pending = pending.split(b"\n", 1)
                if command == f"X {req}".encode("ascii"):
                    record_cancel(req)
                    if not cancelled:
                        cancelled = True
                        deadline = time.monotonic() + 1.2
        send_line(conn, f"T {req} 12675 -0.125")
        reason = "cancel" if cancelled else "done"
        send_line(conn, f"D {req} {reason} 0 0 0.0 0.0 0 0 0 0 0")
        with TRACE_LOCK:
            entry["done"] = True

    def read_exact(size):
        nonlocal pending
        while len(pending) < size:
            chunk = conn.recv(min(65536, size - len(pending)))
            if not chunk:
                return None
            pending += chunk
        result, pending = pending[:size], pending[size:]
        return result

    with conn:
        while True:
            while b"\n" not in pending:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                pending += chunk
            raw, pending = pending.split(b"\n", 1)
            line = raw.decode("ascii", "strict")
            if line == "PING":
                send_line(conn, "PONG")
                continue
            if line == "INFO":
                pool = f" {POOL}" if POOL > 0 else ""
                send_line(conn, f"I 1 0 262144 8 1 1 0 0 0 {SLOTS} 262144 0 1{pool}")
                continue
            if line == "MEM":
                send_line(conn, "M 1 100 120 200 10 20 310 900 1000 100 400 50")
                continue
            if line == "TESTSTATE":
                with TRACE_LOCK:
                    send_line(conn, json.dumps(TRACE))
                continue
            if line == "GUARDLOG":
                with TRACE_LOCK:
                    send_line(conn, json.dumps(GUARD_LOG))
                    GUARD_LOG.clear()
                continue
            if line == "GENLOG":
                with TRACE_LOCK:
                    send_line(conn, json.dumps(GEN_LOG))
                    GEN_LOG.clear()
                continue
            if line == "RESTART":
                # Test hook, not an engine verb: see drop_other_connections.
                send_line(conn, f"OK {drop_other_connections(conn)}")
                continue
            if line.startswith("X "):
                record_cancel(int(line.split()[1]))
                continue
            if not line.startswith("GEN "):
                continue

            fields = line.split()
            req = int(fields[1])
            n_eos = int(fields[3])
            n_ids_at = 4 + n_eos
            n_ids = int(fields[n_ids_at])
            ids = [int(v) for v in fields[n_ids_at + 1:n_ids_at + 1 + n_ids]]

            bias = []
            if "BIAS" in fields:
                at = fields.index("BIAS")
                count = int(fields[at + 1])
                values = fields[at + 2:at + 2 + count * 2]
                bias = [[int(values[i]), float(values[i + 1])]
                        for i in range(0, len(values), 2)]
            with TRACE_LOCK:
                GEN_LOG.append({"req": req, "sample": "SAMPLE" in fields, "bias": bias})

            grids = []
            if "MROPE" in fields:
                at = fields.index("MROPE")
                count = int(fields[at + 1])
                values = [int(value) for value in fields[at + 2:at + 2 + count * 3]]
                if len(values) != count * 3:
                    return
                grids = [tuple(values[index:index + 3])
                         for index in range(0, len(values), 3)]

            frame_ok = True
            if "VIMG" in fields:
                at = fields.index("VIMG")
                count = int(fields[at + 1])
                frame_ok = count == len(grids) and count <= 8
                for index in range(count):
                    header = read_exact(16)
                    if header is None:
                        return
                    magic, patches, nbytes = struct.unpack("<IIQ", header)
                    if nbytes > 512 * 1024 * 1024:
                        return
                    payload = read_exact(nbytes)
                    if payload is None:
                        return
                    expected = grids[index][0] * grids[index][1] * grids[index][2]
                    frame_ok = (frame_ok and magic == 0x56494D31 and
                                patches == expected and nbytes == patches * 1536 * 4)
            elif grids:
                frame_ok = False
            if not frame_ok:
                send_line(conn, f"D {req} error 0 0 0.0 0.0 0 0 0 0 0")
                continue

            seed = int(fields[fields.index("SAMPLE") + 5]) if "SAMPLE" in fields else 0
            if seed in (616161, 616162, 616163):
                cancellable(req, seed)
                continue

            max_tokens = int(fields[2])
            guard_key = (max_tokens + 99) // 100 * 100
            if 7100 <= guard_key <= 7900:
                resumed = ids[-2:] == [248069, 271] and guard_key != 7700
                tokens, reason = guard_scenario(guard_key, resumed, max_tokens)
                with TRACE_LOCK:
                    GUARD_LOG.append({"key": guard_key, "resumed": resumed,
                                      "max_tokens": max_tokens, "seed": seed,
                                      "sampled": "SAMPLE" in fields, "tail": ids[-40:]})
                for token in tokens:
                    send_line(conn, f"T {req} {token} -0.1")
                send_line(conn, f"D {req} {reason} {n_ids} {len(tokens)} 1.0 2.0 0 0 0 0 0")
                continue

            if "424242" in fields:
                return
            if "31337" in fields:
                time.sleep(2)
            if "5150" in fields:  # aborted mid-decode (shared KV pool full)
                send_line(conn, f"T {req} 12675 -0.125")
                send_line(conn, f"D {req} error 0 0 0.0 0.0 0 0 0 0 0")
                continue
            if "9001" in fields:
                tokens = [
                    248069, 271, 248058, 198, 27, 1628, 27362, 67017, 29, 198,
                    27, 15704, 28, 8656, 29, 198, 98116, 198, 510, 15704, 29,
                    198, 27, 15704, 28, 13382, 29, 198, 18, 198, 510, 15704,
                    29, 198, 510, 1628, 29, 198, 248059,
                ]
                tokens = [(token, -0.1) for token in tokens]
            elif "9002" in fields:
                tokens = [27, 15704, 28, 41843, 29, 198, 37186, 73594, 29086,
                          198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059]
                tokens = [(token, -0.1) for token in tokens]
            elif "9003" in fields:
                tokens = [447, 67017, 29, 198, 27, 15704, 28, 8656, 29, 198,
                          98817, 198, 510, 15704, 29, 198, 510, 1628, 29, 198,
                          248059]
                tokens = [(token, -0.1) for token in tokens]
            elif n_ids <= 2:
                tokens = [(4130, -0.5), (284, -0.25)]  # " Par" + "is"
            elif 248069 in ids:
                tokens = [(12675, -0.125)]  # enable_thinking=false -> "Hi"
            else:
                tokens = [(248069, -0.75), (12675, -0.125)]  # "</think>Hi"
            for token, logprob in tokens:
                send_line(conn, f"T {req} {token} {logprob}")
            send_line(conn, f"D {req} done {n_ids} {len(tokens)} 1.0 2.0 0 0 0 0 0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18730)
    parser.add_argument("--slots", type=int, default=1, help="INFO kv_slots")
    parser.add_argument("--pool", type=int, default=0,
                        help="INFO kv_pool_tokens; 0 = omit (old engine)")
    args = parser.parse_args()
    global SLOTS, POOL
    SLOTS = args.slots
    POOL = args.pool
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", args.port))
        server.listen(16)
        while True:
            conn, _ = server.accept()
            threading.Thread(target=handle, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    main()
