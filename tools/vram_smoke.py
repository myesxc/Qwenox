#!/usr/bin/env python3
"""One GEN on the engine at 127.0.0.1:8732 (tools/vram_footprint_win.sh).

  python tools/vram_smoke.py <tokens-file> <prompt-tokens> <gen-tokens>

Prints INFO, then "SMOKE OK ..." with TTFT and decode speed, or "SMOKE FAIL ..."
(exit 1) on an engine error / cancel / disconnect.
"""
import socket, sys, time

ids = open(sys.argv[1]).read().split()[:int(sys.argv[2])]
s = socket.create_connection(('127.0.0.1', 8732), timeout=1800)
f = s.makefile('rb')
s.sendall(b'INFO\n')
print('INFO', f.readline().decode().strip()[:300])
t0 = time.time()
s.sendall(('GEN 1 %s 0 %d %s 4\n' % (sys.argv[3], len(ids), ' '.join(ids))).encode())
n, tf = 0, None
while True:
    v = f.readline().decode().split()
    if not v:
        print('SMOKE FAIL engine disconnected')
        sys.exit(1)
    if v[0] == 'T':
        n += 1
        tf = tf or time.time()
    elif v[0] == 'D' and len(v) > 2 and v[2] in ('error', 'cancel'):
        print('SMOKE FAIL', ' '.join(v))
        sys.exit(1)
    elif v[0] == 'D':
        first = tf or t0
        dt = time.time() - first
        tps = (n - 1) / dt if n > 1 and dt > 0 else 0.0
        print('SMOKE OK prompt %d gen %d ttft %.1f s decode %.1f tok/s | %s'
              % (len(ids), n, first - t0, tps, ' '.join(v)[:120]))
        break
