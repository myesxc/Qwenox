#!/usr/bin/env python3
"""prefill_sched.py — client for tools/prefill_sched_win.sh (10-13 prefill scheduling fix).

The engine is already serving on 127.0.0.1:8732 with QWENOX_PARALLEL=2 and
QWENOX_RCKPT=0 / QWENOX_KVSNAP=0 (every prompt prefills from scratch).
Prompts are real text from data/qsa-oracle/131072.tokens (one id per line).

  run  --tag T     T1..T4 below -> logs/psched/T.json
  gate --old O --new N [--olog F --nlog F]   print the comparison, last line PASS/FAIL

  T1  2 x 24K greedy (3 chunks each, same chunk grid): concurrent output must
      equal each prompt's solo output token for token. Before the fix the
      first-admitted prompt read the other prompt's PLE rows from chunk 2 on.
  T2  2 x 64K greedy: solo A, solo B (also warms the PLE page cache), then both
      concurrently: per-request TTFT + bit-exact vs solo.
  T3  decode stall: a short prompt decodes; a 32K prompt arrives; the largest
      gap between the decoder's tokens while the 32K prompt prefills.
  T4  a 8K prompt arrives 3 s into a 64K prefill: its TTFT (must not wait
      for the whole 64K prompt).
"""
import argparse, json, socket, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = 8732


def root_dir():
    import os
    return Path(os.environ.get('PSCHED_ROOT', str(ROOT)))


def out_dir():
    return root_dir() / 'logs' / 'psched'


class Conn:
    def __init__(self, timeout=3600):
        self.sock = socket.create_connection(('127.0.0.1', PORT), timeout=timeout)
        self.f = self.sock.makefile('rb')
        self.req = 0

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def gen(self, ids, n, drafter, on_tok=None):
        self.req += 1
        fields = ['GEN', self.req, n, 0, len(ids), *ids, drafter]
        t_send = time.time()
        self.sock.sendall((' '.join(map(str, fields)) + '\n').encode())
        out, times = [], []
        while True:
            v = self.f.readline().decode().split()
            if not v:
                raise RuntimeError('engine disconnected')
            if v[0] == 'T':
                out.append(int(v[2]))
                times.append(time.time())
                if on_tok:
                    on_tok(len(out))
            elif v[0] == 'D':
                num = lambda i, t=int: t(v[i]) if len(v) > i else None
                return dict(tokens=out, reason=v[2], pre_ms=num(5, float), dec_ms=num(6, float),
                            cached=num(10), done=' '.join(v), t_send=t_send,
                            ttft=(times[0] - t_send) if times else None,
                            times=[t - t_send for t in times])


def one(ids, n, d, res, key, delay=0.0, on_tok=None):
    try:
        if delay:
            time.sleep(delay)
        c = Conn()
        r = c.gen(ids, n, d, on_tok)
        c.close()
        res[key] = r
    except Exception as e:  # noqa: BLE001
        res[key] = dict(error=str(e))


def pair(a, b, n, d, gap=0.3):
    res = {}
    th = [threading.Thread(target=one, args=(a, n, d, res, 'A')),
          threading.Thread(target=one, args=(b, n, d, res, 'B', gap))]
    t0 = time.time()
    for t in th:
        t.start()
    for t in th:
        t.join()
    res['wall'] = time.time() - t0
    return res


def show(tag, key, r):
    if 'error' in r:
        print(f'  {tag} {key}: ERROR {r["error"]}', flush=True)
        return
    print(f'  {tag} {key:<10} n={len(r["tokens"]):<4} {r["reason"]:<7} ttft {r["ttft"]:7.1f}s '
          f'prompt {r["pre_ms"] / 1000:7.1f}s decode {r["dec_ms"] / 1000:6.1f}s cached {r["cached"]}',
          flush=True)


def load_tokens():
    p = root_dir() / 'data' / 'qsa-oracle' / '131072.tokens'
    return [int(x) for x in p.read_text().split()]


def cmd_run(tag):
    T = load_tokens()
    assert len(T) >= 131072, len(T)
    res = {}
    solo = Conn()

    def s(key, ids, n, d):
        r = solo.gen(ids, n, d)
        res[key] = r
        show(tag, key, r)

    print(f'[{time.strftime("%H:%M:%S")}] T1: 2 x 24K greedy, solo vs concurrent', flush=True)
    A24, B24 = T[0:24576], T[60000:84576]
    s('t1_soloA', A24, 48, 4)
    s('t1_soloB', B24, 48, 4)
    p = pair(A24, B24, 48, 4)
    res['t1_conc'] = p
    for k in 'AB':
        show(tag, 't1_conc' + k, p[k])

    print(f'[{time.strftime("%H:%M:%S")}] T2: 2 x 64K greedy, solo A / solo B / concurrent', flush=True)
    A64, B64 = T[0:65536], T[65536:131072]
    s('t2_soloA', A64, 64, 4)
    s('t2_soloB', B64, 64, 4)
    p = pair(A64, B64, 64, 4)
    res['t2_conc'] = p
    for k in 'AB':
        show(tag, 't2_conc' + k, p[k])
    print(f'  {tag} t2 wall {p["wall"]:.1f}s', flush=True)

    print(f'[{time.strftime("%H:%M:%S")}] T3: decode stall while a 32K prompt prefills', flush=True)
    short = T[100000:100200]
    r3 = {}
    started = threading.Event()
    ta = threading.Thread(target=one, args=(short, 700, 4, r3, 'dec', 0.0,
                                            lambda k: k == 20 and started.set()))
    ta.start()
    started.wait(120)
    t_b = time.time()
    one(T[0:32768], 1, 0, r3, 'pp')
    t_b_first = t_b + (r3['pp'].get('ttft') or 0)
    ta.join()
    dec = r3['dec']
    gaps = []
    if 'error' not in dec and 'error' not in r3['pp']:
        abs_t = [dec['t_send'] + t for t in dec['times']]
        win = [t for t in abs_t if t_b <= t <= t_b_first]
        # the token right before the window counts too (gap into the window)
        before = [t for t in abs_t if t < t_b]
        seq = ([before[-1]] if before else []) + win
        gaps = [b - a for a, b in zip(seq, seq[1:])]
        res['t3'] = dict(max_gap=max(gaps) if gaps else None, n_in_window=len(win),
                         pp_ttft=r3['pp']['ttft'], dec_tokens=len(dec['tokens']))
        print(f'  {tag} t3 decoder tokens during prefill {len(win)}, max gap '
              f'{res["t3"]["max_gap"] or 0:.2f}s, 32K TTFT {r3["pp"]["ttft"]:.1f}s', flush=True)
    else:
        res['t3'] = dict(error=str(dec.get('error') or r3['pp'].get('error')))
        print(f'  {tag} t3 ERROR {res["t3"]["error"]}', flush=True)

    print(f'[{time.strftime("%H:%M:%S")}] T4: 8K prompt arrives 3 s into a 64K prefill', flush=True)
    p = pair(T[0:65536], T[70000:78192], 16, 4, gap=3.0)
    res['t4'] = p
    show(tag, 't4_long', p['A'])
    show(tag, 't4_short', p['B'])
    solo.close()

    out_dir().mkdir(parents=True, exist_ok=True)
    (out_dir() / f'{tag}.json').write_text(json.dumps(res))
    errs = [k for k, v in res.items() if isinstance(v, dict) and 'error' in v]
    for k in ('t1_conc', 't2_conc', 't4'):
        for kk in 'AB':
            if 'error' in res[k][kk]:
                errs.append(f'{k}.{kk}')
    if errs:
        print('RUN FAIL: errors in', errs)
        return False
    return True


def cmd_smoke():
    """Quick liveness check of a new build (no gate): 2 x 16K, then 2K into 16K."""
    import socket as _s
    _s.setdefaulttimeout(300)
    T = load_tokens()
    p = pair(T[0:16384], T[40000:56384], 16, 4)
    for k in 'AB':
        show('smoke', 'pair' + k, p[k])
    q = pair(T[0:16384], T[70000:72048], 16, 4, gap=1.0)
    show('smoke', 'long', q['A'])
    show('smoke', 'short', q['B'])
    ok = all('error' not in x[k] for x in (p, q) for k in 'AB')
    print('SMOKE', 'PASS' if ok else 'FAIL')
    return ok


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def cmp_tok(r, ref):
    if 'error' in r or 'error' in ref:
        return 'ERR'
    d = first_diff(r['tokens'], ref['tokens'])
    return '=' if d is None else f'DIFF@{d}'


def log_stats(path):
    if not path or not Path(path).exists():
        return {}
    txt = Path(path).read_text(encoding='utf-8', errors='replace').replace('\r', '\n')
    return dict(lends=txt.count('prefill lends GPU'), ple_drop=txt.count('ple: drop flight'),
                sched=next((ln for ln in txt.split('\n') if 'prefill scheduling:' in ln), ''))


def cmd_gate(old, new, olog, nlog):
    O = json.loads((out_dir() / f'{old}.json').read_text())
    N = json.loads((out_dir() / f'{new}.json').read_text())
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        ok &= bool(cond)
        print(f'  [{"PASS" if cond else "FAIL"}] {name}: {detail}')

    print('== correctness (concurrent tokens vs solo tokens, greedy) ==')
    for lab, R in (('old', O), ('new', N)):
        t1 = [cmp_tok(R['t1_conc'][k], R['t1_solo' + k]) for k in 'AB']
        t2 = [cmp_tok(R['t2_conc'][k], R['t2_solo' + k]) for k in 'AB']
        print(f'  {lab}: 24K A {t1[0]}  B {t1[1]} | 64K A {t2[0]}  B {t2[1]}')
    print('  (old: a DIFF on the first-admitted prompt A = the PLE cross-slot bug reproduced)')
    n1 = [cmp_tok(N['t1_conc'][k], N['t1_solo' + k]) for k in 'AB']
    n2 = [cmp_tok(N['t2_conc'][k], N['t2_solo' + k]) for k in 'AB']
    check('new 2x24K concurrent == solo', n1 == ['=', '='], ' '.join(n1))
    check('new 2x64K concurrent == solo', n2 == ['=', '='], ' '.join(n2))
    o_same = all(cmp_tok(O[f'{t}_solo{k}'], N[f'{t}_solo{k}']) == '=' for t in ('t1', 't2') for k in 'AB')
    print(f'  solo outputs old vs new identical: {o_same} (info: single-stream path unchanged)')

    print('== 2 x 64K timing (client TTFT, s) ==')
    for lab, R in (('old', O), ('new', N)):
        c = R['t2_conc']
        print(f'  {lab}: solo A {R["t2_soloA"]["ttft"]:.1f}  solo B {R["t2_soloB"]["ttft"]:.1f} | '
              f'conc A {c["A"]["ttft"]:.1f}  B {c["B"]["ttft"]:.1f}  wall {c["wall"]:.1f}')
    sa, sb = N['t2_soloA']['ttft'], N['t2_soloB']['ttft']
    ca, cb = N['t2_conc']['A']['ttft'], N['t2_conc']['B']['ttft']
    first, last = min(ca, cb), max(ca, cb)
    check('new first-finished TTFT <= 1.15 x its solo', first <= 1.15 * max(sa, sb),
          f'{first:.1f}s vs solo {max(sa, sb):.1f}s')
    check('new last TTFT <= 1.12 x (solo A + solo B)', last <= 1.12 * (sa + sb),
          f'{last:.1f}s vs {sa + sb:.1f}s')
    oc = O['t2_conc']
    olast = max(oc['A']['ttft'], oc['B']['ttft'])
    check('new last TTFT < old last TTFT', last < olast, f'{last:.1f}s vs old {olast:.1f}s')
    agg_o = 131072 / olast
    agg_n = 131072 / last
    print(f'  aggregate prefill (131072 tok / last TTFT): old {agg_o:.0f} -> new {agg_n:.0f} tok/s; '
          f'solo new {65536 / sa:.0f} tok/s')

    print('== decode stall while a 32K prompt prefills ==')
    og, ng = O['t3'].get('max_gap'), N['t3'].get('max_gap')
    print(f'  old max gap {og}  new max gap {ng}  (32K TTFT old {O["t3"].get("pp_ttft")} new {N["t3"].get("pp_ttft")})')
    check('new decode max gap <= max(1.5 s, 1.3 x old)',
          ng is not None and og is not None and ng <= max(1.5, 1.3 * og), f'{ng} vs old {og}')

    print('== short prompt arriving during a long prefill ==')
    for lab, R in (('old', O), ('new', N)):
        print(f'  {lab}: 8K TTFT {R["t4"]["B"]["ttft"]:.1f}s (arrived 3 s in), 64K TTFT {R["t4"]["A"]["ttft"]:.1f}s')
    st = N['t4']['B']['ttft']
    check('new 8K TTFT <= 15 s', st <= 15, f'{st:.1f}s')

    so, sn = log_stats(olog), log_stats(nlog)
    if so or sn:
        print('== engine logs ==')
        print(f'  old: lends {so.get("lends")}')
        print(f'  new: lends {sn.get("lends")}, PLE flight drops {sn.get("ple_drop")}')
        print(f'  new: {sn.get("sched", "").strip()}')
        check('new engine runs the new scheduler', 'shortest-remaining-first' in sn.get('sched', ''),
              'log line found' if sn.get('sched') else 'missing')
    return ok


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    r = sp.add_parser('run')
    r.add_argument('--tag', required=True)
    sp.add_parser('smoke')
    g = sp.add_parser('gate')
    g.add_argument('--old', required=True)
    g.add_argument('--new', required=True)
    g.add_argument('--olog')
    g.add_argument('--nlog')
    a = ap.parse_args()
    if a.cmd == 'run':
        sys.exit(0 if cmd_run(a.tag) else 1)
    if a.cmd == 'smoke':
        sys.exit(0 if cmd_smoke() else 1)
    ok = cmd_gate(a.old, a.new, a.olog, a.nlog)
    print('PREFILL_SCHED_GATE', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
