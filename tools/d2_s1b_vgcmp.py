#!/usr/bin/env python3
# d2_s1b_vgcmp.py — 跨 γ 的 VGDUMP 对齐比较器（D2-S1b）。
# 用法: python tools/d2_s1b_vgcmp.py logs/d2s1b_vge_g2.log logs/d2s1b_vge_g3.log
# 行探针格式: [tag] P=N base=B rows h0 h1 ...
# 键 = (tag, base+row)（绝对位置），行内容必须与 (base,P,row) 无关。
# 同一键在一次运行中可出现多次（回滚后重算）：按出现序逐一比较。
# 报告: 每个 tag 的首次分叉绝对位置 + 全局最早分叉位置的所有 tag。
import re, sys, collections

# 非行语义的探针：ir* 是 P=2 硬编码的 buf 切半；ikp/ikl/ih/pt 长度依赖
# nb=(base+P)/4，跨 P 不可比；vge 的 base=-1 无位置语义。
SKIP = re.compile(r'^(ir[TDCEH]|ir0_|ikp|ikl|ih\d|pt\d)')

def load(path):
    # (tag, abspos) -> last (P, row, hash)。同一位置回滚后重算多次，
    # 只有最后一次（输入=已提交 token）是可比的提交语义计算。
    d = {}
    tagorder = {}
    for line in open(path, errors='replace'):
        m = re.match(r'\[(\w+)\] P=(\d+) base=(-?\d+) rows((?: [0-9a-f]{16})+)', line)
        if not m:
            continue
        tag, P, base, hs = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4).split()
        if base < 0 or SKIP.match(tag):
            continue
        if tag not in tagorder:
            tagorder[tag] = len(tagorder)
        for r, h in enumerate(hs):
            d[(tag, base + r)] = (P, r, h)
    return d, tagorder

a, order_a = load(sys.argv[1])
b, order_b = load(sys.argv[2])
print(f'{sys.argv[1]}: {len(a)} committed row-probes, '
      f'{sys.argv[2]}: {len(b)}')

mism = []   # (abspos, tag, detA, detB)
only_a = only_b = 0
for key in sorted(set(a) | set(b), key=lambda k: (k[1], k[0])):
    xa, xb = a.get(key), b.get(key)
    if xa is None:
        only_b += 1; continue
    if xb is None:
        only_a += 1; continue
    if xa[2] != xb[2]:
        mism.append((key[1], key[0], xa, xb))

print(f'keys only in A: {only_a}, only in B: {only_b}, mismatching: {len(mism)}')
if not mism:
    print('ALIGNED-IDENTICAL: all shared (tag, abspos) committed rows bit-match')
    sys.exit(0)

first_pos = mism[0][0]
print(f'\nearliest diverging abspos = {first_pos}:')
for pos, tag, xa, xb in mism:
    if pos != first_pos:
        break
    print(f'  [{tag}] A(P={xa[0]} row={xa[1]} h={xa[2]}) vs B(P={xb[0]} row={xb[1]} h={xb[2]})')

print('\nper-tag first divergence (pipeline order):')
seen = {}
for pos, tag, xa, xb in mism:
    if tag not in seen:
        seen[tag] = pos
for tag in sorted(seen, key=lambda t: order_a.get(t, 1 << 30)):
    print(f'  [{tag:8s}] first diff at abspos {seen[tag]}')
sys.exit(1)
