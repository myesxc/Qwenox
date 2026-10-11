#!/usr/bin/env python3
# chat_turns_replay.py — replay a fixed multi-turn chat against a test API and print
# per-turn wall / prompt / cached / reply. Usage: python -I tools/chat_turns_replay.py <api_port> <label>
import json, os, sys, time, urllib.request
port, label = sys.argv[1], sys.argv[2]
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs') + '/'
text1 = open(D + 'PREFILL.md', encoding='utf-8').read()
suffix = open(D + 'CONCURRENCY.md', encoding='utf-8').read()
mid = open(D + 'BENCHMARK.md', encoding='utf-8').read()[:2500]   # ~600-token user turn

def chat(messages, tag, max_tokens=4):
    t0 = time.time()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
        data=json.dumps({"messages": messages, "enable_thinking": False,
                         "temperature": 0, "max_tokens": max_tokens}).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    u = d['usage']; c = d['choices'][0]['message']['content']
    cached = u.get('prompt_tokens_details', {}).get('cached_tokens', 0)
    print(f"{label} {tag:9s} wall={(time.time()-t0)*1000:7.0f}ms prompt={u['prompt_tokens']:6d} "
          f"new={u['prompt_tokens']-cached:5d} reply={c!r}", flush=True)
    return c

h = []
turns = [
    ('T1-cold', text1),
    ('T2-big', "Continue. " + suffix),
    ('T3-small', "What is the main point of the second document? Answer in one short line."),
    ('T4-small', "And one more thing: name the file that documents concurrency behavior."),
    ('T5-small', "Thanks. Is paging enabled by default?"),
    ('T6-mid', "Here is a third document:\n\n" + mid + "\n\nSummarize it in one line."),
    ('T7-small', "Which of the three is longest?"),
    ('T8-small', "OK. Final question: what port does the engine listen on?"),
]
for tag, user in turns:
    reply = chat(h + [{"role": "user", "content": user}], tag)
    h += [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]
