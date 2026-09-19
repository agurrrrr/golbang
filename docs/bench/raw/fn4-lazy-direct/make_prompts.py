#!/usr/bin/env python3
"""Build diverse ~500k-char prompts from the ik_llama.cpp github-data corpus.

Each prompt is a different RNG-shuffled concatenation so no long common prefix
exists across repairs (defeats the server KV prefix cache).
"""
import pathlib
import random
import sys

SRC = pathlib.Path("/home/agurrrrr/code/ik_llama.cpp/github-data")
OUT = pathlib.Path("/tmp/opencode/fn4/prompts")
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 480_000

files = sorted(str(p) for p in SRC.rglob("*") if p.is_file())
if not files:
    sys.exit("no corpus files found")

NAMES = [
    ("cold_1", 101),
    ("cold_2", 202),
    ("cold_3", 303),
    ("warm_1", 411),
    ("warm_2", 522),
    ("warm_3", 633),
]

OUT.mkdir(parents=True, exist_ok=True)
for name, seed in NAMES:
    rng = random.Random(seed)
    order = files[:]
    rng.shuffle(order)
    buf, n = [], 0
    for f in order:
        try:
            t = pathlib.Path(f).read_text(errors="ignore")
        except OSError:
            continue
        buf.append(t)
        n += len(t)
        if n >= TARGET:
            break
    text = "\n\n".join(buf)[:TARGET]
    (OUT / name).write_text(text, encoding="utf-8")
    print(f"{name}: {len(text)} chars")
