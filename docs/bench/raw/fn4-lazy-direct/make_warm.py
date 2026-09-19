import pathlib, random
SRC = pathlib.Path("/home/agurrrrr/code/ik_llama.cpp/github-data")
OUT = pathlib.Path("/tmp/opencode/fn4/prompts")
files = sorted(str(p) for p in SRC.rglob("*") if p.is_file())
def build(name, chars, seed):
    rng = random.Random(seed); order = files[:]; rng.shuffle(order)
    buf, n = [], 0
    for f in order:
        try: t = pathlib.Path(f).read_text(errors="ignore")
        except OSError: continue
        buf.append(t); n += len(t)
        if n >= chars: break
    (OUT/name).write_text("\n\n".join(buf)[:chars], encoding="utf-8")
    print(name, chars)
build("wu", 120000, 9001)
for i, s in enumerate([7001,7002,7003,7004,7005], 1):
    build(f"t{i}", 40000, s)
