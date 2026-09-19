#!/usr/bin/env python3
"""Send one prompt file, record prefill timings and /proc/<pid> fault+IO deltas."""
import argparse
import json
import os
import pathlib
import time
import urllib.request

API_KEY = os.environ.get("BENCH_API_KEY", "bench")


def proc_stats(pid: int) -> dict:
    st = pathlib.Path(f"/proc/{pid}/stat").read_text().split()
    out = {"utime": int(st[13]), "stime": int(st[14]), "minflt": int(st[9]), "majflt": int(st[11])}
    try:
        for line in pathlib.Path(f"/proc/{pid}/io").read_text().splitlines():
            k, v = line.split(": ")
            out[f"io_{k}"] = int(v)
    except OSError:
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=4)
    ap.add_argument("--url", default="http://127.0.0.1:8090/v1/chat/completions")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--out")
    args = ap.parse_args()

    prompt = pathlib.Path(args.prompt).read_text(encoding="utf-8")
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "stream": False,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
    }
    before = proc_stats(args.pid)
    t0 = time.time()
    req = urllib.request.Request(
        args.url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=3600) as resp:
        payload = json.loads(resp.read())
    wall = time.time() - t0
    after = proc_stats(args.pid)

    t = payload.get("timings", {})
    rec = {
        "prompt_tokens": payload.get("usage", {}).get("prompt_tokens"),
        "completion_tokens": payload.get("usage", {}).get("completion_tokens"),
        "wall_s": round(wall, 2),
        "prompt_per_second": t.get("prompt_per_second"),
        "predicted_per_second": t.get("predicted_per_second"),
        "prompt_n": t.get("prompt_n"),
        "cache_n": t.get("cache_n"),
        "draft_n": t.get("draft_n"),
        "draft_n_accepted": t.get("draft_n_accepted"),
        "deltas": {k: after[k] - before.get(k, 0) for k in after},
    }
    rec["text"] = (payload.get("choices", [{}])[0].get("message", {}).get("content") or "")[:120]
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(payload, ensure_ascii=False))
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
