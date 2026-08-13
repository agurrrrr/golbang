#!/usr/bin/env python3
"""P2 bench: golbang (n_parallel=1 vs 2) and llama-server, same GGUF/prompt."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


BODY = json.dumps(
    {
        "model": "qwen",
        "messages": [{"role": "user", "content": "안녕"}],
        "stream": True,
        "max_tokens": 16,
        "temperature": 0,
    }
).encode()


def vram_used_mib() -> int | None:
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        data = json.loads(out)
        for key, val in data.items():
            if not isinstance(val, dict):
                continue
            for k, v in val.items():
                if "used" in k.lower() and "memory" in k.lower():
                    return int(v) // (1024 * 1024)
            if "VRAM Total Used Memory (B)" in val:
                return int(val["VRAM Total Used Memory (B)"]) // (1024 * 1024)
    except Exception:
        pass
    try:
        out = subprocess.check_output(["rocm-smi"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if "%" in line and "0x66a1" in line:
                # Device line: ... 95% 0%
                parts = line.split()
                for p in parts:
                    if p.endswith("%"):
                        pct = float(p[:-1])
                        return int(32 * 1024 * pct / 100)
    except Exception:
        return None
    return None


def one_stream(url: str, start_gate: threading.Event) -> dict:
    start_gate.wait()
    t0 = time.perf_counter()
    req = urllib.request.Request(
        url,
        data=BODY,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    ttft = None
    times = []
    n = 0
    with urllib.request.urlopen(req, timeout=60) as resp:
        buf = b""
        while True:
            chunk = resp.read(1)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    continue
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = (
                    obj.get("choices", [{}])[0].get("delta")
                    or obj.get("choices", [{}])[0].get("message")
                    or {}
                )
                # Qwen3 via llama-server --jinja emits thinking as reasoning_content.
                if delta.get("content") or delta.get("reasoning_content"):
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                    times.append(now)
                    n += 1
    wall = time.perf_counter() - t0
    itls = [b - a for a, b in zip(times, times[1:])]
    gen = (times[-1] - times[0]) if len(times) >= 2 else 0.0
    return {
        "ttft_ms": (ttft or wall) * 1000,
        "itl_ms": (statistics.mean(itls) * 1000) if itls else 0.0,
        "tok": n,
        "tok_s": (n / gen) if gen > 0 else 0.0,
        "wall_ms": wall * 1000,
    }


def run_pair(url: str, n: int = 2) -> list[dict]:
    gate = threading.Event()
    results: list[dict] = []
    errors: list[BaseException] = []

    def worker():
        try:
            results.append(one_stream(url, gate))
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    time.sleep(0.05)
    gate.set()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    results.sort(key=lambda r: r["ttft_ms"])
    return results


def summarize(label: str, rows: list[dict], vram: int | None) -> dict:
    late = max(r["ttft_ms"] for r in rows)
    early = min(r["ttft_ms"] for r in rows)
    itl = statistics.mean(r["itl_ms"] for r in rows)
    toks = statistics.mean(r["tok_s"] for r in rows)
    wall = max(r["wall_ms"] for r in rows)
    out = {
        "label": label,
        "early_ttft_ms": early,
        "late_ttft_ms": late,
        "itl_ms": itl,
        "tok_s": toks,
        "wall_ms": wall,
        "vram_mib": vram,
        "per_req": rows,
    }
    print(
        f"{label}: late_ttft={late:.1f}ms early={early:.1f}ms "
        f"itl={itl:.1f}ms tok/s={toks:.2f} wall={wall:.1f}ms vram={vram}"
    )
    return out


def wait_http(url: str, timeout: float = 30.0) -> None:
    """POST a tiny completion so GET-only probes don't 404/405."""
    deadline = time.time() + timeout
    last = None
    probe = json.dumps(
        {
            "model": "qwen",
            "messages": [{"role": "user", "content": "x"}],
            "stream": False,
            "max_tokens": 1,
            "temperature": 0,
        }
    ).encode()
    while time.time() < deadline:
        try:
            req = urllib.request.Request(
                url,
                data=probe,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10)
            return
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(0.2)
    raise RuntimeError(f"server at {url} did not accept POST: {last}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--golbang", default="http://127.0.0.1:8088/v1/chat/completions")
    p.add_argument("--llama", default="")
    p.add_argument("--out", default="")
    args = p.parse_args()

    all_rows = []
    vram = vram_used_mib()
    try:
        wait_http(args.golbang, timeout=5)
        all_rows.append(summarize("golbang", run_pair(args.golbang), vram))
    except Exception as e:  # noqa: BLE001
        print(f"golbang measure failed: {e}", file=sys.stderr)

    if args.llama:
        try:
            wait_http(args.llama, timeout=5)
            all_rows.append(summarize("llama-server", run_pair(args.llama), vram_used_mib()))
        except Exception as e:  # noqa: BLE001
            print(f"llama-server measure failed: {e}", file=sys.stderr)

    if args.out:
        Path(args.out).write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
