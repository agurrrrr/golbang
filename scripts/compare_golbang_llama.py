#!/usr/bin/env python3
"""Controlled A/B: production golbang-deepseek vs llama-server on the same host.

Measures short / medium / long prefill, decode, 2-turn prefix reuse, and
queue back-pressure. Writes a JSON report. Does not start/stop systemd.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


API_KEY = os.environ.get("GOLBANG_API_KEY", "")
PARA = (
    "DeepSeek-V4-Flash is a mixture-of-experts language model. "
    "Each token activates a small subset of experts. "
    "On a 32 GiB MI50 the remaining experts stay memory-mapped on the host. "
    "This paragraph is filler so the prompt length is stable across servers. "
)


def vram_used_mib() -> int | None:
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        data = json.loads(out)
        for val in data.values():
            if not isinstance(val, dict):
                continue
            if "VRAM Total Used Memory (B)" in val:
                return int(val["VRAM Total Used Memory (B)"]) // (1024 * 1024)
    except Exception:
        return None
    return None


def gpu_sclk() -> str | None:
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showclocks"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            if "sclk clock level" in line.lower() or "sclk" in line.lower():
                return line.strip()
    except Exception:
        return None
    return None


def repeat_to_chars(n_chars: int) -> str:
    buf = []
    while sum(len(x) for x in buf) < n_chars:
        buf.append(PARA)
    return "".join(buf)[:n_chars]


def chat(
    url: str,
    messages: list[dict],
    *,
    max_tokens: int = 32,
    stream: bool = False,
    timeout: float = 600.0,
    temperature: float = 0.0,
) -> dict:
    body = json.dumps(
        {
            "model": "dsv4",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
    ).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.status
            retry_after = resp.headers.get("Retry-After")
    except urllib.error.HTTPError as e:
        wall = time.perf_counter() - t0
        payload = e.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "http": e.code,
            "wall_s": wall,
            "retry_after": e.headers.get("Retry-After") if e.headers else None,
            "error": payload[:400],
        }
    wall = time.perf_counter() - t0
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "http": status,
            "wall_s": wall,
            "error": raw[:400].decode("utf-8", errors="replace"),
        }
    choice = (obj.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = obj.get("usage") or {}
    timings = obj.get("timings") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    return {
        "ok": True,
        "http": status,
        "wall_s": wall,
        "retry_after": retry_after,
        "finish_reason": choice.get("finish_reason"),
        "content": content,
        "reasoning_n": len(reasoning),
        "content_n": len(content),
        "usage": usage,
        "timings": timings,
        "prompt_n": timings.get("prompt_n", usage.get("prompt_tokens")),
        "predicted_n": timings.get("predicted_n", usage.get("completion_tokens")),
        "cache_n": timings.get("cache_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "predicted_ms": timings.get("predicted_ms"),
        "prompt_per_second": timings.get("prompt_per_second"),
        "predicted_per_second": timings.get("predicted_per_second"),
    }


def wait_ready(url: str, timeout: float = 600.0) -> None:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            r = chat(
                url,
                [{"role": "user", "content": "ping"}],
                max_tokens=1,
                timeout=180,
            )
            if r.get("ok") or r.get("http") in (200, 400, 401):
                if r.get("http") == 401:
                    raise RuntimeError("server rejected API key")
                return
            last = r
        except Exception as e:  # noqa: BLE001
            last = str(e)
        time.sleep(2)
    raise RuntimeError(f"server not ready: {last}")


def summarize_run(label: str, r: dict) -> dict:
    out = {
        "label": label,
        **{k: r.get(k) for k in (
            "ok",
            "http",
            "wall_s",
            "finish_reason",
            "prompt_n",
            "predicted_n",
            "cache_n",
            "prompt_ms",
            "predicted_ms",
            "prompt_per_second",
            "predicted_per_second",
            "content_n",
            "reasoning_n",
            "content",
        )},
    }
    print(
        f"  {label}: ok={r.get('ok')} http={r.get('http')} "
        f"prompt_n={r.get('prompt_n')} cache_n={r.get('cache_n')} "
        f"prefill={r.get('prompt_per_second')} tok/s "
        f"decode={r.get('predicted_per_second')} tok/s "
        f"wall={r.get('wall_s'):.2f}s"
        if isinstance(r.get("wall_s"), (int, float))
        else f"  {label}: {r}"
    )
    return out


def queue_probe(url: str, n: int = 4) -> dict:
    """Fire n concurrent short-but-not-instant jobs. Record statuses."""
    t0 = time.perf_counter()
    results = []

    def one(i: int) -> dict:
        r = chat(
            url,
            [{"role": "user", "content": f"Write the number {i} and stop."}],
            max_tokens=16,
            timeout=180,
        )
        r["idx"] = i
        return r

    with ThreadPoolExecutor(max_workers=n) as pool:
        futs = [pool.submit(one, i) for i in range(n)]
        for fut in as_completed(futs):
            results.append(fut.result())
    wall = time.perf_counter() - t0
    results.sort(key=lambda r: r.get("idx", 0))
    codes = [r.get("http") for r in results]
    return {
        "n": n,
        "wall_s": wall,
        "http": codes,
        "n_200": sum(1 for c in codes if c == 200),
        "n_503": sum(1 for c in codes if c == 503),
        "retry_after": [r.get("retry_after") for r in results],
        "per_req_wall_s": [r.get("wall_s") for r in results],
    }


def run_suite(name: str, url: str) -> dict:
    print(f"\n=== {name} {url} ===")
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    wait_ready(url)
    vram0 = vram_used_mib()
    sclk0 = gpu_sclk()
    print(f"  ready vram={vram0} MiB sclk={sclk0}")

    # Warmup already happened in wait_ready (max_tokens=1).
    short_prompt = [{"role": "user", "content": "1+1="}]
    medium_text = repeat_to_chars(2800)
    long_text = repeat_to_chars(11000)
    medium_prompt = [
        {
            "role": "user",
            "content": medium_text + "\n\n한 문장으로 이 글의 주제를 말하세요.",
        }
    ]
    long_prompt = [
        {
            "role": "user",
            "content": long_text + "\n\n한 문장으로 이 글의 주제를 말하세요.",
        }
    ]

    rows = []
    print("-- short (x2) --")
    for i in range(2):
        rows.append(summarize_run(f"short-{i+1}", chat(url, short_prompt, max_tokens=32)))

    print("-- medium --")
    rows.append(summarize_run("medium", chat(url, medium_prompt, max_tokens=32)))

    print("-- long --")
    long_r = chat(url, long_prompt, max_tokens=32)
    rows.append(summarize_run("long", long_r))

    print("-- multiturn --")
    turn1 = chat(url, long_prompt, max_tokens=32)
    rows.append(summarize_run("turn1", turn1))
    assistant = (turn1.get("content") or "").strip() or "ok"
    turn2_msgs = [
        long_prompt[0],
        {"role": "assistant", "content": assistant},
        {"role": "user", "content": "방금 답을 한 단어로 바꿔 주세요."},
    ]
    turn2 = chat(url, turn2_msgs, max_tokens=32)
    rows.append(summarize_run("turn2", turn2))

    print("-- queue 4-way --")
    q = queue_probe(url, n=4)
    print(
        f"  queue: http={q['http']} 200={q['n_200']} 503={q['n_503']} "
        f"wall={q['wall_s']:.2f}s per_req={q['per_req_wall_s']}"
    )

    vram1 = vram_used_mib()
    sclk1 = gpu_sclk()
    return {
        "name": name,
        "url": url,
        "started": started,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vram_mib_before": vram0,
        "vram_mib_after": vram1,
        "sclk_before": sclk0,
        "sclk_after": sclk1,
        "runs": rows,
        "queue": q,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    result = run_suite(args.name, args.url)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
