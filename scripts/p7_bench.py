#!/usr/bin/env python3
"""P7 controlled A/B: Qwen3.8-27B decode on the already-running :8083 unit.

Does not start/stop systemd. Writes JSON to stdout (and optionally --out).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API_KEY = "REDACTED_KEY1"

PROMPTS = [
    {
        "id": "smoke16",
        "max_tokens": 16,
        "content": "Reply with exactly: OK",
    },
    {
        "id": "para84",
        "max_tokens": 84,
        "content": (
            "In one short paragraph, explain what speculative decoding is "
            "and why it can raise tokens per second on a single GPU."
        ),
    },
    {
        "id": "mid200",
        "max_tokens": 200,
        "content": (
            "Write a detailed technical paragraph (about 180 words) on how "
            "flash attention reduces HBM traffic during transformer decode. "
            "Do not use a bullet list."
        ),
    },
    {
        "id": "long1024",
        "max_tokens": 1024,
        "content": (
            "Write a long technical essay (at least 800 words) on speculative "
            "decoding, flash attention, and KV-cache traffic on a single GPU. "
            "Use only paragraphs, no bullet lists."
        ),
    },
]


def rocm_snapshot() -> dict:
    out: dict = {}
    try:
        text = subprocess.check_output(
            ["/opt/rocm/bin/rocm-smi", "--showtemp", "--showpower", "--showclocks"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    for line in text.splitlines():
        low = line.lower()
        if "junction" in low:
            out["junction"] = line.split(":")[-1].strip()
        elif "sensor edge" in low:
            out["edge"] = line.split(":")[-1].strip()
        elif "sclk clock level" in low:
            out["sclk"] = line.split(":", 1)[-1].strip()
        elif "current socket graphics package power" in low:
            out["power_w"] = line.split(":")[-1].strip()
    return out


def metrics_text(base: str) -> str | None:
    try:
        req = urllib.request.Request(f"{base}/metrics")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return None


def parse_golbang_drafts(text: str | None) -> dict:
    if not text:
        return {}
    out = {}
    for line in text.splitlines():
        if line.startswith("golbang_draft_tokens_total "):
            out["draft_tokens_total"] = int(float(line.split()[1]))
        elif line.startswith("golbang_draft_accepted_total "):
            out["draft_accepted_total"] = int(float(line.split()[1]))
    return out


def chat(base: str, content: str, max_tokens: int, model: str = "qwen3.8-27b-q6", timeout: float = 600.0) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
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
    except urllib.error.HTTPError as e:
        wall = time.perf_counter() - t0
        return {
            "ok": False,
            "http": e.code,
            "wall_s": wall,
            "error": e.read().decode("utf-8", errors="replace")[:600],
        }
    wall = time.perf_counter() - t0
    obj = json.loads(raw)
    choice = (obj.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = obj.get("usage") or {}
    timings = obj.get("timings") or {}
    content_out = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    return {
        "ok": True,
        "http": status,
        "wall_s": wall,
        "finish_reason": choice.get("finish_reason"),
        "content": content_out,
        "reasoning": reasoning,
        "content_n": len(content_out),
        "reasoning_n": len(reasoning),
        "usage": usage,
        "timings": timings,
        "prompt_n": timings.get("prompt_n", usage.get("prompt_tokens")),
        "predicted_n": timings.get("predicted_n", usage.get("completion_tokens")),
        "cache_n": timings.get("cache_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "predicted_ms": timings.get("predicted_ms"),
        "prompt_per_second": timings.get("prompt_per_second"),
        "predicted_per_second": timings.get("predicted_per_second"),
        "draft_n": timings.get("draft_n"),
        "draft_n_accepted": timings.get("draft_n_accepted"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8083")
    ap.add_argument("--model", default="qwen3.8-27b-q6", help="model name for chat requests")
    ap.add_argument("--label", required=True, help="e.g. llama-pre, golbang-pre")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--sleep", type=float, default=8.0, help="cool-down between cases")
    ap.add_argument(
        "--only",
        default="",
        help="comma-separated case ids (default: all). e.g. para84,long1024",
    )
    args = ap.parse_args()
    wanted = {s.strip() for s in args.only.split(",") if s.strip()}
    cases = [c for c in PROMPTS if not wanted or c["id"] in wanted]
    if not cases:
        raise SystemExit(f"no cases match --only {args.only!r}")

    report = {
        "label": args.label,
        "when": datetime.now(timezone.utc).isoformat(),
        "base": args.base,
        "model": args.model,
        "gpu_before": rocm_snapshot(),
        "cases": [],
    }
    drafts0 = parse_golbang_drafts(metrics_text(args.base))
    report["metrics_before"] = drafts0

    for i, case in enumerate(cases):
        if i:
            time.sleep(args.sleep)
        gpu0 = rocm_snapshot()
        m0 = parse_golbang_drafts(metrics_text(args.base))
        r = chat(args.base, case["content"], case["max_tokens"], model=args.model)
        gpu1 = rocm_snapshot()
        m1 = parse_golbang_drafts(metrics_text(args.base))
        delta = {}
        if m0 and m1:
            for k in ("draft_tokens_total", "draft_accepted_total"):
                if k in m0 and k in m1:
                    delta[k] = m1[k] - m0[k]
        row = {
            "id": case["id"],
            "max_tokens": case["max_tokens"],
            "gpu_before": gpu0,
            "gpu_after": gpu1,
            "metrics_delta": delta,
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
                "draft_n",
                "draft_n_accepted",
                "content_n",
                "reasoning_n",
                "content",
                "reasoning",
                "error",
            )},
        }
        report["cases"].append(row)
        tps = row.get("predicted_per_second")
        print(
            f"[{args.label} {case['id']}] ok={row.get('ok')} "
            f"pred_n={row.get('predicted_n')} decode={tps} t/s "
            f"draft={row.get('draft_n')}/{row.get('draft_n_accepted')} "
            f"metrics_delta={delta} "
            f"sclk={gpu1.get('sclk')} junc={gpu1.get('junction')} "
            f"power={gpu1.get('power_w')}",
            flush=True,
        )

    report["gpu_after"] = rocm_snapshot()
    report["metrics_after"] = parse_golbang_drafts(metrics_text(args.base))
    blob = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(blob + "\n")
        print(f"wrote {args.out}", flush=True)
    else:
        print(blob)


if __name__ == "__main__":
    main()
