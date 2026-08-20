#!/usr/bin/env python3
"""P6: wrap golbang-server with rocprofv3 (kernel + memcpy).

Attach to a live HIP process does not emit kernel CSV on this box
(yama ptrace_scope, then rocprofiler-register assert). Start-from-scratch
wrap is the working path. Model-load copies are in the same files —
cut the last timestamp burst before ranking (see docs/bench/p6.md).

    python3 scripts/p6_rocprof.py --phase prefill --out /tmp/p6/prefill
    python3 scripts/p6_rocprof.py --phase decode  --out /tmp/p6/decode
    python3 scripts/p6_rocprof.py --phase prefix  --out /tmp/p6/prefix
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PARA = (
    "The MI50 is a gfx906 CDNA accelerator with 32 GiB of HBM2. "
    "golbang serves DeepSeek-V4-Flash IQ2_M through ggml-hip. "
)
MODEL = (
    "/home/agurrrrr/models/dsv4/UD-IQ2_M/"
    "DeepSeek-V4-Flash-0731-UD-IQ2_M-00001-of-00003.gguf"
)
SERVER = "/home/agurrrrr/code/golbang/target/release/golbang-server"


def http_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def wait_log(path: Path, needle: str, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"wrapper exited {proc.returncode} before ready")
        if path.is_file() and needle in path.read_text(errors="replace"):
            return
        time.sleep(1)
    raise SystemExit(f"timeout waiting for {needle!r} in {path}")


def chat(base: str, messages: list, max_tokens: int, timeout: float) -> dict:
    return http_json(
        base + "/v1/chat/completions",
        {
            "model": "golbang",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        },
        timeout=timeout,
    )


def slim(name: str, wall_s: float, resp: dict) -> dict:
    return {
        "name": name,
        "wall_s": wall_s,
        "usage": resp.get("usage"),
        "timings": resp.get("timings"),
        "message": (resp.get("choices") or [{}])[0].get("message"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("prefill", "decode", "prefix"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--repeats", type=int, default=110)
    args = ap.parse_args()

    os.environ["PATH"] = "/opt/rocm/bin:" + os.environ.get("PATH", "")
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.0.6"
    os.environ["ROCR_VISIBLE_DEVICES"] = "0"
    os.environ["ROCBLAS_USE_HIPBLASLT"] = "0"
    llama_bin = "/home/agurrrrr/code/local-llm/llama.cpp/build/bin"
    prev = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = llama_bin + (":" + prev if prev else "")
    os.environ["RUST_LOG"] = "info"

    args.out.mkdir(parents=True, exist_ok=True)
    server_log = args.out / "server.log"
    rocprof_log = args.out / "rocprof.log"
    server = [
        SERVER,
        "--model",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--n-gpu-layers",
        "99",
        "--n-cpu-moe",
        "32",
        "--flash-attn",
        "on",
        "--n-ctx",
        "60000",
        "--n-batch",
        "5800",
        "--n-ubatch",
        "1024",
        "--n-threads",
        "8",
        "--n-rs-seq",
        "1",
        "--n-parallel",
        "1",
        "--queue-size",
        "1",
        "--jinja",
        "--reasoning-format",
        "deepseek",
    ]
    cmd = [
        "rocprofv3",
        "--kernel-trace",
        "--memory-copy-trace",
        "--stats",
        "--summary",
        "--summary-per-domain",
        "--process-sync",
        "--output-format",
        "csv",
        "-d",
        str(args.out),
        "-o",
        args.phase,
        "--",
        *server,
    ]
    print("+", " ".join(cmd), flush=True)
    with server_log.open("w") as slog, rocprof_log.open("w") as rlog:
        proc = subprocess.Popen(cmd, stdout=slog, stderr=rlog, start_new_session=True)
    try:
        wait_log(server_log, "listening", proc, timeout=240)
        print("server listening", flush=True)
        base = f"http://127.0.0.1:{args.port}"
        user1 = (PARA * args.repeats) + "한 문장으로, 위 단락이 몇 번 반복됐는지 숫자만 답하라."
        results = []
        if args.phase == "prefill":
            t0 = time.time()
            resp = chat(base, [{"role": "user", "content": user1}], 1, 900)
            results.append(slim("prefill", time.time() - t0, resp))
        elif args.phase == "decode":
            t0 = time.time()
            resp = chat(
                base,
                [
                    {
                        "role": "user",
                        "content": "80까지의 정수를 공백으로만 나열하라. 다른 말은 하지 마라.",
                    }
                ],
                120,
                300,
            )
            results.append(slim("decode", time.time() - t0, resp))
        else:
            t0 = time.time()
            r1 = chat(base, [{"role": "user", "content": user1}], 8, 900)
            results.append(slim("prefix-turn1", time.time() - t0, r1))
            msg = (r1.get("choices") or [{}])[0].get("message") or {}
            assistant = (msg.get("content") or "") or (msg.get("reasoning_content") or "ok")
            t1 = time.time()
            r2 = chat(
                base,
                [
                    {"role": "user", "content": user1},
                    {"role": "assistant", "content": assistant},
                    {"role": "user", "content": "같은 숫자만 다시 말해."},
                ],
                1,
                300,
            )
            results.append(slim("prefix-turn2", time.time() - t1, r2))
        for item in results:
            print(json.dumps(item, ensure_ascii=False, indent=2), flush=True)
        (args.out / "requests.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2)
        )
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=15)
        print(f"wrapper exit {proc.returncode}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
