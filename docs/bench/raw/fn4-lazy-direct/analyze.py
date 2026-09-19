#!/usr/bin/env python3
"""Summarize FN-SPEED-4 A/B records into paired comparison tables."""
import json
import pathlib
import statistics

RAW = pathlib.Path(__file__).resolve().parent / "rec"


def load(path: pathlib.Path):
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def row(rec):
    d = rec.get("deltas", {})
    return {
        "tok": rec.get("prompt_tokens"),
        "pp": rec.get("prompt_per_second"),
        "majflt": d.get("majflt"),
        "read_mib": (d.get("io_read_bytes") or 0) / 1048576,
    }


def pair(name, a_path, b_path):
    a, b = load(a_path), load(b_path)
    if not a or not b:
        return f"| {name} | - | - | - | (missing: {a_path.name}={bool(a)}, {b_path.name}={bool(b)}) |"
    ra, rb = row(a), row(b)
    dp = (rb["pp"] / ra["pp"] - 1) * 100
    return (
        f"| {name} | {ra['tok']} | {ra['pp']:.1f} | {rb['pp']:.1f} | {dp:+.1f}% "
        f"| {ra['majflt']} | {rb['majflt']} | {ra['read_mib']:.1f} | {rb['read_mib']:.1f} |"
    )


HEAD = ("| prompt | tokens | on t/s | on-direct t/s | Δ | on majflt | direct majflt "
        "| on MiB | direct MiB |\n|--|--:|--:|--:|--:|--:|--:|--:|--:|")


def section(title, pairs):
    print(f"\n### {title}\n")
    print(HEAD)
    for nm, ap, bp in pairs:
        print(pair(nm, ap, bp))
    ps = []
    for nm, ap, bp in pairs:
        a, b = load(ap), load(bp)
        if a and b:
            ps.append(row(b)["pp"] / row(a)["pp"] - 1)
    if ps:
        print(f"\npaired Δ mean = {statistics.mean(ps)*100:+.1f}%, median = {statistics.median(ps)*100:+.1f}%, n={len(ps)}")


section("Cold 80k (drop_caches + restart)", [
    (f"cold_{i}", RAW / f"on/cold_{i}.rec.json", RAW / f"ondirect/cold_{i}.rec.json")
    for i in (1, 2, 3)
])
section("Warm 80k (same server after cold reps)", [
    (f"warm_{i}", RAW / f"on/warm_{i}.rec.json", RAW / f"ondirect/warm_{i}.rec.json")
    for i in (1, 2, 3)
])
section("Warm ~15k (fresh server, warmup then t1..t5; on first)", [
    (nm, RAW / f"warm_on/{nm}.rec.json", RAW / f"warm_ondirect/{nm}.rec.json")
    for nm in ("wu", "t1", "t2", "t3", "t4", "t5")
])
section("Warm ~15k reversed (on-direct first, then on)", [
    (nm, RAW / f"warm_on_rev/{nm}.rec.json", RAW / f"warm_ondirect_rev/{nm}.rec.json")
    for nm in ("wu", "t1", "t2", "t3", "t4", "t5")
])
