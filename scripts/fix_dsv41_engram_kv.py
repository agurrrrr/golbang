#!/usr/bin/env python3
"""Repair the first shard of a DeepSeek-V4.1-Flash Q2_K GGUF for the deepseek41 runtime.

Two things are wrong with GGUF files converted before the converter fix landed, and both live in
the first shard's header:

  1. the architecture keys carry a `deepseek4.` prefix instead of `deepseek41.`, so the runtime
     cannot find the engram configuration
  2. the five engram hash constants (`multipliers`, `primes`, `offsets`, `token_map`, `pad_id`)
     are absent, because gguf-py's add_array() truncated the 47-bit multipliers and a broad
     except downgraded that to a warning

The Hub's current first shard already has the `deepseek41` prefix, but still lacks the five
constants (see vcruz305/DeepSeek-V4.1-Flash-GGUF-DGX-Spark-recipe). The tensor payload is
byte-identical between the old and the new first shard, so this script takes the KV header from
the Hub copy, appends the constants computed from the original checkpoint tokenizer, and copies
the local tensor payload untouched. It never re-downloads the tensor data.

  python scripts/fix_dsv41_engram_kv.py \
      --model-dir /path/to/DeepSeek-V4.1-Flash \
      --local-shard1 /path/to/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf \
      --out /path/to/fixed/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf

Then symlink the remaining shards 00002..00007 beside --out.
"""
import argparse
import os
import struct
import sys
import urllib.request

import numpy as np

GGUF_MAGIC = b"GGUF"
T_UINT32 = 4
T_INT32 = 5
T_STRING = 8
T_ARRAY = 9
T_UINT64 = 10
FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

HUB_HEADER_URL = ("https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF/"
                  "resolve/main/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf")
HUB_HEADER_BYTES = 20 << 20


def read_header(path):
    f = open(path, "rb")
    assert f.read(4) == GGUF_MAGIC, "not a gguf"
    version, = struct.unpack("<I", f.read(4))
    n_tensors, = struct.unpack("<Q", f.read(8))
    n_kv, = struct.unpack("<Q", f.read(8))

    def rstr():
        n, = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8")

    kvs = []
    alignment = 32
    for _ in range(n_kv):
        key = rstr()
        vstart = f.tell()
        t, = struct.unpack("<I", f.read(4))
        if t == T_STRING:
            n, = struct.unpack("<Q", f.read(8))
            f.seek(n, os.SEEK_CUR)
        elif t == T_ARRAY:
            et, = struct.unpack("<I", f.read(4))
            cnt, = struct.unpack("<Q", f.read(8))
            if et == T_STRING:
                for _ in range(cnt):
                    n, = struct.unpack("<Q", f.read(8))
                    f.seek(n, os.SEEK_CUR)
            else:
                f.seek(FIXED[et] * cnt, os.SEEK_CUR)
        elif key == "general.alignment":
            alignment, = struct.unpack("<I", f.read(4))
        else:
            f.seek(FIXED[t], os.SEEK_CUR)
        vend = f.tell()
        f.seek(vstart)
        kvs.append((key, f.read(vend - vstart)))

    ti_start = f.tell()
    for _ in range(n_tensors):
        rstr()
        ndim, = struct.unpack("<I", f.read(4))
        f.seek(8 * ndim + 4 + 8, os.SEEK_CUR)
    ti_end = f.tell()
    f.seek(ti_start)
    tensor_info = f.read(ti_end - ti_start)
    data_start = (ti_end + alignment - 1) // alignment * alignment
    return version, n_tensors, kvs, tensor_info, alignment, data_start


def is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    i = 41
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def next_prime(start, seen):
    c = start + 1
    while not is_prime(c) or c in seen:
        c += 1
    return c


def build_token_map(model_dir):
    from tokenizers import Regex, normalizers
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    sentinel = "\ue000"
    norm = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])
    backend = tok.backend_tokenizer
    key_to_new, lookup = {}, [0] * len(tok)
    for tid in range(len(tok)):
        text = backend.decode([tid], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(tid)
        else:
            normalized = norm.normalize_str(text)
            key = normalized if normalized else text
        new = key_to_new.get(key)
        if new is None:
            new = len(key_to_new)
            key_to_new[key] = new
        lookup[tid] = new
    return lookup, len(key_to_new)


def build_constants(model_dir, layer_ids, max_ngram, n_heads, vocab_size, pad_raw):
    token_map, compressed = build_token_map(model_dir)
    bound = max(1, (np.iinfo(np.int64).max // compressed) // 2)
    mults = []
    for lid in layer_ids:
        rng = np.random.default_rng(10007 * lid)
        mults.extend(int(v) * 2 + 1 for v in rng.integers(0, bound, size=(max_ngram,), dtype=np.int64))
    primes, seen = [], set()
    for _ in layer_ids:
        for _ in range(max_ngram - 1):
            cur = vocab_size - 1
            for _ in range(n_heads):
                cur = next_prime(cur, seen)
                seen.add(cur)
                primes.append(cur)
    per_layer = (max_ngram - 1) * n_heads
    offsets = []
    for l in range(len(layer_ids)):
        acc = 0
        for b in range(per_layer):
            offsets.append(acc)
            acc += primes[l * per_layer + b]
    return {"multipliers": mults, "primes": primes, "offsets": offsets,
            "token_map": token_map, "pad_id": int(token_map[pad_raw]), "compressed": compressed}


def kv_uint32(v):
    return struct.pack("<I", T_UINT32) + struct.pack("<I", v)


def kv_array(elem_type, values):
    fmt = {T_INT32: "<i", T_UINT64: "<Q"}[elem_type]
    out = [struct.pack("<I", T_ARRAY), struct.pack("<I", elem_type), struct.pack("<Q", len(values))]
    out.extend(struct.pack(fmt, int(v)) for v in values)
    return b"".join(out)


def kv_entry(key, value_bytes):
    k = key.encode("utf-8")
    return struct.pack("<Q", len(k)) + k + value_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="original checkpoint, for its tokenizer and config")
    ap.add_argument("--local-shard1", required=True, help="the old-arch first shard with the tensor payload")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hub-header", help="already-downloaded Hub first-shard header (else fetched by range)")
    ap.add_argument("--arch", default="deepseek41")
    ap.add_argument("--engram-vocab", type=int, default=16_000_000)
    ap.add_argument("--engram-pad-id", type=int, default=2)
    args = ap.parse_args()

    hub = args.hub_header
    if not hub:
        hub = os.path.join(os.path.dirname(os.path.abspath(args.out)), ".hub-shard1-header.gguf")
        print(f"fetching Hub header ({HUB_HEADER_BYTES >> 20} MiB range) -> {hub}")
        req = urllib.request.Request(HUB_HEADER_URL, headers={"Range": f"bytes=0-{HUB_HEADER_BYTES - 1}"})
        with urllib.request.urlopen(req) as r, open(hub, "wb") as f:
            f.write(r.read())

    version, n_tensors, kvs, tensor_info, alignment, _ = read_header(hub)
    v2, nt2, _, _, _, local_data_start = read_header(args.local_shard1)
    assert (version, n_tensors) == (v2, nt2), "tensor headers differ"

    # the Hub header already uses the deepseek41 prefix; support the old prefix too
    kvs = [(args.arch + k[len("deepseek4."):] if k.startswith("deepseek4.") else k, v) for k, v in kvs]
    assert any(k == "general.architecture" for k, _ in kvs)

    def get_scalar(name, default=None):
        for key, raw in kvs:
            if key == name and key.startswith(args.arch + ".engram."):
                t, = struct.unpack("<I", raw[:4])
                return struct.unpack("<I" if t == T_UINT32 else "<i", raw[4:8])[0]
        return default

    layer_ids = None
    for key, raw in kvs:
        if key == f"{args.arch}.engram.layer_ids":
            et, = struct.unpack("<I", raw[4:8])
            cnt, = struct.unpack("<Q", raw[8:16])
            fmt = {T_INT32: "<i", T_UINT32: "<I", T_UINT64: "<Q"}[et]
            sz = FIXED[et]
            layer_ids = [struct.unpack(fmt, raw[16 + i * sz:16 + (i + 1) * sz])[0] for i in range(cnt)]
    n_heads = get_scalar(f"{args.arch}.engram.head_count")
    max_ngram = get_scalar(f"{args.arch}.engram.max_ngram_size")
    if layer_ids is None or n_heads is None or max_ngram is None:
        sys.exit("could not read the engram layer ids, head count or ngram size from the header")

    print(f"  arch={args.arch} layer_ids={layer_ids} heads={n_heads} max_ngram={max_ngram}")
    const = build_constants(args.model_dir, layer_ids, max_ngram, n_heads,
                            args.engram_vocab, args.engram_pad_id)
    print(f"  compressed vocab {const['compressed']}, token map {len(const['token_map'])}, "
          f"{len(const['primes'])} primes, pad_id {const['pad_id']}")

    additions = [
        (f"{args.arch}.engram.multipliers", kv_array(T_UINT64, const["multipliers"])),
        (f"{args.arch}.engram.primes",      kv_array(T_UINT64, const["primes"])),
        (f"{args.arch}.engram.offsets",     kv_array(T_UINT64, const["offsets"])),
        (f"{args.arch}.engram.token_map",   kv_array(T_INT32,  const["token_map"])),
        (f"{args.arch}.engram.pad_id",      kv_uint32(const["pad_id"])),
    ]
    existing = {k for k, _ in kvs}
    additions = [(k, v) for k, v in additions if k not in existing]
    print(f"  adding {len(additions)} keys")

    header = bytearray()
    header += GGUF_MAGIC
    header += struct.pack("<I", version)
    header += struct.pack("<Q", n_tensors)
    header += struct.pack("<Q", len(kvs) + len(additions))
    for key, raw in kvs:
        header += kv_entry(key, raw)
    for key, raw in additions:
        header += kv_entry(key, raw)
    header += tensor_info
    header += b"\x00" * ((-len(header)) % alignment)

    src_size = os.path.getsize(args.local_shard1)
    print(f"  data {(src_size - local_data_start)/1e9:.1f} GB, "
          f"new total {(len(header) + src_size - local_data_start)/1e9:.1f} GB")
    with open(args.local_shard1, "rb") as f, open(args.out, "wb") as out:
        f.seek(local_data_start)
        out.write(header)
        while True:
            chunk = f.read(64 << 20)
            if not chunk:
                break
            out.write(chunk)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
