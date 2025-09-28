#!/usr/bin/env python3
# split_jsonl_by_hash.py
"""
Deterministically split a large JSONL of MS MARCO candidate records into train/dev
*without* loading the file into memory.

Strategy: single pass; compute md5(query_id) and send line to dev if
(int(md5, 16) % denom) < numer. This gives an exact proportion in expectation,
and it’s stable & reproducible.

Usage examples:
  # ~1% dev (by hash), 99% train
  python split_jsonl_by_hash.py --inp big.jsonl --out-train train.jsonl --out-dev dev.jsonl --dev-rate 0.01

  # 5k dev (exact count) via reservoir sampling (two-pass write, still streaming)
  python split_jsonl_by_hash.py --inp big.jsonl --out-train train.jsonl --out-dev dev.jsonl --dev-count 5000
"""
import argparse, json, hashlib, os, sys, time, random

def dev_by_hash(qid: str, numer: int, denom: int) -> bool:
    h = hashlib.md5(qid.encode("utf-8")).hexdigest()
    v = int(h, 16) % denom
    return v < numer

def split_by_hash(inp, out_train, out_dev, dev_rate: float):
    numer = int(round(dev_rate * 10_000))
    denom = 10_000
    n = n_dev = 0
    t0 = time.time()
    for line in inp:
        if not line.strip():
            continue
        obj = json.loads(line)
        qid = obj.get("query_id") or obj.get("qid") or obj.get("id")
        if qid is None:
            # if somehow missing, shove to train
            out_train.write(line)
            n += 1
            continue
        if dev_by_hash(qid, numer, denom):
            out_dev.write(line)
            n_dev += 1
        else:
            out_train.write(line)
        n += 1
        if n % 50_000 == 0:
            dt = time.time() - t0
            print(f"[{time.strftime('%H:%M:%S')}] processed {n:,} lines "
                  f"({n_dev:,} → dev, {n-n_dev:,} → train) at {n/max(dt,1e-9):.0f} lps", file=sys.stderr)
    return n, n_dev

def split_by_reservoir(inp_path: str, out_train_path: str, out_dev_path: str, dev_count: int):
    # Pass 1: reservoir sample qids (keeps only dev_count items in RAM)
    print(f"Reservoir sampling {dev_count} dev examples...", file=sys.stderr)
    R = []
    with open(inp_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = obj.get("query_id") or obj.get("qid") or obj.get("id")
            if qid is None:
                continue
            if len(R) < dev_count:
                R.append(qid)
            else:
                j = random.randint(1, i)
                if j <= dev_count:
                    R[j-1] = qid
            if i % 100_000 == 0:
                print(f"[pass1] seen {i:,} lines", file=sys.stderr)
    dev_set = set(R)

    # Pass 2: stream and write to train/dev by membership
    print("Writing train/dev...", file=sys.stderr)
    n = n_dev = 0
    with open(inp_path, "r", encoding="utf-8") as f, \
         open(out_train_path, "w", encoding="utf-8") as tr, \
         open(out_dev_path, "w", encoding="utf-8") as dv:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = obj.get("query_id") or obj.get("qid") or obj.get("id")
            if qid in dev_set:
                dv.write(line); n_dev += 1
            else:
                tr.write(line)
            n += 1
            if n % 100_000 == 0:
                print(f"[pass2] wrote {n:,} ({n_dev:,} dev)", file=sys.stderr)
    return n, n_dev

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="Input JSONL")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-dev", required=True)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--dev-rate", type=float, help="Fraction in dev, e.g., 0.01 for 1%")
    grp.add_argument("--dev-count", type=int, help="Exact dev size (uses reservoir; two-pass)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_train)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_dev)) or ".", exist_ok=True)

    if args.dev_rate is not None:
        with open(args.inp, "r", encoding="utf-8") as f, \
             open(args.out_train, "w", encoding="utf-8") as tr, \
             open(args.out_dev, "w", encoding="utf-8") as dv:
            n, n_dev = split_by_hash(f, tr, dv, args.dev_rate)
    else:
        n, n_dev = split_by_reservoir(args.inp, args.out_train, args.out_dev, args.dev_count)

    print(f"Done. total={n:,}, dev={n_dev:,} ({(n_dev/max(n,1))*100:.2f}%), train={n-n_dev:,}", file=sys.stderr)

if __name__ == "__main__":
    main()
