#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build fixed candidate lists for MS MARCO (BEIR namespace):

For each query, output 1 positive + N negatives (default 999), sampled stratified by
BM25 rank buckets (hard/mid/easy).

- Pyserini BM25 with batched retrieval (multi-threaded)
- Optional RM3 backoff, also batched (only for underfilled queries)
- IRDS/HF loaders for queries + qrels (robust to config quirks)
- Skips/Logs pathological queries (insufficient uniques) instead of crashing

Examples:
python build_msmarco_beir_candidates.py \
  --irds-key irds/beir_msmarco_train \
  --index-mode prebuilt --index msmarco-v1-passage \
  --out data/candidates_train_1k_rm3_backoff.jsonl \
  --skip-log logs/skipped_train.jsonl \
  --topk 5000 --n-negs 999 --hard-cut 400 --mid-cut 1000 \
  --batch-size 256 --threads 16 --rm3-backoff

python build_msmarco_beir_candidates.py \       
  --irds-key irds/beir_msmarco_dev \  
  --index-mode prebuilt --index msmarco-v1-passage \
  --out data/candidates_dev_1k_rm3_backoff.jsonl \                         
  --skip-log logs/skipped_dev.jsonl \
  --topk 5000 --n-negs 999 --hard-cut 400 --mid-cut 1000 \
  --rm3-backoff --batch-size 256 --threads 16
"""

from __future__ import annotations

import argparse, json, os, sys, time, random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from tqdm import tqdm

try:
    from datasets import load_dataset, DatasetDict
except Exception:
    print("ERROR: This script requires datasets (HuggingFace). pip install datasets", file=sys.stderr)
    raise

try:
    from pyserini.search.lucene import LuceneSearcher
except Exception:
    print("ERROR: This script requires pyserini. pip install pyserini", file=sys.stderr)
    raise


# -------------------- Utils --------------------

def dedup_preserve_order(seq: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def chunked(seq: List, n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


class SkipLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w") as f:
                pass

    def log(self, qid: str, reason: str, stats: Dict):
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "qid": qid, "reason": reason}
        for k, v in (stats or {}).items():
            rec[f"stat_{k}"] = v
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())


def round_counts(total: int, ratios: Tuple[float, float, float]) -> List[int]:
    raw = [r * total for r in ratios]
    base = [int(x) for x in raw]
    rem = total - sum(base)
    fracs = sorted([(raw[i] - base[i], i) for i in range(3)], reverse=True)
    # add back in the remainder equally across each bucket
    for k in range(rem):
        base[fracs[k % 3][1]] += 1
    return base


def stratified_sample_negs(
    ranked_docids: Sequence[str],
    pos_id: str,
    *,
    n_negs: int = 999,
    hard_cut: int = 200,
    mid_cut: int = 1000,
    ratios: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    seed: Optional[int] = 42,
    on_shortfall: str = "skip",  # "skip" | "take_all" | "raise"
) -> Optional[Tuple[List[str], int]]:
    rnd = random.Random(seed) if seed is not None else random
    ranked = dedup_preserve_order(ranked_docids)

    ranked_wo_pos = [d for d in ranked if d != pos_id]
    hard = ranked_wo_pos[:hard_cut]
    mid = ranked_wo_pos[hard_cut:mid_cut]
    easy = ranked_wo_pos[mid_cut:]

    want = round_counts(n_negs, ratios)
    chosen: List[str] = []

    for bucket, k in zip([hard, mid, easy], want):
        take = min(k, len(bucket))
        if take > 0:
            chosen.extend(rnd.sample(bucket, take))

    deficit = n_negs - len(chosen)
    if deficit > 0:
        leftovers = [d for d in hard + mid + easy if d not in set(chosen)]
        take = min(deficit, len(leftovers))
        if take > 0:
            chosen.extend(rnd.sample(leftovers, take))

    deficit = n_negs - len(chosen)
    if deficit > 0:
        if on_shortfall == "skip":
            return None
        if on_shortfall == "take_all":
            return dedup_preserve_order(chosen)
        raise ValueError(
            f"Not enough negatives to sample {n_negs} (available after removing pos = {len(ranked_wo_pos)})."
        )

    chosen = dedup_preserve_order(chosen)
    assert len(chosen) == n_negs
    return chosen


# -------------------- Data loading --------------------

def _split_from_key(irds_key: str) -> Optional[str]:
    key = irds_key.lower()
    if key.endswith("_train") or ":train" in key:
        return "train"
    if key.endswith("_dev") or ":dev" in key or key.endswith("_validation"):
        return "dev"
    if key.endswith("_test") or ":test" in key:
        return "test"
    return None


def _load_irds_resource(irds_key: str, resource: str, split_hint: Optional[str]):
    # 1) config + split
    try:
        if split_hint is not None:
            return load_dataset(irds_key, name=resource, split=split_hint)
    except Exception:
        pass
    # 2) config only, pick a split
    try:
        ds_any = load_dataset(irds_key, name=resource)
        if isinstance(ds_any, DatasetDict):
            if split_hint and split_hint in ds_any:
                return ds_any[split_hint]
            if len(ds_any.keys()) == 1:
                return next(iter(ds_any.values()))
            for cand in ["train", "dev", "validation", "test"]:
                if cand in ds_any:
                    return ds_any[cand]
            return next(iter(ds_any.values()))
        return ds_any
    except Exception:
        pass
    # 3) split == resource
    try:
        return load_dataset(irds_key, split=resource)
    except Exception as e3:
        raise RuntimeError(f"Failed to load IRDS '{irds_key}' resource '{resource}': {e3}")


def load_queries_qrels_irds(irds_key: str, pos_policy: str = "highest_score"):
    split_hint = _split_from_key(irds_key)
    q_ds = _load_irds_resource(irds_key, resource="queries", split_hint=split_hint)
    r_ds = _load_irds_resource(irds_key, resource="qrels", split_hint=split_hint)

    def _get(obj, *names):
        for n in names:
            if n in obj:
                return obj[n]
        raise KeyError(f"None of {names} present; keys: {list(obj.keys())}")

    queries = [
        {"query_id": _get(x, "query_id", "id", "qid"), "text": _get(x, "text", "query", "title")}
        for x in q_ds
    ]

    tmp: Dict[str, List[Tuple[str, float]]] = {}
    for row in r_ds:
        qid = _get(row, "query-id", "query_id", "qid")
        did = _get(row, "corpus-id", "corpus_id", "doc_id", "doc-no", "docno")
        sc = float(row["score"]) if "score" in row else 1.0
        tmp.setdefault(qid, []).append((did, sc))

    qid2pos: Dict[str, str] = {}
    for qid, pairs in tmp.items():
        if pos_policy == "highest_score":
            did = sorted(pairs, key=lambda t: (-t[1], t[0]))[0][0]
        else:
            did = pairs[0][0]
        qid2pos[qid] = did

    return queries, qid2pos


# -------------------- Retrieval --------------------

def make_searcher(index_mode: str, index: str, k1: float, b: float) -> LuceneSearcher:
    if index_mode == "prebuilt":
        s = LuceneSearcher.from_prebuilt_index(index)
    elif index_mode == "dir":
        s = LuceneSearcher(index)
    else:
        raise ValueError("--index-mode must be 'prebuilt' or 'dir'")
    s.set_bm25(k1=k1, b=b)
    return s


def make_rm3_searcher(index_mode: str, index: str, k1: float, b: float, fb_terms: int, fb_docs: int, alpha: float) -> LuceneSearcher:
    if index_mode == "prebuilt":
        back = LuceneSearcher.from_prebuilt_index(index)
    elif index_mode == "dir":
        back = LuceneSearcher(index)
    else:
        raise ValueError("--index-mode must be 'prebuilt' or 'dir'")
    back.set_bm25(k1=k1, b=b)
    back.set_rm3(fb_terms=fb_terms, fb_docs=fb_docs, original_query_weight=alpha)
    return back


def normalize_id(docid: str, mode: str) -> str:
    if mode == "strip_msmarco_passage_prefix" and docid.startswith("msmarco_passage_"):
        return docid.split("_")[-1]
    return docid


# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--irds-key", type=str, required=True)
    ap.add_argument("--index-mode", type=str, choices=["prebuilt", "dir"], required=True)
    ap.add_argument("--index", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--skip-log", type=str, required=True)

    ap.add_argument("--topk", type=int, default=5000)
    ap.add_argument("--n-negs", type=int, default=999)
    ap.add_argument("--hard-cut", type=int, default=200)
    ap.add_argument("--mid-cut", type=int, default=1000)

    ap.add_argument("--ratio-hard", type=float, default=0.34)
    ap.add_argument("--ratio-mid", type=float, default=0.33)
    ap.add_argument("--ratio-easy", type=float, default=0.33)

    ap.add_argument("--k1", type=float, default=0.9)
    ap.add_argument("--b", type=float, default=0.4)

    ap.add_argument("--rm3-backoff", action="store_true")
    ap.add_argument("--rm3-k", type=int, default=20000)
    ap.add_argument("--rm3-fb-terms", type=int, default=10)
    ap.add_argument("--rm3-fb-docs", type=int, default=10)
    ap.add_argument("--rm3-alpha", type=float, default=0.5)

    ap.add_argument("--id-normalize", type=str, default="identity",
                    choices=["identity", "strip_msmarco_passage_prefix"])
    ap.add_argument("--pos-policy", type=str, default="highest_score", choices=["highest_score", "first"])

    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--flush-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # normalize ratios
    s = args.ratio_hard + args.ratio_mid + args.ratio_easy
    if abs(s - 1.0) > 1e-6:
        args.ratio_hard /= s
        args.ratio_mid  /= s
        args.ratio_easy /= s

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(out_path, "w")
    skiplogger = SkipLogger(args.skip_log)

    # Load queries + qrels
    print(f"Loading IRDS split: {args.irds_key}")
    queries, qid2pos = load_queries_qrels_irds(args.irds_key, pos_policy=args.pos_policy)
    print(f"Loaded {len(queries)} queries; {len(qid2pos)} qrels entries.")

    # Searchers
    searcher = make_searcher(args.index_mode, args.index, args.k1, args.b)
    rm3_searcher = None
    if args.rm3_backoff:
        rm3_searcher = make_rm3_searcher(
            args.index_mode, args.index, args.k1, args.b,
            args.rm3_fb_terms, args.rm3_fb_docs, args.rm3_alpha
        )
        print("RM3 backoff enabled.")

    ratios = (args.ratio_hard, args.ratio_mid, args.ratio_easy)
    n_written = 0
    n_skipped = 0

    # Build easy lookups
    qid2text = {q["query_id"]: q["text"] for q in queries}
    qid_list = [q["query_id"] for q in queries]

    # Batched retrieval
    for chunk_qids in tqdm(list(chunked(qid_list, args.batch_size)), desc="Building candidate lists (batched)"):
        chunk_texts = [qid2text[qid] for qid in chunk_qids]

        # 1) BM25 batch
        bm25_results: Dict[str, List] = searcher.batch_search(
            queries=chunk_texts, qids=chunk_qids, k=args.topk, threads=args.threads
        )

        # normalize ids
        bm25_ids: Dict[str, List[str]] = {
            qid: [normalize_id(h.docid, args.id_normalize) for h in (bm25_results.get(qid) or [])]
            for qid in chunk_qids
        }

        # 2) Decide which qids need RM3 backoff
        need_rm3 = []
        for qid in chunk_qids:
            pos = qid2pos.get(qid)
            if pos is None:
                continue
            ids = bm25_ids.get(qid, [])
            uniq = len(set(ids))
            if uniq < (args.n_negs + 1):
                need_rm3.append(qid)

        rm3_ids: Dict[str, List[str]] = {}
        if need_rm3 and rm3_searcher is not None:
            rm3_texts = [qid2text[qid] for qid in need_rm3]
            rm3_results = rm3_searcher.batch_search(
                queries=rm3_texts, qids=need_rm3, k=args.rm3_k, threads=args.threads
            )
            rm3_ids = {
                qid: [normalize_id(h.docid, args.id_normalize) for h in (rm3_results.get(qid) or [])]
                for qid in need_rm3
            }

        # 3) Build outputs per qid in this chunk
        for qid in chunk_qids:
            pos_id = qid2pos.get(qid)
            if pos_id is None:
                skiplogger.log(qid, reason="no_positive_label", stats={})
                n_skipped += 1
                continue

            pos_norm = normalize_id(pos_id, args.id_normalize)

            try:
                raw_ids = bm25_ids.get(qid, [])
                used_ids = raw_ids
                used_from_backoff = 0

                if qid in rm3_ids:
                    used_ids = dedup_preserve_order(raw_ids + rm3_ids[qid])
                    used_from_backoff = 1

                uniq = len(set(used_ids))
                stats = {
                    "topk": args.topk,
                    "hits": len(raw_ids),
                    "unique": uniq,
                    "pos_in_hits": pos_norm in set(raw_ids),
                    "used_from_backoff": used_from_backoff,
                }

                ranked = used_ids

                ## additional bit
                undup_ranked =  dedup_preserve_order(ranked)
                if pos_norm in undup_ranked:
                    pos_rank = undup_ranked.index(pos_norm) + 1
                ## additional bit

                if pos_norm not in set(ranked):
                    ranked = [pos_norm] + ranked

                sample = stratified_sample_negs(
                    ranked,
                    pos_norm,
                    n_negs=args.n_negs,
                    hard_cut=args.hard_cut,
                    mid_cut=args.mid_cut,
                    ratios=ratios,
                    seed=args.seed,
                    on_shortfall="skip",
                )
                if sample is None:
                    skiplogger.log(qid, reason="insufficient_candidates", stats=stats)
                    n_skipped += 1
                    continue

                negs = sample

                candidates = [pos_norm] + negs
                labels = [1] + [0] * len(negs)

                rec = {
                    "query_id": qid,
                    "query": qid2text[qid],
                    "pos_id": pos_norm,
                    "candidates": candidates,
                    "labels": labels,
                    "pos_rank_in_topk": pos_rank,
                }
                out_f.write(json.dumps(rec) + "\n")
                n_written += 1

            except Exception as e:
                skiplogger.log(qid, reason="exception", stats={"error": str(e)})
                n_skipped += 1
                continue

            if (n_written + n_skipped) % args.flush_every == 0:
                out_f.flush()

    out_f.flush()
    out_f.close()

    print(f"Done. Wrote {n_written} candidates; skipped {n_skipped}.")
    print(f"Candidates: {out_path}")
    print(f"Skipped log: {args.skip_log}")


if __name__ == "__main__":
    main()
