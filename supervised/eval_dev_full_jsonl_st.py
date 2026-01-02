#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from itertools import islice
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm


class STEncoder:
    """
    Sentence-Transformers encoder wrapper:
      - encode_queries([str]) -> torch.Tensor [B, D]
      - encode_docs([str])    -> torch.Tensor [B, D]

    Uses the model's configured pooling head (mean pooling, etc.), unlike raw HF AutoModel.
    """
    def __init__(self, model_name: str, device: str = "cuda", normalize: bool = False):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.model = SentenceTransformer(model_name, device=device)
        self.embed_dim = int(self.model.get_sentence_embedding_dimension())

    @torch.inference_mode()
    def encode_queries(self, texts: List[str]) -> torch.Tensor:
        return self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize,
        )

    @torch.inference_mode()
    def encode_docs(self, texts: List[str]) -> torch.Tensor:
        return self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize,
        )


def _ensure_parent_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


@torch.inference_mode()
def eval_dev_full_jsonl_st(
    jsonl_path: str,
    text_store,  # MsMarcoTextStore-like: .get(docid) -> str
    model: STEncoder,
    device: str = "cuda",
    doc_batch: int = 256,
    limit_queries: Optional[int] = 1000,   # default to 1000 queries
) -> Dict[str, float]:
    """
    Evaluate over the provided candidate list per query (e.g., 1k candidates),
    but only for the first `limit_queries` queries (JSONL lines).

    JSONL format per line:
      {"query": str, "pos_id": str|int, "candidates": [str|int, ...]}
    """
    n = 0
    mrr10 = 0.0
    hit1 = hit10 = hit100 = 0

    def encode_docs_batched(texts: List[str]) -> torch.Tensor:
        embs = []
        for j in range(0, len(texts), doc_batch):
            chunk = texts[j : j + doc_batch]
            de = model.encode_docs(chunk)  # [b, D] on model.device
            # ensure on the requested device for matmul (usually same already)
            if de.device.type != torch.device(device).type:
                de = de.to(device)
            embs.append(de)
        if not embs:
            return torch.empty(0, model.embed_dim, device=device)
        return torch.cat(embs, dim=0)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        iterable = f if limit_queries is None else islice(f, limit_queries)
        pbar = tqdm(
            iterable,
            total=limit_queries,
            unit="q",
            desc="Dev eval (ST full candidates)",
            dynamic_ncols=True,
        )

        for line in pbar:
            rec = json.loads(line)
            qtext = rec["query"]
            pos_id = rec["pos_id"]
            cands = rec["candidates"]

            # Put positive first, then all other candidates (deduping pos)
            other_ids = [d for d in cands if d != pos_id]
            doc_ids = [pos_id] + other_ids

            # Encode query (ST does its own tokenization + pooling)
            q = model.encode_queries([qtext])  # [1, D]
            if q.device.type != torch.device(device).type:
                q = q.to(device)

            texts = [text_store.get(d) for d in doc_ids]
            D = encode_docs_batched(texts)  # [N, D]

            scores = (q @ D.T).squeeze(0)  # [N]
            order = torch.argsort(scores, descending=True)

            # positive is index 0 in doc_ids
            rank_pos = 1 + (order == 0).nonzero(as_tuple=True)[0].item()

            if rank_pos == 1:
                hit1 += 1
            if rank_pos <= 10:
                hit10 += 1
                mrr10 += 1.0 / rank_pos
            if rank_pos <= 100:
                hit100 += 1

            n += 1

    N = float(max(n, 1))
    return {
        "N": n,
        "MRR@10": mrr10 / N,
        "Hit@1": hit1 / N,
        "Recall@10": hit10 / N,
        "Recall@100": hit100 / N,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/msmarco-bert-base-dot-v5",
        help="Sentence-Transformers model identifier.",
    )
    parser.add_argument("--jsonl", type=str, required=True, help="Candidates dev/test JSONL.")
    parser.add_argument(
        "--prebuilt_index",
        type=str,
        default="msmarco-v1-passage",
        help="Passed to MsMarcoTextStore(prebuilt_index).",
    )
    parser.add_argument("--device", type=str, default="cuda", help='"cuda", "cpu", or "auto"')
    parser.add_argument("--doc_batch", type=int, default=256)

    parser.add_argument(
        "--limit_queries",
        type=int,
        default=1000,
        help="Evaluate only the first N queries (JSONL lines). Set to -1 to disable.",
    )

    parser.add_argument(
        "--l2_normalize_scores",
        action="store_true",
        help="If set, normalize embeddings before dot product (cosine-style scoring). "
             "For *-dot-* models, you usually want this OFF.",
    )
    parser.add_argument("--out_json", type=str, default=None)

    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    limit_queries = None if args.limit_queries == -1 else args.limit_queries

    # Local import so script stays drop-in for your repo layout.
    from pyserini_dataset import MsMarcoTextStore

    print(f"Loading text store: {args.prebuilt_index}")
    store = MsMarcoTextStore(args.prebuilt_index)

    print(f"Loading SentenceTransformer model: {args.model_name}")
    model = STEncoder(
        args.model_name,
        device=device,
        normalize=bool(args.l2_normalize_scores),
    )

    print(f"Running eval on first {limit_queries if limit_queries is not None else 'ALL'} queries...")
    t0 = time.time()
    metrics = eval_dev_full_jsonl_st(
        jsonl_path=args.jsonl,
        text_store=store,
        model=model,
        device=device,
        doc_batch=args.doc_batch,
        limit_queries=limit_queries,
    )
    elapsed = time.time() - t0

    print("\n=== Dev metrics (ST full candidates) ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    if args.out_json is not None:
        _ensure_parent_dir(args.out_json)
        record = {
            "type": "eval_full_candidates_sentence_transformers",
            "timestamp": time.time(),
            "elapsed_sec": elapsed,
            "model_name": args.model_name,
            "jsonl_path": args.jsonl,
            "device": device,
            "doc_batch": args.doc_batch,
            "limit_queries": limit_queries,
            "l2_normalize_scores": bool(args.l2_normalize_scores),
            "prebuilt_index": args.prebuilt_index,
            "metrics": metrics,
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, sort_keys=True)
        print(f"\nWrote metrics JSON to: {args.out_json}")


if __name__ == "__main__":
    main()
