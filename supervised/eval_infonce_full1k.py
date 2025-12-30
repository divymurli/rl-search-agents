from __future__ import annotations
import json
import argparse
import time
import os
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from transformers import AutoTokenizer

from pyserini_dataset import MsMarcoTextStore
from dual_encoder import DualEncoder


@torch.inference_mode()
def eval_dev_full_jsonl(
    jsonl_path: str,
    text_store: MsMarcoTextStore,
    model: DualEncoder,
    tokenizer,
    device: str = "cuda",
    max_q_len: int = 64,
    max_d_len: int = 256,
    doc_batch: int = 256,
    limit_queries: Optional[int] = None,
    l2_normalize_scores: bool = False,
) -> Dict[str, float]:
    """
    Evaluate over the full candidate list per query (e.g., 1k candidates),
    not just top/bottom-100 buckets.
    """
    model.eval()
    n = 0
    mrr10 = 0.0
    hit1 = hit10 = hit100 = 0

    def encode_docs(texts: List[str]) -> torch.Tensor:
        embs = []
        for j in range(0, len(texts), doc_batch):
            dtok = tokenizer(
                texts[j:j + doc_batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_d_len,
            ).to(device)
            de = model.encode_docs(**dtok)  # [B, D]
            if l2_normalize_scores:
                de = F.normalize(de, dim=-1)
            embs.append(de)
        return torch.cat(embs, dim=0) if embs else torch.empty(0, model.embed_dim, device=device)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        pbar = tqdm(
            total=limit_queries,
            unit="q",
            desc="Dev eval (full candidates)",
            dynamic_ncols=True,
        )
        for line in f:
            if limit_queries and n >= limit_queries:
                break
            rec = json.loads(line)
            qtext = rec["query"]
            pos_id = rec["pos_id"]
            cands = rec["candidates"]  # full candidate list (e.g., 1k)

            # Put positive first, then all other candidates (deduping pos)
            other_ids = [d for d in cands if d != pos_id]
            doc_ids = [pos_id] + other_ids

            # encode query once
            qtok = tokenizer(
                qtext,
                return_tensors="pt",
                truncation=True,
                max_length=max_q_len,
            ).to(device)
            q = model.encode_queries(**qtok)  # [1, D]
            if l2_normalize_scores:
                q = F.normalize(q, dim=-1)

            texts = [text_store.get(d) for d in doc_ids]
            D = encode_docs(texts)  # [N, D]
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
            pbar.update(1)

    pbar.close()

    N = float(max(n, 1))
    return {
        "N": n,
        "MRR@10": mrr10 / N,
        "Hit@1": hit1 / N,
        "Recall@10": hit10 / N,
        "Recall@100": hit100 / N,
    }


# Defaults (can be overridden via CLI)
DEFAULT_CKPT = "/home/ubuntu/rl-search-agents/ckpts/infonce_group_curriculum_groupk_4/step30000.pt"
DEFAULT_JSONL = "/home/ubuntu/rl-search-agents/data/rl-search-datasets/candidates_dev_1k_rm3_backoff.jsonl"


def _ensure_parent_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--jsonl", type=str, default=DEFAULT_JSONL)
    parser.add_argument("--device", type=str, default="cuda", help='"cuda", "cpu", or "auto"')

    parser.add_argument(
        "--out_json",
        type=str,
        default=None,
        help="If set, write metrics + provenance to this JSON file.",
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Loading checkpoint from {args.ckpt}")
    ckpt_obj = torch.load(args.ckpt, map_location=device)

    ckpt_args = ckpt_obj.get("args", {})  # vars(args) from training

    model_name = ckpt_args["model_name"]
    proj_dim = ckpt_args["proj_dim"]
    normalize = ckpt_args["normalize"]
    max_q_len = ckpt_args["max_q_len"]
    max_d_len = ckpt_args["max_d_len"]
    dev_limit = ckpt_args.get("dev_limit", None)
    prebuilt_index = ckpt_args.get("prebuilt_index", "msmarco-v1-passage")

    print(f"Loaded training args from ckpt:")
    print(json.dumps(ckpt_args, indent=2))

    # Text store + tokenizer + model
    store = MsMarcoTextStore(prebuilt_index)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    model = DualEncoder(
        model_name,
        proj_dim=proj_dim,
        normalize=normalize,
    ).to(device)

    model.load_state_dict(ckpt_obj["model"])
    model.eval()

    print("Running dev eval over full candidate lists...")
    t0 = time.time()
    metrics = eval_dev_full_jsonl(
        args.jsonl,
        store,
        model,
        tokenizer,
        device=device,
        max_q_len=max_q_len,
        max_d_len=max_d_len,
        doc_batch=256,
        limit_queries=dev_limit,
        l2_normalize_scores=normalize,
    )
    elapsed = time.time() - t0

    print("\n=== Dev metrics (full candidates) ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    if args.out_json is not None:
        _ensure_parent_dir(args.out_json)
        record = {
            "type": "eval_full_candidates",
            "timestamp": time.time(),
            "elapsed_sec": elapsed,
            "ckpt_path": args.ckpt,
            "jsonl_path": args.jsonl,
            "device": device,
            "metrics": metrics,
            "ckpt_train_args": {
                # keep the full thing if you want; this subset is usually enough
                "model_name": model_name,
                "proj_dim": proj_dim,
                "normalize": normalize,
                "max_q_len": max_q_len,
                "max_d_len": max_d_len,
                "dev_limit": dev_limit,
                "prebuilt_index": prebuilt_index,
                "loss_mode": ckpt_args.get("loss_mode"),
                "margin_alpha": ckpt_args.get("margin_alpha"),
                "margin_m": ckpt_args.get("margin_m"),
                "group_k": ckpt_args.get("group_k"),
                "group_beta": ckpt_args.get("group_beta"),
                "group_temperature": ckpt_args.get("group_temperature"),
                "lr": ckpt_args.get("lr"),
                "batch_size": ckpt_args.get("batch_size"),
                "accum_steps": ckpt_args.get("accum_steps"),
            },
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, sort_keys=True)
        print(f"\nWrote metrics JSON to: {args.out_json}")


if __name__ == "__main__":
    main()
