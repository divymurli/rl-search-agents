# train_infonce.py
from __future__ import annotations
import os, math, json, argparse, time
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup, AdamW

from pyserini_dataset import MsMarcoCandidatesIndexedDataset, MsMarcoTextStore
from collators import BiEncoderInBatchCollator  # InfoNCE collator
from dual_encoder import DualEncoder

# ----------------- Loss -----------------

def infonce_inbatch(q_emb: torch.Tensor, d_emb: torch.Tensor, temperature: float = 0.05):
    """
    q_emb: [B, D]; d_emb: [B, D]  (positives aligned by index; other docs act as negatives)
    """
    logits = (q_emb @ d_emb.t()) / temperature  # [B, B]
    labels = torch.arange(q_emb.size(0), device=q_emb.device)
    return F.cross_entropy(logits, labels)

# ----------------- Dev eval (top-100 & bottom-100 from JSONL) -----------------

def _third_slices_from_cands(cands: List[str]) -> Tuple[slice, slice, slice]:
    # cands = [pos] + negatives
    n_neg = len(cands) - 1
    n_h = n_neg // 3
    n_m = n_neg // 3
    n_e = n_neg - n_h - n_m
    h = slice(1, 1 + n_h)
    m = slice(1 + n_h, 1 + n_h + n_m)
    e = slice(1 + n_h + n_m, 1 + n_h + n_m + n_e)
    return h, m, e

def _get_hard_easy_ids(rec: Dict, take_hard=100, take_easy=100) -> Tuple[List[str], List[str]]:
    cands = rec["candidates"]  # [pos][hard…][mid…][easy…]
    if "meta" in rec and "neg_bucket_counts" in rec["meta"]:
        hc = rec["meta"]["neg_bucket_counts"]["hard"]
        mc = rec["meta"]["neg_bucket_counts"]["mid"]
        ec = rec["meta"]["neg_bucket_counts"]["easy"]
        h = slice(1, 1 + hc)
        m = slice(1 + hc, 1 + hc + mc)
        e = slice(1 + hc + mc, 1 + hc + mc + ec)
    else:
        h, m, e = _third_slices_from_cands(cands)
    hard_block = cands[h]
    easy_block = cands[e]
    hard_ids = hard_block[:min(take_hard, len(hard_block))]
    easy_ids = easy_block[-min(take_easy, len(easy_block)):]
    return hard_ids, easy_ids

@torch.inference_mode()
def eval_dev_top_bottom_jsonl(
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
    model.eval()
    n = 0
    mrr10 = {"hard": 0.0, "easy": 0.0, "pooled": 0.0}
    hit1 = hit10 = hit100 = 0

    def encode_docs(texts: List[str]) -> torch.Tensor:
        embs = []
        for j in range(0, len(texts), doc_batch):
            dtok = tokenizer(texts[j:j+doc_batch], return_tensors="pt", padding=True,
                             truncation=True, max_length=max_d_len).to(device)
            de = model.encode_docs(**dtok)  # [B, D]
            if l2_normalize_scores: de = F.normalize(de, dim=-1)
            embs.append(de)
        return torch.cat(embs, dim=0) if embs else torch.empty(0, model.embed_dim, device=device)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if limit_queries and n >= limit_queries: break
            rec = json.loads(line)
            qtext = rec["query"]; pos_id = rec["pos_id"]
            hard_ids, easy_ids = _get_hard_easy_ids(rec, 100, 100)

            # encode query once
            qtok = tokenizer(qtext, return_tensors="pt", truncation=True, max_length=max_q_len).to(device)
            q = model.encode_queries(**qtok)  # [1, D]
            if l2_normalize_scores: q = F.normalize(q, dim=-1)

            def rank_of_pos(doc_ids: List[str]) -> int:
                texts = [text_store.get(d) for d in doc_ids]
                D = encode_docs(texts)                             # [N, D]
                scores = (q @ D.T).squeeze(0)                      # [N]
                order = torch.argsort(scores, descending=True)
                # by construction, positive is index 0 in our list
                return 1 + (order == 0).nonzero(as_tuple=True)[0].item()

            # hard-only
            r = rank_of_pos([pos_id] + hard_ids)
            if r <= 10: mrr10["hard"] += 1.0 / r

            # easy-only
            r = rank_of_pos([pos_id] + easy_ids)
            if r <= 10: mrr10["easy"] += 1.0 / r

            # pooled (pos + hard + easy)
            pooled = [pos_id] + hard_ids + [d for d in easy_ids if d not in set(hard_ids)]
            r = rank_of_pos(pooled)
            if r == 1: hit1 += 1
            if r <= 10: hit10 += 1; mrr10["pooled"] += 1.0 / r
            if r <= 100: hit100 += 1

            n += 1

    N = float(max(n, 1))
    return {
        "N": n,
        "MRR@10/hard":   mrr10["hard"] / N,
        "MRR@10/easy":   mrr10["easy"] / N,
        "MRR@10/pooled": mrr10["pooled"] / N,
        "Hit@1/pooled":  hit1 / N,
        "Recall@10/pooled":  hit10 / N,
        "Recall@100/pooled": hit100 / N,
    }

