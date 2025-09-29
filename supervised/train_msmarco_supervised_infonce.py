# train_infonce.py
from __future__ import annotations
import os, math, json, argparse, time
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup

from pyserini_dataset import MsMarcoCandidatesIndexedDataset, MsMarcoTextStore
from collators import BiEncoderInBatchCollator  # InfoNCE collator
from dual_encoder import DualEncoder

## USAGE
"""
 python supervised/train_msmarco_supervised_infonce.py \
 --train_jsonl data/rl-search-datasets/msmarco-supervised/training_data/train.jsonl \
 --dev_jsonl data/rl-search-datasets/msmarco-supervised/training_data/dev.jsonl \
 --model_name bert-base-uncased \
 --batch_size 16 --epochs 1 \
 --eval_every 1000 \
 --dev_limit 2000  \
 --prebuilt_index msmarco-v1-passage
"""

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
        pbar = tqdm(total=limit_queries, unit="q", desc="Dev eval (top/bottom-100)", dynamic_ncols=True)
        for line in f:
            if limit_queries and n >= limit_queries:
                break
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
            pbar.update(1)

    pbar.close()
    
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

# ----------------- Train scaffold -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--dev_jsonl",   required=True)
    ap.add_argument("--prebuilt_index", default="msmarco-v1-passage")
    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--proj_dim", type=int, default=768)
    ap.add_argument("--normalize", action="store_true", help="L2-normalize embeddings")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--accum_steps", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--max_q_len", type=int, default=64)
    ap.add_argument("--max_d_len", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--eval_every", type=int, default=2000)  # steps
    ap.add_argument("--dev_limit", type=int, default=2000)   # queries used in quick dev
    ap.add_argument("--outdir", default="ckpts/infonce_baseline")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    store = MsMarcoTextStore(args.prebuilt_index)
    train_ds = MsMarcoCandidatesIndexedDataset(args.train_jsonl, text_store=store, group_k=0, qid_limit=0)
    dev_path = args.dev_jsonl

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    collate = BiEncoderInBatchCollator(tok, max_q_len=args.max_q_len, max_d_len=args.max_d_len, include_margin=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate,
        drop_last=True,   # important for in-batch negatives
    )

    # Model/opt
    model = DualEncoder(args.model_name, proj_dim=args.proj_dim, normalize=args.normalize).to(device)
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * math.ceil(len(train_loader))
    warmup_steps = int(args.warmup_ratio * total_steps)
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # gradient scaling for mixed precision training
    use_cuda_amp = (device == "cuda")
    use_bf16 = use_cuda_amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # Grad scaling is only needed for FP16; BF16 doesn’t use it.
    scaler = torch.amp.GradScaler(enabled=use_cuda_amp and (amp_dtype is torch.float16))
    step = 0

    print(f"Training steps: {total_steps} | warmup: {warmup_steps} | eval_every: {args.eval_every}")

    # metrics = eval_dev_top_bottom_jsonl(
    #                 dev_path, store, model, tok,
    #                 device=device,
    #                 max_q_len=args.max_q_len, max_d_len=args.max_d_len,
    #                 doc_batch=max(128, args.batch_size),
    #                 limit_queries=args.dev_limit,
    #                 l2_normalize_scores=args.normalize,
    #             )

    # print("Starting metrics ...")
    # print(f"[DEV @ step {step}] "
    #                   f"N={metrics['N']}  "
    #                   f"MRR10(h)={metrics['MRR@10/hard']:.4f}  "
    #                   f"MRR10(e)={metrics['MRR@10/easy']:.4f}  "
    #                   f"MRR10(p)={metrics['MRR@10/pooled']:.4f}  "
    #                   f"H@1={metrics['Hit@1/pooled']:.4f}  "
    #                   f"R@10={metrics['Recall@10/pooled']:.4f}  "
    #                   f"R@100={metrics['Recall@100/pooled']:.4f}")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        for batch in train_loader:
            step += 1
            q = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k.startswith("query_")}
            d = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k.startswith("doc_")}

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda_amp):

                q_kwargs = {
                    "input_ids": q["query_input_ids"],
                    "attention_mask": q["query_attention_mask"],
                }
                if "query_token_type_ids" in q:
                    q_kwargs["token_type_ids"] = q["query_token_type_ids"]
                q_emb = model.encode_queries(**q_kwargs)

                d_kwargs = {
                    "input_ids": d["doc_input_ids"],
                    "attention_mask": d["doc_attention_mask"],
                }
                if "doc_token_type_ids" in d:
                    d_kwargs["token_type_ids"] = d["doc_token_type_ids"]
                d_emb = model.encode_docs(**d_kwargs)
                loss = infonce_inbatch(q_emb, d_emb, temperature=args.temperature) / args.accum_steps

            scaler.scale(loss).backward()

            if step % args.accum_steps == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()

            if step % 100 == 0:
                elapsed = time.time() - t0
                print(f"[ep {epoch+1}] step {step}/{total_steps}  loss {loss.item()*args.accum_steps:.4f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  {elapsed:.1f}s")

            if step % args.eval_every == 0 or step == 1:
                print("running eval ...")
                metrics = eval_dev_top_bottom_jsonl(
                    dev_path, store, model, tok,
                    device=device,
                    max_q_len=args.max_q_len, max_d_len=args.max_d_len,
                    doc_batch=max(64, args.batch_size),
                    limit_queries=args.dev_limit,
                    l2_normalize_scores=args.normalize,
                )
                print(f"[DEV @ step {step}] "
                      f"N={metrics['N']}  "
                      f"MRR10(h)={metrics['MRR@10/hard']:.4f}  "
                      f"MRR10(e)={metrics['MRR@10/easy']:.4f}  "
                      f"MRR10(p)={metrics['MRR@10/pooled']:.4f}  "
                      f"H@1={metrics['Hit@1/pooled']:.4f}  "
                      f"R@10={metrics['Recall@10/pooled']:.4f}  "
                      f"R@100={metrics['Recall@100/pooled']:.4f}")

                # lightweight checkpoint
                ckpt_path = os.path.join(args.outdir, f"step{step}.pt")
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optim.state_dict(),
                    "scheduler": sched.state_dict(),
                    "step": step,
                    "args": vars(args),
                    "metrics": metrics
                }, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

        # end epoch

    print("Done.")

if __name__ == "__main__":
    main()

    


    


