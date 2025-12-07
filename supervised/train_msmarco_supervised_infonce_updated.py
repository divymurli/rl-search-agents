# train_msmarco_supervised_infonce.py
from __future__ import annotations
import os, math, json, argparse, time
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from pyserini_dataset import MsMarcoCandidatesIndexedDataset, MsMarcoTextStore
from collators import BiEncoderInBatchCollator, BiEncoderGroupedCollator
from dual_encoder import DualEncoder

"""
USAGE EXAMPLES:

# E0: baseline InfoNCE
python supervised/train_msmarco_supervised_infonce.py \
  --train_jsonl data/.../train.jsonl \
  --dev_jsonl   data/.../dev.jsonl \
  --model_name bert-base-uncased \
  --batch_size 16 --epochs 1 \
  --eval_every 1000 \
  --dev_limit 2000 \
  --prebuilt_index msmarco-v1-passage \
  --loss_mode infonce

# E1: InfoNCE + margin (hard negative)
python supervised/train_msmarco_supervised_infonce.py \
  --train_jsonl data/.../train.jsonl \
  --dev_jsonl   data/.../dev.jsonl \
  --loss_mode infonce_margin \
  --margin_alpha 0.4 --margin_m 0.2

# E2: InfoNCE + grouped InfoNCE (mined negatives)
python supervised/train_msmarco_supervised_infonce.py \
  --train_jsonl data/.../train.jsonl \
  --dev_jsonl   data/.../dev.jsonl \
  --loss_mode infonce_grouped \
  --group_k 8
"""

# -------------------------------------------------------------------
# Losses
# -------------------------------------------------------------------

def infonce_inbatch(q_emb: torch.Tensor, d_emb: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    """
    Baseline E0: in-batch InfoNCE
    q_emb: [B, D]; d_emb: [B, D]  (positives aligned by index; other docs act as negatives)
    """
    logits = (q_emb @ d_emb.t()) / temperature  # [B, B]
    labels = torch.arange(q_emb.size(0), device=q_emb.device)
    return F.cross_entropy(logits, labels)


def pairwise_scores(q_emb: torch.Tensor, d_emb: torch.Tensor) -> torch.Tensor:
    """
    q_emb: [B, D], d_emb: [B, D]
    returns diag similarity: [B]
    """
    return (q_emb * d_emb).sum(dim=-1)


def margin_loss(
    q_emb: torch.Tensor,
    d_pos_emb: torch.Tensor,
    d_neg_emb: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """
    E1: margin term for curriculum hard negative
    margin loss: max(0, m - s(q,pos) + s(q,neg))
    q_emb, d_pos_emb, d_neg_emb: [B, D]
    """
    s_pos = pairwise_scores(q_emb, d_pos_emb)    # [B]
    s_neg = pairwise_scores(q_emb, d_neg_emb)    # [B]
    return torch.relu(margin - s_pos + s_neg).mean()


def grouped_infonce_variable(
    q_emb: torch.Tensor,
    d_all_emb: torch.Tensor,
    group_offsets: torch.Tensor,
    group_sizes: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """
    E2: grouped InfoNCE over per-query candidate pools with variable group sizes.

    Inputs:
      - q_emb: [B, D]
      - d_all_emb: [N, D] flattened docs; for each i,
           docs[group_offsets[i] : group_offsets[i] + group_sizes[i]]
           == [pos_i, neg1_i, ..., negK_i]
      - group_offsets: [B]
      - group_sizes: [B]
    """
    B, D = q_emb.shape
    losses = []

    for i in range(B):
        off = int(group_offsets[i])
        sz = int(group_sizes[i])
        group = d_all_emb[off:off + sz]         # [Gi, D], where Gi = 1 + (#negs_i)
        # positive is assumed to be at index 0
        # logits: [1, Gi]
        logits = (q_emb[i:i+1] @ group.T) / temperature
        labels = torch.zeros(1, dtype=torch.long, device=q_emb.device)
        loss_i = F.cross_entropy(logits, labels)
        losses.append(loss_i)

    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=q_emb.device)

# -------------------------------------------------------------------
# Dev eval (top-100 & bottom-100 from JSONL)
# -------------------------------------------------------------------

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
            desc="Dev eval (top/bottom-100)",
            dynamic_ncols=True,
        )
        for line in f:
            if limit_queries and n >= limit_queries:
                break
            rec = json.loads(line)
            qtext = rec["query"]
            pos_id = rec["pos_id"]
            hard_ids, easy_ids = _get_hard_easy_ids(rec, 100, 100)

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

            def rank_of_pos(doc_ids: List[str]) -> int:
                texts = [text_store.get(d) for d in doc_ids]
                D = encode_docs(texts)                             # [N, D]
                scores = (q @ D.T).squeeze(0)                      # [N]
                order = torch.argsort(scores, descending=True)
                # by construction, positive is index 0 in our list
                return 1 + (order == 0).nonzero(as_tuple=True)[0].item()

            # hard-only
            r = rank_of_pos([pos_id] + hard_ids)
            if r <= 10:
                mrr10["hard"] += 1.0 / r

            # easy-only
            r = rank_of_pos([pos_id] + easy_ids)
            if r <= 10:
                mrr10["easy"] += 1.0 / r

            # pooled (pos + hard + easy)
            pooled = [pos_id] + hard_ids + [d for d in easy_ids if d not in set(hard_ids)]
            r = rank_of_pos(pooled)
            if r == 1:
                hit1 += 1
            if r <= 10:
                hit10 += 1
                mrr10["pooled"] += 1.0 / r
            if r <= 100:
                hit100 += 1

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

# -------------------------------------------------------------------
# Logging helper
# -------------------------------------------------------------------

def log_jsonl(path: str, record: Dict):
    """Append a JSON record to a JSONL log file."""
    record.setdefault("wall_time", time.time())
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

# -------------------------------------------------------------------
# Train scaffold
# -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--dev_jsonl",   required=True)
    ap.add_argument("--prebuilt_index", default="msmarco-v1-passage")
    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--proj_dim", type=int, default=768)
    ap.add_argument("--normalize", action="store_true", help="L2-normalize embeddings")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--accum_steps", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--max_q_len", type=int, default=64)
    ap.add_argument("--max_d_len", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.05)

    # IMPORTANT: Pyserini + curriculum → safest with num_workers=0
    ap.add_argument("--num_workers", type=int, default=0)   # <<< CURRICULUM / PYSerini: changed default to 0
    ap.add_argument("--eval_every", type=int, default=2000)  # steps
    ap.add_argument("--dev_limit", type=int, default=1000)   # queries used in quick dev
    ap.add_argument("--outdir", default="ckpts/infonce_baseline")

    # loss mode & hyperparams (E0/E1/E2)
    ap.add_argument(
        "--loss_mode",
        choices=["infonce", "infonce_margin", "infonce_grouped"],
        default="infonce",
        help="E0: infonce, E1: infonce_margin, E2: infonce_grouped",
    )
    ap.add_argument("--margin_alpha", type=float, default=0.4,
                    help="Weight for margin term in infonce_margin (E1)")
    ap.add_argument("--margin_m", type=float, default=0.2,
                    help="Margin m for margin loss (E1)")
    ap.add_argument("--group_beta", type=float, default=1.0,
                    help="Weight for grouped InfoNCE term (E2)")
    ap.add_argument("--group_temperature", type=float, default=0.05,
                    help="Temperature for grouped InfoNCE (E2)")

    # how many mined negatives per query your Dataset returns for grouped mode
    ap.add_argument("--group_k", type=int, default=0,
                    help="Num mined negatives per query (passed to Dataset); relevant for grouped mode")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # log file path
    log_path = os.path.join(args.outdir, "train_log.jsonl")
    log_jsonl(log_path, {"type": "config", "args": vars(args)})

    # Data
    store = MsMarcoTextStore(args.prebuilt_index)
    train_ds = MsMarcoCandidatesIndexedDataset(
        args.train_jsonl,
        text_store=store,
        group_k=args.group_k,  # 0 for E0/E1; >0 for E2
        qid_limit=0,
    )
    dev_path = args.dev_jsonl

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # choose collator based on loss_mode
    if args.loss_mode in ["infonce", "infonce_margin"]:
        collate = BiEncoderInBatchCollator(
            tok,
            max_q_len=args.max_q_len,
            max_d_len=args.max_d_len,
            include_margin=(args.loss_mode == "infonce_margin"),
        )
    else:  # infonce_grouped
        collate = BiEncoderGroupedCollator(
            tok,
            max_q_len=args.max_q_len,
            max_d_len=args.max_d_len,
            include_margin=False,  # margin not used in E2 here
        )

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
    model = DualEncoder(
        args.model_name,
        proj_dim=args.proj_dim,
        normalize=args.normalize,
    ).to(device)
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * math.ceil(len(train_loader)) # micro steps
    total_updates = max(1, math.ceil(total_steps / args.accum_steps))
    warmup_steps = int(args.warmup_ratio * total_steps)
    sched = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # <<< CURRICULUM: tell dataset the total number of steps so it can ramp p_h/p_m/p_e
    if hasattr(train_ds, "set_total_steps"):
        print("Setting total steps for train loader")
        train_ds.set_total_steps(total_updates)

    # mixed precision
    use_cuda_amp = (device == "cuda")
    use_bf16 = use_cuda_amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler(enabled=use_cuda_amp and (amp_dtype is torch.float16))
    step = 0

    print(
        f"Training steps: {total_steps} | warmup: {warmup_steps} | "
        f"eval_every: {args.eval_every} | loss_mode: {args.loss_mode}"
    )

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        for batch in train_loader:
            step += 1

            # tie curriculum & logging to optimizer update steps
            update_step = step // args.accum_steps

            # Set the curriculum, so as to only sampler harder negatives as training
            # goes on
            if hasattr(train_ds, "set_global_step"):
                train_ds.set_global_step(update_step)

            # move query / doc tensors
            q = {k: v.to(device, non_blocking=True)
                 for k, v in batch.items() if k.startswith("query_")}
            d = {k: v.to(device, non_blocking=True)
                 for k, v in batch.items() if k.startswith("doc_")}

            group_offsets = batch.get("group_offsets")
            group_sizes = batch.get("group_sizes")

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda_amp):

                # ----- encode queries -----
                q_kwargs = {
                    "input_ids": q["query_input_ids"],
                    "attention_mask": q["query_attention_mask"],
                }
                if "query_token_type_ids" in q:
                    q_kwargs["token_type_ids"] = q["query_token_type_ids"]
                q_emb = model.encode_queries(**q_kwargs)  # [B, D]

                if args.loss_mode in ["infonce", "infonce_margin"]:
                    # ----- E0/E1: in-batch positives only -----
                    d_kwargs = {
                        "input_ids": d["doc_input_ids"],
                        "attention_mask": d["doc_attention_mask"],
                    }
                    if "doc_token_type_ids" in d:
                        d_kwargs["token_type_ids"] = d["doc_token_type_ids"]
                    d_emb = model.encode_docs(**d_kwargs)  # [B, D]

                    # E0: baseline InfoNCE
                    main_loss = infonce_inbatch(q_emb, d_emb, temperature=args.temperature)
                    total_loss = main_loss

                    # E1: InfoNCE + margin hard negative
                    if args.loss_mode == "infonce_margin":
                        if not all(k in batch for k in ["margin_input_ids", "margin_attention_mask"]):
                            raise KeyError(
                                "loss_mode=infonce_margin but batch is missing "
                                "'margin_input_ids' / 'margin_attention_mask'. "
                                "Make sure BiEncoderInBatchCollator(include_margin=True) is used."
                            )
                            # (They are created by your collator.)

                        m_ids = batch["margin_input_ids"].to(device, non_blocking=True)
                        m_attn = batch["margin_attention_mask"].to(device, non_blocking=True)
                        marg_kwargs = {
                            "input_ids": m_ids,
                            "attention_mask": m_attn,
                        }
                        # no token_type_ids in your collator, but we could add later

                        d_margin_emb = model.encode_docs(**marg_kwargs)  # [B, D]
                        m_loss = margin_loss(
                            q_emb,
                            d_pos_emb=d_emb,
                            d_neg_emb=d_margin_emb,
                            margin=args.margin_m,
                        )
                        total_loss = main_loss + args.margin_alpha * m_loss

                else:
                    # ----- E2: grouped InfoNCE mode -----
                    # docs are flattened: [sum_i Gi, L], groups defined by offsets/sizes
                    if group_offsets is None or group_sizes is None:
                        raise KeyError(
                            "loss_mode=infonce_grouped but batch is missing "
                            "'group_offsets' / 'group_sizes'. "
                            "Make sure BiEncoderGroupedCollator is used."
                        )

                    group_offsets = group_offsets.to(device)
                    group_sizes = group_sizes.to(device)

                    d_kwargs = {
                        "input_ids": d["doc_input_ids"],        # [N, L]
                        "attention_mask": d["doc_attention_mask"],
                    }
                    if "doc_token_type_ids" in d:
                        d_kwargs["token_type_ids"] = d["doc_token_type_ids"]

                    d_all_emb = model.encode_docs(**d_kwargs)  # [N, D]

                    # positives are the first doc in each group
                    pos_indices = group_offsets  # [B]
                    d_pos_emb = d_all_emb[pos_indices]  # [B, D]

                    # baseline InfoNCE over positives (in-batch negatives)
                    main_loss = infonce_inbatch(q_emb, d_pos_emb, temperature=args.temperature)

                    # grouped InfoNCE over per-query pools
                    g_loss = grouped_infonce_variable(
                        q_emb,
                        d_all_emb,
                        group_offsets=group_offsets,
                        group_sizes=group_sizes,
                        temperature=args.group_temperature,
                    )
                    total_loss = main_loss + args.group_beta * g_loss

                # scale for grad accumulation
                loss = total_loss / args.accum_steps

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
                true_loss = loss.item() * args.accum_steps  # undo accum scaling
                lr = sched.get_last_lr()[0]
                print(
                    f"[ep {epoch+1}] step {step}/{total_steps}  "
                    f"loss {true_loss:.4f}  lr {lr:.2e}  {elapsed:.1f}s"
                )
                log_jsonl(log_path, {
                    "type": "train",
                    "epoch": epoch + 1,
                    "step": step,
                    "update_step": update_step,
                    "loss": true_loss,
                    "lr": lr,
                    "elapsed_since_epoch_start": elapsed,
                    "loss_mode": args.loss_mode,
                })

            if step > 0 and step % args.eval_every == 0:
                print("running eval ...")
                print(f"[DEV @ step {step} (updates {update_step})] ...")
                metrics = eval_dev_top_bottom_jsonl(
                    dev_path, store, model, tok,
                    device=device,
                    max_q_len=args.max_q_len,
                    max_d_len=args.max_d_len,
                    doc_batch=max(64, args.batch_size),
                    limit_queries=args.dev_limit,
                    l2_normalize_scores=args.normalize,
                )
                print(
                    f"[DEV @ step {step}] "
                    f"N={metrics['N']}  "
                    f"MRR10(h)={metrics['MRR@10/hard']:.4f}  "
                    f"MRR10(e)={metrics['MRR@10/easy']:.4f}  "
                    f"MRR10(p)={metrics['MRR@10/pooled']:.4f}  "
                    f"H@1={metrics['Hit@1/pooled']:.4f}  "
                    f"R@10={metrics['Recall@10/pooled']:.4f}  "
                    f"R@100={metrics['Recall@100/pooled']:.4f}"
                )

                eval_record = {
                    "type": "eval",
                    "epoch": epoch + 1,
                    "step": step,
                    "update_step": update_step,
                    "loss_mode": args.loss_mode,
                }
                eval_record.update(metrics)
                log_jsonl(log_path, eval_record)

                ckpt_path = os.path.join(args.outdir, f"step{step}.pt")
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optim.state_dict(),
                    "scheduler": sched.state_dict(),
                    "step": step,
                    "args": vars(args),
                    "metrics": metrics,
                }, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

    print("Done.")


if __name__ == "__main__":
    main()
