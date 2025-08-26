# eval_hf_v21_pool.py
# Mini-retrieval eval on HF MS MARCO v2.1 validation global pool
# Uses provided `is_selected` labels, skips rows with "No Answer Present"

import argparse, random, sys
from typing import List, Dict, Any

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import BertTokenizer

import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from supervised.dual_encoder import DualEncoder
try:
    from supervised.train_supervised_updated import load_dual_encoder
    HAVE_LOAD_HELPER = True
except Exception:
    HAVE_LOAD_HELPER = False

try:
    import faiss
except ImportError as e:
    print("ERROR: faiss is required. Install via `pip install faiss-cpu` (or faiss-gpu).")
    raise e


def extract_passage_texts(passages_field):
    # passages are usually list[str], but sometimes list[dict]
    out = []
    for p in passages_field:
        if isinstance(p, str):
            out.append(p)
        elif isinstance(p, dict):
            if "passage_text" in p:
                out.append(p["passage_text"])
            elif "text" in p:
                out.append(p["text"])
    return out


def encode_texts(texts, model, tokenizer, device, is_query, batch_size, max_len):
    vecs = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            toks = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(device)

            if is_query:
                v = model._encode(model.query_encoder, toks["input_ids"], toks["attention_mask"])
            else:
                v = model._encode(model.doc_encoder, toks["input_ids"], toks["attention_mask"])
            v = F.normalize(v, dim=-1)
            vecs.append(v.cpu())
    return torch.cat(vecs, dim=0).numpy()


def mrr_at_k(ranked_ids, pos_set, k=10):
    for i, pid in enumerate(ranked_ids[:k], start=1):
        if pid in pos_set:
            return 1.0 / i
    return 0.0

def main():
    ap = argparse.ArgumentParser(description="Mini-retrieval eval on HF MS MARCO v2.1 validation global pool.")
    ap.add_argument("--pool_queries", type=int, default=5000)
    ap.add_argument("--pool_size", type=int, default=100_000)
    ap.add_argument("--eval_queries", type=int, default=1000)
    ap.add_argument("--model_name", type=str, default="bert-base-uncased")
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--q_batch_size", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--metrics_k", type=str, default="10,100")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading HF dataset: microsoft/ms_marco v2.1 validation ...")
    ds = load_dataset("microsoft/ms_marco", "v2.1", split="validation")

    # --- Build pool ---
    # downsize to the validation query pool in question
    ds_pool = ds.shuffle(seed=args.seed).select(range(min(args.pool_queries, len(ds))))

    text2id = {}
    id2text = []
    def get_pid(text):
        if text not in text2id:
            text2id[text] = len(id2text)
            id2text.append(text)
        return text2id[text]

    q_to_posids = {}
    # loop over rows (queries)
    for ex in tqdm(ds_pool, desc="Collecting candidates"):
        # skip "No Answer Present"
        if "answers" in ex and any("no answer present" in a.lower() for a in ex["answers"]):
            continue

        passages = extract_passage_texts(ex["passages"]["passage_text"])
        labels = ex["passages"]["is_selected"]  # 0/1 per passage

        pos_ids = []
        for p, y in zip(passages, labels):
            pid = get_pid(p)
            if y == 1:
                pos_ids.append(pid)

        if pos_ids:
            q_to_posids[ex["query"]] = set(pos_ids)

    pool_ids = set()
    for posids in q_to_posids.values():
        pool_ids.update(posids)

    all_ids = list(range(len(id2text)))
    random.shuffle(all_ids)
    for pid in all_ids:
        if len(pool_ids) >= args.pool_size:
            break
        pool_ids.add(pid)
    pool_ids = list(pool_ids)
    import ipdb
    ipdb.set_trace()

    print(f"Pool size (deduped): {len(pool_ids)}")

    # --- Choose eval queries ---
    ds_eval = ds.shuffle(seed=args.seed + 1)
    eval_set = []
    pool_ids_set = set(pool_ids)
    for ex in ds_eval:
        if "answers" in ex and any("no answer present" in a.lower() for a in ex["answers"]):
            continue
        q = ex["query"]
        if q in q_to_posids:
            pos_here = q_to_posids[q] & pool_ids_set
            if pos_here:
                eval_set.append((q, pos_here))
        if len(eval_set) >= args.eval_queries:
            break
    print(f"Eval queries: {len(eval_set)}")

    if len(eval_set) == 0:
        print("No evaluation queries found with positives in pool.")
        sys.exit(1)

    tok = BertTokenizer.from_pretrained(args.model_name)

    if args.ckpt:
        if HAVE_LOAD_HELPER:
            model = load_dual_encoder(args.ckpt, model_name=args.model_name, device=device)
        else:
            model = DualEncoder(model_name=args.model_name).to(device)
            state = torch.load(args.ckpt, map_location=device)
            if isinstance(state, dict):
                if "model_state_dict" in state:
                    state = state["model_state_dict"]
                elif "state_dict" in state:
                    state = state["state_dict"]
            model.load_state_dict(state, strict=False)
            model.eval()
    else:
        model = DualEncoder(model_name=args.model_name).to(device)
        model.eval()

    print("Encoding pool passages...")
    pool_texts = [id2text[i] for i in pool_ids]
    D = encode_texts(pool_texts, model, tok, device, is_query=False,
                     batch_size=args.batch_size, max_len=args.max_len)

    index = faiss.IndexFlatIP(D.shape[1])
    index.add(D)
    pool_ids_arr = np.array(pool_ids)

    Ks = [int(k.strip()) for k in args.metrics_k.split(",") if k.strip()]
    maxK = max(Ks + [10])

    mrr10_sum = 0.0
    recall = {k: 0.0 for k in Ks}
    n_eval = 0

    print("Evaluating...")
    BQ = args.q_batch_size
    for i in range(0, len(eval_set), BQ):
        batch = eval_set[i:i + BQ]
        q_texts = [q for (q, _) in batch]
        q_possets = [pos for (_, pos) in batch]

        Q = encode_texts(q_texts, model, tok, device,
                         is_query=True, batch_size=args.q_batch_size, max_len=args.max_len)

        scores, idxs = index.search(Q, maxK)
        for row, pos_set in enumerate(q_possets):
            ranked = pool_ids_arr[idxs[row]].tolist()
            mrr10_sum += mrr_at_k(ranked, pos_set, k=10)
            for k in Ks:
                hit = any(pid in pos_set for pid in ranked[:k])
                recall[k] += float(hit)
            n_eval += 1

    print("\n========== Results ==========")
    print(f"Eval queries: {n_eval}")
    print(f"MRR@10: {mrr10_sum / n_eval:.4f}")
    for k in Ks:
        print(f"Recall@{k}: {recall[k] / n_eval:.4f}")


if __name__ == "__main__":
    main()
