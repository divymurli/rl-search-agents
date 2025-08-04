# data/make_msmarco_subset.py
# -----------------------------------------------------------
# Download MS MARCO (passage) via Hugging Face datasets, filter
# for queries with ≥1 positive, stratify by query length, and
# write a compact subset plus dev split. [#TODO: toggle between 
# in-batch negative sampling and hard triples.]
# -----------------------------------------------------------

import argparse, random, json
from pathlib import Path
from typing import List, Dict

import datasets  # pip install datasets==2.19.0
from tqdm import tqdm

# --------------------------- helpers ---------------------------

def stratified_sample(examples: List[Dict], num: int, seed: int) -> List[Dict]:
    """Sample `num` triples stratified over query length buckets."""
    random.seed(seed)
    buckets = {"short": [], "med": [], "long": []}
    for ex in tqdm(examples):
        q_len = len(ex["query"].split())
        if q_len <= 5:
            buckets["short"].append(ex)
        elif q_len <= 10:
            buckets["med"].append(ex)
        else:
            buckets["long"].append(ex)

    # proportional allocation 40/40/20
    target = {
        "short": int(num * 0.4),
        "med": int(num * 0.4),
        "long": num - int(num * 0.8),
    }
    sample: List[Dict] = []
    for k in tqdm(buckets):
        sample.extend(random.sample(buckets[k], min(target[k], len(buckets[k]))))
    random.shuffle(sample)
    return sample


def write_jsonl(path: Path, data: List[Dict]):
    with path.open("w", encoding="utf-8") as fp:
        for row in data:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=50000, help="size of train subset")
    ap.add_argument("--dev", type=int, default=5000, help="size of dev subset")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--with_negatives", action="store_true",
                   help="attach a same-query negative passage")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    print("🔄  Loading MS MARCO passage (train) from HF datasets…")
    dset = datasets.load_dataset("ms_marco", "v2.1", split="train", streaming=False)

    triples: List[Dict] = []
    for ex in tqdm(dset, total=len(dset)):
        query = ex["query"]
        query_id = ex["query_id"]
        passages = ex["passages"]

         # collect indices of positives & negatives
        pos_indices = [i for i, flag in enumerate(passages["is_selected"]) if flag==1]
        neg_indices = [i for i, flag in enumerate(passages["is_selected"]) if flag==0]
        if not neg_indices:
            continue  # skip if query has no negative passage (rare)
        for pi in pos_indices:
            pos_txt = passages["passage_text"][pi].strip()
            pos_id  = passages["url"][pi] or f"{query_id}_{pi}"
            row = {
                "query": query,
                "query_id": query_id,
                "positive": pos_txt,
                "pos_id": pos_id,
            }
            if args.with_negatives:
                ni = rng.choice(neg_indices)
                neg_txt = passages["passage_text"][ni].strip()
                neg_id  = passages["url"][ni] or f"{query_id}_{ni}"
                row["negative"] = neg_txt
                row["neg_id"] = neg_id
            triples.append(row)

    print(f"🥡 Built {len(triples):,} rows (positives {'+ negatives' if args.with_negatives else ''}).")

    train = stratified_sample(triples, args.num, args.seed)
    train_key = {(r["query_id"], r["pos_id"]) for r in train}
    remaining = [r for r in triples if (r["query_id"], r["pos_id"]) not in train_key]
    dev = stratified_sample(remaining, args.dev, args.seed+1)

    write_jsonl(args.out_dir/"train.jsonl", train)
    write_jsonl(args.out_dir/"dev.jsonl", dev)
    print("✅ Saved stratified splits →", args.out_dir)

    print("✅  Wrote",
          args.out_dir / "train.jsonl",
          "and", args.out_dir / "dev.jsonl")

if __name__ == "__main__":
    main()