# data/make_msmarco_subset.py
# -----------------------------------------------------------
# Download MS MARCO (passage) via Hugging Face datasets, filter
# for queries with ≥1 positive, stratify by query length, and
# write a compact subset plus dev split. [#TODO: toggle between 
# in-batch negative sampling and hard triples.]
# -----------------------------------------------------------

import argparse, random, json, os
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
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("🔄  Loading MS MARCO passage (train) from HF datasets…")
    dset = datasets.load_dataset("ms_marco", "v2.1", split="train", streaming=False)

    query_positive_pairs: List[Dict] = []
    for ex in tqdm(dset, total=len(dset)):
        query = ex["query"]
        query_id = ex["query_id"]
        passages_data = ex["passages"]

        is_selected = passages_data["is_selected"]
        passage_texts = passages_data["passage_text"]
        urls = passages_data["url"]

        for i in range(len(is_selected)):
    
            if is_selected[i] == 1:
                positive_passage = passage_texts[i]
                positive_passage_id = urls[i]

                query_positive_pairs.append(
                    {
                        "query": query,
                        "query_id": query_id,
                        "passage": positive_passage,
                        "passage_id": positive_passage_id
                    }
                )

    print(f"Positives pool size: {len(query_positive_pairs)}")

    train_subset = stratified_sample(query_positive_pairs, args.num, args.seed)
    print("train stratification complete")

    # extremely inefficient and can be heavily optimized. just take away the smaller set from the larger one and sample
    # without replacement
    train_ids = {(t["query_id"], t["passage_id"]) for t in query_positive_pairs}
    remaining = [t for t in query_positive_pairs if (t["query_id"], t["passage_id"]) not in train_ids]
    dev_subset = stratified_sample(remaining, args.dev, args.seed + 1)

    print("stratification complete")
    write_jsonl(args.out_dir / "train.jsonl", train_subset)
    write_jsonl(args.out_dir / "dev.jsonl", dev_subset)

    print("✅  Wrote",
          args.out_dir / "train.jsonl",
          "and", args.out_dir / "dev.jsonl")

if __name__ == "__main__":
    main()