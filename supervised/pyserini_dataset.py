# msmarco_indexed_dataset.py
from __future__ import annotations
import os, io, json, struct, hashlib, random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from torch.utils.data import Dataset

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False

# ---- Pyserini text store (unchanged) ----
from pyserini.index.lucene import LuceneIndexReader

class MsMarcoTextStore:
    def __init__(self, prebuilt_index: str = "msmarco-v1-passage", cache_size: int = 200_000):
        self.reader = LuceneIndexReader.from_prebuilt_index(prebuilt_index)
        self.cache: Dict[str, str] = {}
        self.cache_size = cache_size

    def get(self, docid: str) -> str:
        if docid in self.cache:
            return self.cache[docid]
        raw = self.reader.doc_raw(docid)
        if raw is None:
            contents = self.reader.doc_contents(docid)
            text = contents or ""
        else:
            try:
                text = json.loads(raw).get("contents", "")
            except Exception:
                text = raw
        if len(self.cache) >= self.cache_size:
            # drop ~10% arbitrary to cap memory
            for i, k in enumerate(list(self.cache.keys())):
                if i % 10 == 0:
                    self.cache.pop(k, None)
        self.cache[docid] = text
        return text

# ---- offset index utils ----

def build_offset_index(jsonl_path: str, idx_path: Optional[str] = None) -> str:
    """
    Create a binary .idx file with 8-byte little-endian offsets per line.
    Returns the idx path.
    """
    idx_path = idx_path or (jsonl_path + ".idx")
    os.makedirs(os.path.dirname(os.path.abspath(idx_path)) or ".", exist_ok=True)

    with open(jsonl_path, "rb") as f, open(idx_path, "wb") as out:
        off = f.tell()
        line = f.readline()
        n = 0
        while line:
            out.write(off.to_bytes(8, "little", signed=False))
            n += 1
            off = f.tell()
            line = f.readline()
    return idx_path

def load_offsets(idx_path: str):
    """
    Load 8-byte little-endian offsets. Uses numpy if available, else pure Python.
    Returns a sequence of ints (numpy array or list).
    """
    with open(idx_path, "rb") as f:
        data = f.read()
    if _HAS_NUMPY:
        arr = np.frombuffer(data, dtype="<u8")
        return arr
    else:
        # pure python
        n = len(data) // 8
        return [int.from_bytes(data[i*8:(i+1)*8], "little", signed=False) for i in range(n)]

# ---- small helpers ----

def _per_query_rng(seed: int, qid: str) -> random.Random:
    h = hashlib.md5(qid.encode("utf-8")).hexdigest()
    salt = int(h[:8], 16)
    return random.Random(seed ^ salt)

def _slice_buckets(counts: Dict[str, int]) -> Tuple[slice, slice, slice]:
    n_h, n_m, n_e = counts.get("hard", 0), counts.get("mid", 0), counts.get("easy", 0)
    h0, h1 = 1, 1 + n_h
    m0, m1 = h1, h1 + n_m
    e0, e1 = m1, m1 + n_e
    return slice(h0, h1), slice(m0, m1), slice(e0, e1)

# ---- the memory-light Dataset ----

class MsMarcoCandidatesIndexedDataset(Dataset):
    """
    Streams JSONL by seeking to byte offsets; does NOT load all JSON into memory.
    Expects each JSON line to contain:
      query_id, query, pos_id, candidates (with layout [pos|hard...|mid...|easy...]),
      meta.neg_bucket_counts
    """
    def __init__(
        self,
        jsonl_path: str,
        text_store: Optional[MsMarcoTextStore] = None,
        *,
        idx_path: Optional[str] = None,
        group_k: int = 8,
        seed: int = 42,
        qid_limit: int = 0,
        subset_stride: int = 1,
    ):
        self.jsonl_path = jsonl_path
        self.idx_path = idx_path or (jsonl_path + ".idx")
        if not os.path.exists(self.idx_path):
            print(f"[index] building offset index: {self.idx_path}")
            build_offset_index(self.jsonl_path, self.idx_path)
        self.offsets = load_offsets(self.idx_path)

        # Subset view without copying big arrays
        if subset_stride > 1:
            idxs = range(0, len(self.offsets), subset_stride)
            self.view = list(idxs)
        else:
            self.view = None

        if qid_limit:
            if self.view is None:
                self.view = list(range(min(qid_limit, len(self.offsets))))
            else:
                self.view = self.view[:qid_limit]

        self.text_store = text_store or MsMarcoTextStore()
        self.group_k = group_k
        self.seed = seed

        # training step for curriculum (can be updated externally)
        self._global_step = 0
        self._total_steps = max(1, len(self))  # default

        # file handle is opened lazily (and reopened in worker processes)
        self._fp = None

    def __len__(self):
        return len(self.view) if self.view is not None else len(self.offsets)

    # make dataset pickle-safe for DataLoader workers
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fp"] = None
        return state

    def _ensure_open(self):
        if self._fp is None:
            self._fp = open(self.jsonl_path, "rb", buffering=io.DEFAULT_BUFFER_SIZE)

    # curriculum controls
    def set_global_step(self, step: int):
        self._global_step = max(0, int(step))

    def set_total_steps(self, total: int):
        self._total_steps = max(1, int(total))

    def _margin_bucket_probs(self) -> Tuple[float, float, float]:
        t = min(1.0, self._global_step / float(self._total_steps))
        p_h = 0.05 * (1 - t) + 0.50 * t
        p_m = 0.15 * (1 - t) + 0.30 * t
        p_e = max(0.0, 1.0 - p_h - p_m)
        return p_h, p_m, p_e

    # sampling helpers (same logic as before, but on-demand)
    def _sample_group_negs(self, rec: Dict, rng: random.Random) -> List[str]:
        counts = rec["meta"]["neg_bucket_counts"]
        cands  = rec["candidates"]
        sl_h, sl_m, sl_e = _slice_buckets(counts)
        pool_h, pool_m, pool_e = cands[sl_h], cands[sl_m], cands[sl_e]

        sizes = [len(pool_h), len(pool_m), len(pool_e)]
        total = max(1, sum(sizes))
        want = [int(self.group_k * s / total) for s in sizes]
        while sum(want) < self.group_k:
            i = max(range(3), key=lambda j: (sizes[j] - want[j], sizes[j]))
            want[i] += 1
        want = [min(want[i], sizes[i]) for i in range(3)]
        need = self.group_k - sum(want)
        if need > 0:
            caps = [sizes[i] - want[i] for i in range(3)]
            for _ in range(need):
                j = max(range(3), key=lambda x: caps[x])
                if caps[j] <= 0: break
                want[j] += 1; caps[j] -= 1

        chosen = []
        if want[0] > 0: chosen.extend(rng.sample(pool_h, want[0]))
        if want[1] > 0: chosen.extend(rng.sample(pool_m, want[1]))
        if want[2] > 0: chosen.extend(rng.sample(pool_e, want[2]))
        if len(chosen) < self.group_k:
            leftovers = [d for d in (pool_h + pool_m + pool_e) if d not in set(chosen)]
            take = min(self.group_k - len(chosen), len(leftovers))
            if take > 0:
                chosen.extend(rng.sample(leftovers, take))
        return chosen

    def _sample_margin_neg(self, rec: Dict, rng: random.Random) -> Optional[str]:
        counts = rec["meta"]["neg_bucket_counts"]
        cands  = rec["candidates"]
        sl_h, sl_m, sl_e = _slice_buckets(counts)
        pool_h, pool_m, pool_e = cands[sl_h], cands[sl_m], cands[sl_e]

        p_h, p_m, p_e = self._margin_bucket_probs()
        buckets = []
        if pool_h: buckets.append(("hard", p_h, pool_h))
        if pool_m: buckets.append(("mid",  p_m, pool_m))
        if pool_e: buckets.append(("easy", p_e, pool_e))
        if not buckets: return None
        names, probs, pools = zip(*buckets)
        s = sum(probs) or 1.0
        probs = [p/s for p in probs]
        idx = _per_query_rng(self.seed, rec["query_id"]).choices(range(len(pools)), weights=probs, k=1)[0]
        pool = pools[idx]
        return rng.choice(pool)

    def _read_line(self, i: int) -> Dict:
        # get byte offset and read just that line
        src_idx = self.view[i] if self.view is not None else i
        offset = int(self.offsets[src_idx])
        self._ensure_open()
        self._fp.seek(offset)
        line = self._fp.readline()
        return json.loads(line)

    def __getitem__(self, i: int) -> Dict:
        rec = self._read_line(i)
        qid = rec["query_id"]
        rng = _per_query_rng(self.seed, qid)

        # guard if counts missing
        if "meta" not in rec or "neg_bucket_counts" not in rec["meta"]:
            n = len(rec["candidates"]) - 1
            c = n // 3
            rec.setdefault("meta", {})["neg_bucket_counts"] = {"hard": c, "mid": c, "easy": n - 2*c}

        q_text  = rec["query"]
        pos_id  = rec["pos_id"]
        pos_txt = self.text_store.get(pos_id)

        neg_ids = self._sample_group_negs(rec, rng)
        neg_txt = [self.text_store.get(d) for d in neg_ids]

        margin_id  = self._sample_margin_neg(rec, rng)
        margin_txt = self.text_store.get(margin_id) if margin_id else None

        return {
            "qid": qid,
            "query_text": q_text,
            "pos_id": pos_id,
            "pos_text": pos_txt,
            "neg_ids": neg_ids,
            "neg_texts": neg_txt,
            "margin_id": margin_id,
            "margin_text": margin_txt,
            "bucket_counts": rec["meta"]["neg_bucket_counts"],
        }
