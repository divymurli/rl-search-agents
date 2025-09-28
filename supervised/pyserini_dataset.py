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

# ---- the memory-light Dataset (fixed curriculum) ----

class MsMarcoCandidatesIndexedDataset(Dataset):
    """
    Streams JSONL by seeking to byte offsets; does NOT load all JSON into memory.
    Expects each JSON line to contain:
      query_id, query, pos_id, candidates (layout [pos|hard...|mid...|easy...]),
      meta.neg_bucket_counts = {"hard": int, "mid": int, "easy": int}
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
        include_texts: bool = True,
        max_step: int = 50_000,
    ):
        self.jsonl_path = jsonl_path
        self.idx_path = idx_path or (jsonl_path + ".idx")
        if not os.path.exists(self.idx_path):
            print(f"[index] building offset index: {self.idx_path}")
            build_offset_index(self.jsonl_path, self.idx_path)
        self.offsets = load_offsets(self.idx_path)

        # Subset view
        self.view = None
        if subset_stride > 1:
            self.view = list(range(0, len(self.offsets), subset_stride))
        if qid_limit:
            if self.view is None:
                self.view = list(range(min(qid_limit, len(self.offsets))))
            else:
                self.view = self.view[:qid_limit]

        self.text_store = text_store or MsMarcoTextStore()
        self.include_texts = include_texts
        self.group_k = int(group_k)
        self.seed = int(seed)

        # curriculum (dataset-level)
        self._global_step = 0
        self._total_steps = max(1, int(max_step))  # for ramp; you can override later

        # file handle (opened lazily; reset on fork)
        self._fp = None

    # ---------- public controls ----------
    def set_global_step(self, step: int):
        self._global_step = max(0, int(step))

    def set_total_steps(self, total: int):
        self._total_steps = max(1, int(total))

    def set_group_k(self, k: int):
        self.group_k = int(k)

    # ---------- Dataset API ----------
    def __len__(self):
        return len(self.view) if self.view is not None else len(self.offsets)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fp"] = None
        return state

    def _ensure_open(self):
        if self._fp is None:
            self._fp = open(self.jsonl_path, "rb", buffering=io.DEFAULT_BUFFER_SIZE)

    def _read_line(self, i: int) -> Dict:
        src_idx = self.view[i] if self.view is not None else i
        offset = int(self.offsets[src_idx])
        self._ensure_open()
        self._fp.seek(offset)
        line = self._fp.readline()
        if isinstance(line, (bytes, bytearray)):
            line = line.decode("utf-8")
        return json.loads(line)

    def __getitem__(self, i: int) -> Dict:
        rec = self._read_line(i)

        # guard: synthesize counts if missing
        if "meta" not in rec or "neg_bucket_counts" not in rec["meta"]:
            n = len(rec["candidates"]) - 1
            c = max(0, n // 3)
            rec.setdefault("meta", {})["neg_bucket_counts"] = {
                "hard": c, "mid": c, "easy": max(0, n - 2*c)
            }

        qid     = rec["query_id"]
        q_text  = rec["query"]
        pos_id  = rec["pos_id"]
        counts  = rec["meta"]["neg_bucket_counts"]
        cands   = rec["candidates"]            # [pos][hard...][mid...][easy...]

        # step-aware sampling
        group_neg_ids = self._sample_group_negs(cands, counts, qid)
        margin_neg_id = self._sample_margin_neg(cands, counts, qid)

        out = {
            "qid": qid,
            "query_text": q_text,
            "pos_id": pos_id,
            "neg_ids": group_neg_ids,
            "margin_id": margin_neg_id,
            "bucket_counts": counts,
            "step": self._global_step,
        }

        if self.include_texts:
            out["pos_text"] = self.text_store.get(pos_id)
            out["neg_texts"] = [self.text_store.get(d) for d in group_neg_ids]
            out["margin_text"] = self.text_store.get(margin_neg_id) if margin_neg_id else None

        return out

    # ---------- curriculum & RNG ----------
    def _p_hme(self) -> Tuple[float, float, float]:
        # linear ramp: easy→hard; always keep some mix
        t = min(1.0, self._global_step / float(self._total_steps))
        p_h = 0.05*(1 - t) + 0.50*t
        p_m = 0.15*(1 - t) + 0.30*t
        p_e = max(0.0, 1.0 - p_h - p_m)
        return (p_h, p_m, p_e)

    @staticmethod
    def _md5_int(s: str) -> int:
        return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:16], 16)

    def _rng(self, qid: str, salt: int = 0) -> random.Random:
        # Deterministic across runs; varies with (seed, qid, step, salt)
        key = f"{self.seed}|{qid}|{self._global_step}|{salt}"
        return random.Random(self._md5_int(key))

    @staticmethod
    def _bucket_slices(counts: Dict[str, int]) -> Tuple[slice, slice, slice]:
        n_h = counts.get("hard", 0)
        n_m = counts.get("mid", 0)
        n_e = counts.get("easy", 0)
        h0, h1 = 1, 1 + n_h
        m0, m1 = h1, h1 + n_m
        e0, e1 = m1, m1 + n_e
        return slice(h0, h1), slice(m0, m1), slice(e0, e1)

    @staticmethod
    def _weighted_pick(rnd: random.Random, weights: List[float]) -> int:
        s = sum(weights) or 1.0
        u = rnd.random() * s
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if u <= acc:
                return i
        return len(weights) - 1

    # ---------- samplers (now step-aware) ----------
    def _sample_group_negs(self, cands: List[str], counts: Dict[str, int], qid: str) -> List[str]:
        sl_h, sl_m, sl_e = self._bucket_slices(counts)
        pool_h, pool_m, pool_e = cands[sl_h], cands[sl_m], cands[sl_e]

        # curriculum-driven targets (NOT pool-size driven)
        K = min(self.group_k, len(pool_h) + len(pool_m) + len(pool_e))
        p_h, p_m, p_e = self._p_hme()
        raw = [p_h*K, p_m*K, p_e*K]
        tgt = [int(x) for x in raw]
        # distribute rounding remainder to largest fractional parts
        for _ in range(K - sum(tgt)):
            j = max(range(3), key=lambda i: (raw[i] - tgt[i], i))
            tgt[j] += 1

        # clamp to availability and backfill
        sizes = [len(pool_h), len(pool_m), len(pool_e)]
        tgt = [min(tgt[i], sizes[i]) for i in range(3)]
        need = K - sum(tgt)
        if need > 0:
            # backfill preferring harder buckets first
            caps = [sizes[i] - tgt[i] for i in range(3)]
            order = [0, 1, 2]  # hard -> mid -> easy
            for _ in range(need):
                j = next((i for i in order if caps[i] > 0), None)
                if j is None: break
                tgt[j] += 1; caps[j] -= 1

        rnd = self._rng(qid, salt=0xA1)
        chosen: List[str] = []
        if tgt[0] > 0: chosen += rnd.sample(pool_h, tgt[0])
        if tgt[1] > 0: chosen += rnd.sample(pool_m, tgt[1])
        if tgt[2] > 0: chosen += rnd.sample(pool_e, tgt[2])

        # backstop uniqueness
        if len(chosen) < K:
            leftovers = [d for d in (pool_h + pool_m + pool_e) if d not in set(chosen)]
            take = min(K - len(chosen), len(leftovers))
            if take > 0:
                chosen += rnd.sample(leftovers, take)
        return chosen

    def _sample_margin_neg(self, cands: List[str], counts: Dict[str, int], qid: str) -> Optional[str]:
        sl_h, sl_m, sl_e = self._bucket_slices(counts)
        pool_h, pool_m, pool_e = cands[sl_h], cands[sl_m], cands[sl_e]
        if not (pool_h or pool_m or pool_e):
            return None

        p_h, p_m, p_e = self._p_hme()
        rnd = self._rng(qid, salt=0xB2)

        pools = [pool_h, pool_m, pool_e]
        weights = [p_h if pool_h else 0.0, p_m if pool_m else 0.0, p_e if pool_e else 0.0]
        j = self._weighted_pick(rnd, weights)
        pool = pools[j] if pools[j] else (pool_h or pool_m or pool_e)
        return rnd.choice(pool)
