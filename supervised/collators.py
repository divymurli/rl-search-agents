# bi_encoder_collators.py
from typing import List, Dict
import torch

class BiEncoderInBatchCollator:
    """
    E0/E1: in-batch negatives. Optionally includes margin docs.
    Expects each item to have: query_text, pos_text, (optional) margin_text, neg_texts (ignored here).
    """
    def __init__(self, tokenizer, max_q_len=64, max_d_len=256, include_margin=False):
        self.tok = tokenizer
        self.max_q_len = max_q_len
        self.max_d_len = max_d_len
        self.include_margin = include_margin

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        q = [b["query_text"] for b in batch]
        d_pos = [b["pos_text"] for b in batch]

        q_tok = self.tok(q, padding=True, truncation=True, max_length=self.max_q_len, return_tensors="pt")
        d_tok = self.tok(d_pos, padding=True, truncation=True, max_length=self.max_d_len, return_tensors="pt")

        out = {
            "query_input_ids": q_tok["input_ids"],
            "query_attention_mask": q_tok["attention_mask"],
            "doc_input_ids": d_tok["input_ids"],
            "doc_attention_mask": d_tok["attention_mask"],
        }

        if self.include_margin:
            m = [ (b["margin_text"] or (b["neg_texts"][0] if b["neg_texts"] else b["pos_text"])) for b in batch ]
            m_tok = self.tok(m, padding=True, truncation=True, max_length=self.max_d_len, return_tensors="pt")
            out.update({
                "margin_input_ids": m_tok["input_ids"],
                "margin_attention_mask": m_tok["attention_mask"],
            })
        return out


class BiEncoderGroupedCollator:
    """
    E2/E3: grouped InfoNCE with mined negatives.
    Flattens docs per example as [pos, neg1..negK] and returns group offsets/sizes.
    """
    def __init__(self, tokenizer, max_q_len=64, max_d_len=256, include_margin=False):
        self.tok = tokenizer
        self.max_q_len = max_q_len
        self.max_d_len = max_d_len
        self.include_margin = include_margin

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        q_texts = [b["query_text"] for b in batch]
        doc_texts, group_offsets, group_sizes = [], [], []
        for b in batch:
            group_offsets.append(len(doc_texts))
            docs = [b["pos_text"]] + b["neg_texts"]   # K mined negatives already sampled in Dataset
            doc_texts.extend(docs)
            group_sizes.append(len(docs))

        q_tok = self.tok(q_texts, padding=True, truncation=True, max_length=self.max_q_len, return_tensors="pt")
        d_tok = self.tok(doc_texts, padding=True, truncation=True, max_length=self.max_d_len, return_tensors="pt")

        out = {
            "query_input_ids": q_tok["input_ids"],
            "query_attention_mask": q_tok["attention_mask"],
            "doc_input_ids": d_tok["input_ids"],
            "doc_attention_mask": d_tok["attention_mask"],
            "group_offsets": torch.tensor(group_offsets, dtype=torch.long),
            "group_sizes": torch.tensor(group_sizes, dtype=torch.long),
        }

        if self.include_margin:
            m = [ (b["margin_text"] or (b["neg_texts"][0] if b["neg_texts"] else b["pos_text"])) for b in batch ]
            m_tok = self.tok(m, padding=True, truncation=True, max_length=self.max_d_len, return_tensors="pt")
            out.update({
                "margin_input_ids": m_tok["input_ids"],
                "margin_attention_mask": m_tok["attention_mask"],
            })
        return out