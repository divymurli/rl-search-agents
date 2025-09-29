import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class DualEncoder(nn.Module):
    def __init__(
        self,
        model_name="bert-base-uncased",
        pooling="mean",          # "mean" or "cls"
        tied=False,              # share weights between q/d
        proj_dim=None,           # e.g., 256; None => no projection
        normalize=False,         # L2-normalize outputs
    ):
        super().__init__()
        self.pooling = pooling
        self.normalize = normalize
        self.query_encoder = AutoModel.from_pretrained(model_name)
        self.doc_encoder   = self.query_encoder if tied else AutoModel.from_pretrained(model_name)

        hid = self.query_encoder.config.hidden_size
        if proj_dim is not None:
            self.q_proj = nn.Linear(hid, proj_dim, bias=False)
            self.d_proj = nn.Linear(hid, proj_dim, bias=False)
            self.out_dim = proj_dim
        else:
            self.q_proj = self.d_proj = None
            self.out_dim = hid

    def _pool(self, last_hidden_state, attention_mask):
        if self.pooling == "cls":
            return last_hidden_state[:, 0]
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def _prune_inputs(self, enc, inputs: dict):
        """Drop keys the encoder can’t handle (e.g., token_type_ids for DistilBERT)."""
        x = {k: v for k, v in inputs.items() if v is not None}
        # If the backbone has no token type embeddings, drop them
        if "token_type_ids" in x and getattr(enc.config, "type_vocab_size", None) in (None, 0):
            x.pop("token_type_ids")
        return x

    def _encode(self, enc, proj, **inputs):
        inputs = self._prune_inputs(enc, inputs)
        outputs = enc(**inputs, return_dict=True)
        attn = inputs.get("attention_mask")
        if attn is None:
            attn = torch.ones(outputs.last_hidden_state.size()[:-1], device=outputs.last_hidden_state.device)
        x = self._pool(outputs.last_hidden_state, attn)
        if proj is not None:
            x = proj(x)
        if self.normalize:
            x = F.normalize(x, dim=-1)
        return x

    # Accept flexible HF-style kwargs (input_ids, attention_mask, token_type_ids, ...)
    def encode_queries(self, **inputs):
        return self._encode(self.query_encoder, self.q_proj, **inputs)

    def encode_docs(self, **inputs):
        return self._encode(self.doc_encoder, self.d_proj, **inputs)

    # Keep forward for older callers that pass dicts explicitly
    def forward(self, query_input, doc_input):
        q = self.encode_queries(**query_input)
        d = self.encode_docs(**doc_input)
        return q, d