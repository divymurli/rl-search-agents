import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel

class DualEncoder(nn.Module):
    def __init__(
        self,
        model_name="bert-base-uncased",
        pooling="mean",              # "mean" or "cls"
        tied=False,                  # share weights?
        proj_dim=None,               # e.g., 256; None => no projection
        normalize=False,             # L2-normalize outputs
    ):
        super().__init__()
        self.pooling = pooling
        self.normalize = normalize

        self.query_encoder = AutoModel.from_pretrained(model_name)
        if tied:
            self.doc_encoder = self.query_encoder
        else:
            self.doc_encoder = AutoModel.from_pretrained(model_name)

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
        # mean pool over non-pad tokens
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def _encode(self, enc, input_ids, attention_mask, proj):
        out = enc(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        x = self._pool(out.last_hidden_state, attention_mask)
        if proj is not None:
            x = proj(x)
        if self.normalize:
            x = F.normalize(x, dim=-1)
        return x

    def encode_queries(self, input_ids, attention_mask):
        return self._encode(self.query_encoder, input_ids, attention_mask, self.q_proj)

    def encode_docs(self, input_ids, attention_mask):
        return self._encode(self.doc_encoder, input_ids, attention_mask, self.d_proj)

    def forward(self, query_input, doc_input):
        q_emb = self.encode_queries(**query_input)
        d_emb = self.encode_docs(**doc_input)
        return q_emb, d_emb
