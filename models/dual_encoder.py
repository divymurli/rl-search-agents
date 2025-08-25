import torch
import torch.nn as nn
from transformers import BertModel

class DualEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', pooling='cls'):
        super().__init__()
        self.query_encoder = BertModel.from_pretrained(model_name)
        self.doc_encoder = BertModel.from_pretrained(model_name)
        self.pooling = pooling

    def _encode(self, model, input_ids, attention_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        if self.pooling == 'cls':
            return outputs.last_hidden_state[:, 0]  # [CLS] token
        elif self.pooling == 'mean':
            # mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size())
            sum_embeddings = torch.sum(outputs.last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            return sum_embeddings / sum_mask.clamp(min=1e-9)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")

    def forward(self, query_input, doc_input):
        q_emb = self._encode(self.query_encoder, **query_input)  # (B, H)
        d_emb = self._encode(self.doc_encoder, **doc_input)      # (B, H)
        return q_emb, d_emb