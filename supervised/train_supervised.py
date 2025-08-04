import argparse

import torch
import  torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, AdamW, get_scheduler
from dataset import QueryPassageDataset
from dual_encoder import DualEncoder

from torch.utils.data import DataLoader

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import load_jsonl_file

def save_model(model, save_dir, step):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"checkpoint_step{step}.pt")
    torch.save(model.state_dict(), model_path)

    print(f"Saved checkpoint at step {step} to {model_path}")

def info_nce_loss(query_embeds, doc_embeds, temperature=1.0):

    """
    Args:
        query_embeds: (B, H)
        doc_embeds: (B, H) — positive document for each query
    """
    # (B, B) similarity matrix
    query_embeds = F.normalize(query_embeds, dim=-1)
    doc_embeds = F.normalize(doc_embeds, dim=-1)

    sim_matrix = torch.matmul(query_embeds, doc_embeds.T)  # [B, B]
    sim_matrix /= temperature

    labels = torch.arange(sim_matrix.size(0)).to(sim_matrix.device)  # correct doc for each query
    loss = nn.CrossEntropyLoss()(sim_matrix, labels)
    return loss

def get_optimizer(model, lr=2e-5, weight_decay=0.01):
    no_decay = ["bias", "LayerNorm.weight"]
    grouped_params = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(grouped_params, lr=lr)

def evaluate_mrr(model, dataloader, device, k=10):
    model.eval()
    mrr_total = 0.0
    num_queries = 0

    with torch.no_grad():
        for batch in dataloader:
            query_input = {
                'input_ids': batch['query_input_ids'].to(device),
                'attention_mask': batch['query_attention_mask'].to(device)
            }
            doc_input = {
                'input_ids': batch['doc_input_ids'].to(device),
                'attention_mask': batch['doc_attention_mask'].to(device)
            }

            q_embeds, d_embeds = model(query_input, doc_input)
            q_embeds = F.normalize(q_embeds, dim=-1)
            d_embeds = F.normalize(d_embeds, dim=-1)

            # (B, B) similarity matrix
            sim_matrix = torch.matmul(q_embeds, d_embeds.T)  # shape [B, B]

            # For each row i, rank d_i among all d_j
            for i in range(sim_matrix.size(0)):
                scores = sim_matrix[i]  # similarity of query i to all docs
                sorted_indices = torch.argsort(scores, descending=True)

                # Rank position of the true doc (which is at index i)
                rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1

                if rank <= k:
                    mrr_total += 1.0 / rank

            num_queries += sim_matrix.size(0)

    return mrr_total / num_queries if num_queries > 0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, default="data/msmarco_50k/train.jsonl")
    ap.add_argument("--dev", type=str, default="data/msmarco_50k/dev.jsonl")
    ap.add_argument("--num_epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--checkpoint_every", type=int, default=500)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--save_dir", type=str, default="./supervised/checkpoints")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    model = DualEncoder().to(device=device)

    optimizer = get_optimizer(model, lr=2e-5, weight_decay=0.01)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dicts = load_jsonl_file(args.train)
    dev_dicts = load_jsonl_file(args.dev)

    train_dataset = QueryPassageDataset(train_dicts, tokenizer=tokenizer)
    dev_dataset = QueryPassageDataset(dev_dicts, tokenizer=tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size)

    num_training_steps = args.num_epochs * len(train_dataloader)
    warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        for batch in train_dataloader:
            query_input = {
                'input_ids': batch['query_input_ids'].to(device),
                'attention_mask': batch['query_attention_mask'].to(device)
            }
            doc_input = {
                'input_ids': batch['doc_input_ids'].to(device),
                'attention_mask': batch['doc_attention_mask'].to(device)
            }

            q_emb, d_emb = model(query_input, doc_input)
            loss = info_nce_loss(q_emb, d_emb)

            import ipdb
            ipdb.set_trace()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            global_step += 1
            if global_step % args.checkpoint_every == 0:
                save_model(model, args.save_dir, global_step)
            if global_step % args.eval_every == 0:
                mrr = evaluate_mrr(model, dev_dataloader, device)
                print(f"MRR @ {global_step} steps: {mrr}")

            print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()    



