import argparse
from tqdm import tqdm

import torch
import  torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import BertTokenizer, get_scheduler
from dataset import QueryPassageDataset

from torch.utils.data import DataLoader
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.dual_encoder import DualEncoder

from utils import load_jsonl_file

def save_model(model, save_dir, step):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"checkpoint_step{step}.pt")
    torch.save(model.state_dict(), model_path)

    print(f"Saved checkpoint at step {step} to {model_path}")

def load_dual_encoder(checkpoint_path, model_name='bert-base-uncased', pooling='cls', device='cpu'):
    """Load DualEncoder from checkpoint with proper error handling"""
    
    # Initialize model
    model = DualEncoder(model_name=model_name, pooling=pooling)
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            # Assume it's directly the state dict
            state_dict = checkpoint
        
        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
        
        print(f"Successfully loaded model from {checkpoint_path}")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise
    
    model.eval()
    return model

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
        for batch in tqdm(dataloader):
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


@torch.no_grad()
def evaluate_mrr_full(model, dev_loader, device, k=10):
    model.eval()
    all_q, all_d = [], []

    # 1) Encode all queries and docs (assumes each batch has paired q_i, d_i)
    for batch in tqdm(dev_loader):
        q_input = {
            'input_ids': batch['query_input_ids'].to(device),
            'attention_mask': batch['query_attention_mask'].to(device)
        }
        d_input = {
            'input_ids': batch['doc_input_ids'].to(device),
            'attention_mask': batch['doc_attention_mask'].to(device)
        }
        q_emb, d_emb = model(q_input, d_input)
        all_q.append(q_emb)
        all_d.append(d_emb)

    Q = F.normalize(torch.cat(all_q, dim=0), dim=-1)   # [N, H]
    D = F.normalize(torch.cat(all_d, dim=0), dim=-1)   # [N, H]

    # 2) Full similarity matrix [N, N]
    S = Q @ D.T

    # 3) Rank of the diagonal item for each row
    # Larger is better, so we argsort descending
    ranks = torch.argsort(S, dim=1, descending=True)
    idx    = torch.arange(S.size(0), device=S.device)
    # Find where the true doc index (i) appears in sorted indices for row i
    diag_positions = (ranks == idx.unsqueeze(1)).nonzero(as_tuple=False)[:,1]  # [N]
    ranks_1based   = diag_positions + 1

    # 4) MRR@k
    mrr = (1.0 / ranks_1based.clamp_min(1)).masked_fill(ranks_1based > k, 0.0).mean().item()
    return mrr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, default="data/msmarco_100k/train.jsonl")
    ap.add_argument("--dev", type=str, default="data/msmarco_100k/dev.jsonl")
    ap.add_argument("--num_epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                    help="Number of steps to accumulate gradients before update")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--checkpoint_every", type=int, default=500)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--save_dir", type=str, default="./supervised/checkpoints")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = DualEncoder().to(device=device)

    # Calculate effective batch size for logging
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(f"Micro batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")

    optimizer = get_optimizer(model, lr=args.lr, weight_decay=0.01)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dicts = load_jsonl_file(args.train)
    dev_dicts = load_jsonl_file(args.dev)

    train_dataset = QueryPassageDataset(train_dicts, tokenizer=tokenizer)
    dev_dataset = QueryPassageDataset(dev_dicts, tokenizer=tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    # Calculate training steps based on gradient accumulation
    steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    num_training_steps = args.num_epochs * steps_per_epoch
    warmup_steps = int(0.2 * num_training_steps)

    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

    best_eval_loss = float('inf')
    patience_counter = 0

    model.to(device)

    mrr = evaluate_mrr_full(model, dev_dataloader, device)
    print(f"Starting MRR: {mrr:.4f}")

    global_step = 0
    accumulation_loss = 0.0
    
    for epoch in range(args.num_epochs):
        model.train()
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            query_input = {
                'input_ids': batch['query_input_ids'].to(device),
                'attention_mask': batch['query_attention_mask'].to(device)
            }
            doc_input = {
                'input_ids': batch['doc_input_ids'].to(device),
                'attention_mask': batch['doc_attention_mask'].to(device)
            }

            q_emb, d_emb = model(query_input, doc_input)
            loss = info_nce_loss(q_emb, d_emb, temperature=args.temperature)
            
            # Scale loss by accumulation steps
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            accumulation_loss += loss.item()

            # Only update weights after accumulating enough gradients
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Print loss (unscaled for readability)
                avg_loss = accumulation_loss * args.gradient_accumulation_steps
                print(f"Epoch {epoch+1} | Step {global_step} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
                accumulation_loss = 0.0
                
                # Checkpointing
                if global_step % args.checkpoint_every == 0:
                    save_model(model, args.save_dir, global_step)

                # Evaluation
                if global_step % args.eval_every == 0:
                    # ---- Eval loss ----
                    model.eval()
                    total_loss, n_batches = 0.0, 0
                    with torch.no_grad():
                        for dev_batch in tqdm(dev_dataloader, desc="Evaluating"):
                            q_in = {
                                "input_ids": dev_batch["query_input_ids"].to(device),
                                "attention_mask": dev_batch["query_attention_mask"].to(device),
                            }
                            d_in = {
                                "input_ids": dev_batch["doc_input_ids"].to(device),
                                "attention_mask": dev_batch["doc_attention_mask"].to(device),
                            }
                            q_e, d_e = model(q_in, d_in)
                            b_loss = info_nce_loss(q_e, d_e, temperature=args.temperature)
                            total_loss += b_loss.item()
                            n_batches += 1
                    eval_loss = total_loss / max(1, n_batches)

                    # ---- Eval MRR ----
                    mrr = evaluate_mrr_full(model, dev_dataloader, device)
                    print(f"[Eval @ step {global_step}] loss={eval_loss:.4f}  MRR@10={mrr:.4f}")

                    # ---- Early stopping on eval loss ----
                    # TODO: add eval accuracy as another patience counter
                    if eval_loss < best_eval_loss - 1e-6:  # tiny margin to avoid float noise
                        best_eval_loss = eval_loss
                        patience_counter = 0
                        print(f"New best eval loss: {best_eval_loss:.4f}")
                    else:
                        patience_counter += 1
                        print(f"No improvement ({patience_counter}/{args.patience}).")
                        if patience_counter >= args.patience:
                            print("Early stopping triggered.")
                            save_model(model, args.save_dir, global_step)  # final save
                            return

                    model.train()

    # Handle remaining gradients at the end of epoch
    if (len(train_dataloader) % args.gradient_accumulation_steps) != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    # Final save
    save_model(model, args.save_dir, global_step)

if __name__ == "__main__":
    main()