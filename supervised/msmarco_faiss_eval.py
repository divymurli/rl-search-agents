import torch
import torch.nn as nn
import numpy as np
import ir_datasets
import os
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import faiss
from collections import defaultdict

class MSMARCODataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

class DualEncoderEvaluator:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def encode_texts(self, texts, encoder_type='doc', batch_size=32, max_length=128):
        """Encode a list of texts using the specified encoder"""
        dataset = MSMARCODataset(texts, self.tokenizer, max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        embeddings = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Encoding {encoder_type}s"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                if encoder_type == 'query':
                    emb = self.model._encode(self.model.query_encoder, 
                                           batch['input_ids'], batch['attention_mask'])
                else:  # doc
                    emb = self.model._encode(self.model.doc_encoder, 
                                           batch['input_ids'], batch['attention_mask'])
                
                embeddings.append(emb.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def build_faiss_index(self, doc_embeddings):
        """Build FAISS index for efficient similarity search"""
        dimension = doc_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(doc_embeddings)
        index.add(doc_embeddings.astype('float32'))
        
        return index
    
    def evaluate_retrieval(self, dataset_name="msmarco-passage/dev/small", 
                          top_k=[1, 5, 10, 100], batch_size=32, 
                          max_docs=None, save_embeddings=True):
        """Complete evaluation pipeline"""
        print(f"Loading dataset: {dataset_name}")
        dataset = ir_datasets.load(dataset_name)
        
        # Load all data with optional limit
        print("Loading documents, queries, and qrels...")
        if max_docs:
            print(f"Limiting to first {max_docs} documents for testing...")
            docs_list = []
            for i, doc in enumerate(dataset.docs_iter()):
                if i >= max_docs:
                    break
                docs_list.append(doc)
        else:
            docs_list = list(dataset.docs_iter())
        
        queries_list = list(dataset.queries_iter())
        qrels_list = list(dataset.qrels_iter())
        
        # Create mappings
        doc_id_to_idx = {doc.doc_id: idx for idx, doc in enumerate(docs_list)}
        query_id_to_idx = {query.query_id: idx for idx, query in enumerate(queries_list)}
        
        # Create relevance mapping
        qrels_dict = defaultdict(set)
        for qrel in qrels_list:
            if qrel.relevance > 0:  # Consider any positive relevance as relevant
                qrels_dict[qrel.query_id].add(qrel.doc_id)
        
        print(f"Loaded {len(docs_list)} documents, {len(queries_list)} queries")
        print(f"Found {len(qrels_dict)} queries with relevant documents")
        
        # Encode documents with optional saving/loading
        doc_embeddings_file = f"doc_embeddings_{dataset_name.replace('/', '_')}.npy"
        
        if save_embeddings and os.path.exists(doc_embeddings_file):
            print(f"Loading pre-computed document embeddings from {doc_embeddings_file}")
            doc_embeddings = np.load(doc_embeddings_file)
        else:
            print("Encoding documents...")
            doc_texts = [doc.text for doc in docs_list]
            doc_embeddings = self.encode_texts(doc_texts, 'doc', batch_size)
            
            if save_embeddings:
                print(f"Saving document embeddings to {doc_embeddings_file}")
                np.save(doc_embeddings_file, doc_embeddings)
        
        # Build FAISS index
        print("Building FAISS index...")
        index = self.build_faiss_index(doc_embeddings)
        
        # Encode queries
        print("Encoding queries...")
        query_texts = [query.text for query in queries_list]
        query_embeddings = self.encode_texts(query_texts, 'query', batch_size, max_length=64)
        
        # Normalize query embeddings
        faiss.normalize_L2(query_embeddings)
        
        # Evaluate
        print("Running retrieval evaluation...")
        max_k = max(top_k)
        
        # Store results for each k
        results = {k: {'hits': 0, 'total_queries': 0} for k in top_k}
        mrr_sum = 0
        valid_queries = 0
        
        for query_idx, query in enumerate(tqdm(queries_list, desc="Evaluating queries")):
            query_id = query.query_id
            
            # Skip if no relevant documents
            if query_id not in qrels_dict:
                continue
                
            relevant_doc_ids = qrels_dict[query_id]
            
            # Search
            query_emb = query_embeddings[query_idx:query_idx+1]
            scores, retrieved_indices = index.search(query_emb.astype('float32'), max_k)
            
            # Convert indices to doc_ids
            retrieved_doc_ids = [docs_list[idx].doc_id for idx in retrieved_indices[0]]
            
            # Calculate metrics
            valid_queries += 1
            
            # MRR calculation
            for rank, doc_id in enumerate(retrieved_doc_ids):
                if doc_id in relevant_doc_ids:
                    mrr_sum += 1.0 / (rank + 1)
                    break
            
            # Recall@k calculation
            for k in top_k:
                retrieved_at_k = set(retrieved_doc_ids[:k])
                if len(retrieved_at_k & relevant_doc_ids) > 0:
                    results[k]['hits'] += 1
                results[k]['total_queries'] = valid_queries
        
        # Calculate final metrics
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        for k in sorted(top_k):
            recall_at_k = results[k]['hits'] / results[k]['total_queries'] if results[k]['total_queries'] > 0 else 0
            print(f"Recall@{k}: {recall_at_k:.4f} ({results[k]['hits']}/{results[k]['total_queries']})")
        
        mrr = mrr_sum / valid_queries if valid_queries > 0 else 0
        print(f"MRR: {mrr:.4f}")
        print(f"Total queries evaluated: {valid_queries}")
        
        return {
            'recall_at_k': {k: results[k]['hits'] / results[k]['total_queries'] 
                           for k in top_k if results[k]['total_queries'] > 0},
            'mrr': mrr,
            'total_queries': valid_queries
        }

# Usage examples for different scales
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DualEncoder(model_name='bert-base-uncased', pooling='cls')
    # model.load_state_dict(torch.load('path_to_your_finetuned_model.pt'))
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    evaluator = DualEncoderEvaluator(model, tokenizer, device)
    
    print("Choose evaluation scale:")
    print("1. Small dev set (~7k docs) - Fast")
    print("2. Full dev set (8.8M docs) - Slow but complete") 
    print("3. Limited dev set (100k docs) - Medium scale for testing")
    
    # Option 1: Small dev set (recommended for initial testing)
    results_small = evaluator.evaluate_retrieval(
        dataset_name="msmarco-passage/dev/small"
    )
    
    # Option 2: Full dev set (8.8M docs - will take hours!)
    # results_full = evaluator.evaluate_retrieval(
    #     dataset_name="msmarco-passage/dev"
    # )
    
    # Option 3: Limited for testing (first 100k docs)
    # results_limited = evaluator.evaluate_retrieval(
    #     dataset_name="msmarco-passage/dev",
    #     max_docs=100000
    # )
    
    return results_small

if __name__ == "__main__":
    results = main()