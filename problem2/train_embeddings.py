#!/usr/bin/env python3
# train_embeddings.py

import argparse
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# -------------------------
# Part B: Preprocessing
# -------------------------
def clean_text(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    text = text.lower()
    # keep only a-z and spaces
    text = re.sub(r'[^a-z\s]+', ' ', text)
    words = text.split()
    words = [w for w in words if len(w) >= 2]
    return words

def build_vocabulary(all_words: List[str], vocab_size: int) -> Tuple[dict, dict, int]:
    counter = Counter(all_words)
    total_words = sum(counter.values())
    most_common = counter.most_common(vocab_size)
    vocab_to_idx = {}
    idx_to_vocab = {}
    # reserve 0 for unknown
    idx = 1
    for word, _freq in most_common:
        vocab_to_idx[word] = idx
        idx_to_vocab[str(idx)] = word
        idx += 1
    return vocab_to_idx, idx_to_vocab, total_words

def words_to_sequence(words: List[str], vocab_to_idx: dict, seq_len: int) -> List[int]:
    seq = [vocab_to_idx.get(w, 0) for w in words]
    if len(seq) >= seq_len:
        return seq[:seq_len]
    else:
        return seq + [0] * (seq_len - len(seq))

def words_to_bow(words: List[str], vocab_to_idx: dict, vocab_size: int) -> np.ndarray:
    vec = np.zeros(vocab_size, dtype=np.float32)
    seen = set()
    for w in words:
        idx = vocab_to_idx.get(w, 0)
        if idx == 0:
            continue
        if idx not in seen:
            vec[idx - 1] = 1.0
            seen.add(idx)
    return vec

# -------------------------
# Dataset wrapper
# -------------------------
class BowDataset(Dataset):
    def __init__(self, bows: np.ndarray, arxiv_ids: List[str]):
        assert len(bows) == len(arxiv_ids)
        self.bows = bows
        self.ids = arxiv_ids

    def __len__(self):
        return len(self.bows)

    def __getitem__(self, idx):
        return self.bows[idx], self.ids[idx]

# -------------------------
# Part C: Autoencoder
# -------------------------
class TextAutoencoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding

# -------------------------
# Utility
# -------------------------
def count_parameters(model: nn.Module) -> int:
    last_three_layers = list(model.encoder.children())[1:] + list(model.decoder.children())
    total = sum(p.numel() for layer in last_three_layers for p in layer.parameters())
    return total

# -------------------------
# Main training logic
# -------------------------
def load_papers(input_json: str) -> List[dict]:
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Data expected as list of papers with fields including 'arxiv_id' and 'abstract'
    return data

def extract_abstracts(papers: List[dict]) -> Tuple[List[str], List[str]]:
    abstracts = []
    ids = []
    for p in papers:
        # try a few keys to be robust
        abstract = p.get('abstract') or p.get('summary') or p.get('title', '')
        # If abstract is empty, skip
        if abstract is None:
            continue
        # If abstract is empty string, still include (will be zero vector)
        abstracts.append(abstract)
        # prefer arxiv_id or id or something else
        arxiv_id = p.get('arxiv_id') or p.get('id') or p.get('paper_id') or p.get('doi') or f"paper_{len(ids)}"
        ids.append(arxiv_id)
    return abstracts, ids

def train(args):
    start_time = datetime.now(timezone.utc)
    print(f"Loading abstracts from {args.input}...")
    papers = load_papers(args.input)
    abstracts, arxiv_ids = extract_abstracts(papers)
    num_papers = len(abstracts)
    print(f"Found {num_papers} abstracts")

    # Clean texts
    cleaned = [clean_text(a) for a in abstracts]
    # Build global vocabulary
    all_words = [w for words in cleaned for w in words]
    print(f"Building vocabulary from {len(all_words):,} words...")
    vocab_to_idx, idx_to_vocab, total_words = build_vocabulary(all_words, args.vocab_size)
    effective_vocab_size = len(vocab_to_idx)  # <= args.vocab_size
    print(f"Vocabulary size: {effective_vocab_size} words (top {args.vocab_size})")

    # Sequence encoding (optional; we still produce as spec)
    sequences = [words_to_sequence(words, vocab_to_idx, args.seq_len) for words in cleaned]

    # Create bag-of-words representations
    vocab_size_for_model = args.vocab_size  # model expects this size (we will zero-fill if less)
    bows = np.zeros((num_papers, vocab_size_for_model), dtype=np.float32)
    for i, words in enumerate(cleaned):
        bows[i] = words_to_bow(words, vocab_to_idx, vocab_size_for_model)

    # Create dataset and dataloader
    ds = BowDataset(bows, arxiv_ids)
    def collate_fn(batch):
        # batch is list of tuples (bow, id)
        bows_batch = np.stack([b[0] for b in batch], axis=0)
        ids_batch = [b[1] for b in batch]
        return torch.from_numpy(bows_batch), ids_batch

    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Model config
    vocab_size = vocab_size_for_model
    hidden_dim = args.hidden_dim
    embedding_dim = args.embedding_dim

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TextAutoencoder(vocab_size=vocab_size, hidden_dim=hidden_dim, embedding_dim=embedding_dim).to(device)
    total_params = count_parameters(model)
    param_msg = f"Total parameters: {total_params:,}"
    under_limit = total_params < args.param_limit
    print(f"Model architecture: {vocab_size} → {hidden_dim} → {embedding_dim} → {hidden_dim} → {vocab_size}")
    print(param_msg + (f" (under {args.param_limit:,} limit ✓)" if under_limit else f" (exceeds {args.param_limit:,} limit ✗)"))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss(reduction='mean')  # multi-label binary cross-entropy

    print("\nTraining autoencoder...")
    t0 = time.time()
    model.train()
    epoch_losses = []
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for bows_batch, _ids in dataloader:
            bows_batch = bows_batch.to(device)
            bows_batch = bows_batch.float()
            optimizer.zero_grad()
            recon, _emb = model(bows_batch)
            loss = criterion(recon, bows_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(1, n_batches)
        epoch_losses.append(avg_loss)
        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch {epoch}/{args.epochs}, Loss: {avg_loss:.4f}")
    elapsed = time.time() - t0
    print(f"Training complete in {elapsed:.1f} seconds")

    # Switch to eval to produce embeddings and reconstruction losses per paper
    model.eval()
    embeddings_out = []
    with torch.no_grad():
        # We'll process in batches to avoid OOM
        eval_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        for bows_batch, ids_batch in eval_loader:
            bows_batch_t = bows_batch.to(device).float()
            recon, emb = model(bows_batch_t)
            # recon and emb are tensors
            # compute per-sample reconstruction loss (mean BCE per sample)
            # BCE per sample can be computed with: -(y*log(p) + (1-y)*log(1-p)) averaged over vocab
            # Avoid taking criterion with reduction='none' for memory efficiency
            eps = 1e-8
            recon_clamped = torch.clamp(recon, eps, 1 - eps)
            bce_per_feature = -(bows_batch_t * torch.log(recon_clamped) + (1 - bows_batch_t) * torch.log(1 - recon_clamped))
            # mean over features
            bce_per_sample = torch.mean(bce_per_feature, dim=1)
            recon_np = bce_per_sample.cpu().numpy()
            emb_np = emb.cpu().numpy()
            for i, paper_id in enumerate(ids_batch):
                embeddings_out.append({
                    "arxiv_id": paper_id,
                    "embedding": emb_np[i].tolist(),
                    "reconstruction_loss": float(recon_np[i])
                })

    # Save model.pth with provided format
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_to_idx': vocab_to_idx,
        'model_config': {
            'vocab_size': vocab_size,
            'hidden_dim': hidden_dim,
            'embedding_dim': embedding_dim
        }
    }, model_path)
    print(f"Saved model to {model_path}")

    # Save embeddings.json
    embeddings_path = os.path.join(args.output_dir, "embeddings.json")
    with open(embeddings_path, 'w', encoding='utf-8') as f:
        json.dump(embeddings_out, f, indent=2)
    print(f"Saved embeddings to {embeddings_path}")

    # Save vocabulary.json
    vocabulary_json = {
        "vocab_to_idx": vocab_to_idx,
        "idx_to_vocab": idx_to_vocab,
        "vocab_size": args.vocab_size,
        "total_words": total_words
    }
    vocab_path = os.path.join(args.output_dir, "vocabulary.json")
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocabulary_json, f, indent=2)
    print(f"Saved vocabulary to {vocab_path}")

    # Save training_log.json
    end_time = datetime.now(timezone.utc)
    training_log = {
        "start_time": start_time.replace(microsecond=0).isoformat() + "Z",
        "end_time": end_time.replace(microsecond=0).isoformat() + "Z",
        "epochs": args.epochs,
        "final_loss": float(epoch_losses[-1]) if epoch_losses else None,
        "total_parameters": total_params,
        "papers_processed": num_papers,
        "embedding_dimension": embedding_dim
    }
    log_path = os.path.join(args.output_dir, "training_log.json")
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(training_log, f, indent=2)
    print(f"Saved training log to {log_path}")

# -------------------------
# Argument parsing
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train a bag-of-words autoencoder on ArXiv abstracts.")
    parser.add_argument('input', type=str, help='Input papers.json file (HW#1 format)')
    parser.add_argument('output_dir', type=str, help='Directory to save outputs')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default 50)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default 32)')
    parser.add_argument('--vocab_size', type=int, default=5000, help='Vocabulary size (default 5000)')
    parser.add_argument('--seq_len', type=int, default=150, help='Sequence length for sequence encoding (default 150)')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size (default 256)')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Bottleneck embedding size (default 64)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default 1e-3)')
    parser.add_argument('--param_limit', type=int, default=2_000_000, help='Parameter limit for reporting (default 2,000,000)')
    parser.add_argument('--log_every', type=int, default=1, help='Logging frequency in epochs')
    args = parser.parse_args()
    # Normalize output_dir into args.output_dir to match function naming
    args.output_dir = args.output_dir
    return args

if __name__ == "__main__":
    args = parse_args()
    train(args)
