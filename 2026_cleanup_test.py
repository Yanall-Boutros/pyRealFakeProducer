import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from pyRealParser import Tune
from utils import ireal_set_add

# ---------------------------------------------------------
# 1. Modern Architectural Components (RMSNorm & SwiGLU FFN)
# ---------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization for stable pre-LN scaling."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SwiGLUFeedForward(nn.Module):
    """SwiGLU feedforward network for enhanced feature representation."""
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(2 * dim_feedforward / 3)
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ---------------------------------------------------------
# 2. Causal Self-Attention with Native RoPE & FlashAttention
# ---------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Frequency buffer for Native Rotary Position Embeddings (RoPE)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        self.dropout = dropout

    def _apply_rotary(self, x: Tensor) -> Tensor:
        # x shape: [B, H, S, D]
        B, H, S, D = x.shape
        t = torch.arange(S, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq) # [S, D/2]
        emb = torch.cat((freqs, freqs), dim=-1) # [S, D]
        cos = emb.cos().view(1, 1, S, D)
        sin = emb.sin().view(1, 1, S, D)
        
        x1 = x[..., : D // 2]
        x2 = x[..., D // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (rotated * sin)

    def forward(self, x: Tensor) -> Tensor:
        B, S, C = x.shape
        
        # Project and reshape into multi-head format: (B, H, S, D)
        q = self.q_proj(x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)

        # Apply Rotary Position Embeddings to queries and keys
        q = self._apply_rotary(q)
        k = self._apply_rotary(k)

        # PyTorch scaled_dot_product_attention with FlashAttention kernels & causal mask
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.dropout if self.training else 0.0, 
            is_causal=True
        )

        # Recombine heads: (B, H, S, D) -> (B, S, C)
        out = out.transpose(1, 2).contiguous().view(B, S, C)
        return self.out_proj(out)


# ---------------------------------------------------------
# 3. Modern Transformer Decoder Block (Pre-LN)
# ---------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, nhead, dropout)
        
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFeedForward(d_model, dim_feedforward, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------
# 4. Complete Autoregressive Chord Language Model
# ---------------------------------------------------------

class ModernChordTransformer(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int = 256, 
        num_layers: int = 6, 
        nhead: int = 8, 
        dim_feedforward: int = 1024, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: Tensor) -> Tensor:
        x = self.dropout(self.embedding(idx))
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        logits = self.head(x)
        return logits


# ---------------------------------------------------------
# 5. Dataset, DataLoader & Tokenizer Pipeline
# ---------------------------------------------------------

class ChordDataset(Dataset):
    def __init__(self, corpus, tok, max_len):
        self.data = []
        for rows in corpus:
            row_ids = [x.ids[0] for x in tok.encode_batch(rows)]
            if len(row_ids) < 2:
                continue
            if len(row_ids) > max_len:
                row_ids = row_ids[:max_len]
            self.data.append(row_ids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)


def get_corpus(tunes):
    return [["%BOS%"] + tune.measures_as_strings + ["%EOS%"] for tune in tunes]


def init_tokenizer_and_data(max_seq_len=128):
    with open("./ireal_url", "r") as f:
        tunes = Tune.parse_ireal_url(f.read())
        
    chord_types = set()
    ireal_set_add(tunes, chord_types)
    corpus = get_corpus(tunes)
    
    tok = Tokenizer(WordPiece())
    trainer = WordPieceTrainer(special_tokens=["%BOS%", "%EOS%", "%PAD%"])
    tok.train_from_iterator(corpus, trainer=trainer)
    
    vocab_size = tok.get_vocab_size()
    return vocab_size, tunes, tok, corpus


# ---------------------------------------------------------
# 6. Constrained Sampling Generation (Temperature, Top-p, Repetition Penalty)
# ---------------------------------------------------------

def generate_sequence(model, tok, start_tokens, max_new_tokens=32, temperature=0.8, top_p=0.9, repetition_penalty=1.2, device="cuda"):
    model.eval()
    context = [tok.token_to_id(t) for t in start_tokens]
    generated = context.copy()
    eos_id = tok.token_to_id("%EOS%")
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor([generated], dtype=torch.long, device=device)
            logits = model(x)[:, -1, :] / temperature
            
            for token_id in set(generated):
                if logits[0, token_id] < 0:
                    logits[0, token_id] *= repetition_penalty
                else:
                    logits[0, token_id] /= repetition_penalty
                    
            probs = F.softmax(logits, dim=-1)
            
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            probs[0, sorted_indices[sorted_indices_to_remove]] = 0.0
            probs.div_(probs.sum(dim=-1, keepdim=True))
            
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            if next_token == eos_id:
                generated.append(next_token)
                break
                
            generated.append(next_token)
            
    return [tok.id_to_token(tid) for tid in generated]


# ---------------------------------------------------------
# 7. Main Training Loop with Batched DataLoader
# ---------------------------------------------------------

def main():
    torch.set_grad_enabled(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    max_seq_len = 128
    emb_dim = 128
    batch_size = 32
    num_epochs = 10
    
    vocab_size, tunes, tok, corpus = init_tokenizer_and_data(max_seq_len)
    pad_id = tok.token_to_id("%PAD%")
    
    dataset = ChordDataset(corpus, tok, max_seq_len)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=lambda b: torch.nn.utils.rnn.pad_sequence(b, batch_first=True, padding_value=pad_id)
    )
    
    model = ModernChordTransformer(
        vocab_size=vocab_size,
        d_model=emb_dim,
        num_layers=6,
        nhead=4,
        dim_feedforward=512,
        dropout=0.1
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    total_steps = len(dataloader) * num_epochs
    warmup_steps = int(0.1 * total_steps) # 10% warmup
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"Starting batched training on {len(dataset)} chord sequences (Batch Size: {batch_size})...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            batch = batch.to(device)
            
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            optimizer.zero_grad()
            logits = model(inputs)
            
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size), 
                targets.reshape(-1), 
                ignore_index=pad_id
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
        
        sample_output = generate_sequence(model, tok, ["%BOS%", "A-"], max_new_tokens=20, device=device)
        print(f"Sample Generation: {' | '.join(sample_output)}")

if __name__ == "__main__":
    main()
