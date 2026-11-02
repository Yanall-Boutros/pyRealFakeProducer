import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch import Tensor
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
        # SwiGLU typically scales intermediate dimension by ~8/3
        hidden_dim = int(2 * dim_feedforward / 3)
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ---------------------------------------------------------
# 2. Causal Self-Attention Block with FlashAttention
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
        
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        B, S, C = x.shape
        
        # Project and reshape into multi-head format: (B, S, H, D) -> (B, H, S, D)
        q = self.q_proj(x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)

        # PyTorch scaled_dot_product_attention automatically applies FlashAttention kernels 
        # and built-in causal masking when is_causal=True.
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.dropout if self.training else 0.0, 
            is_causal=True
        )

        # Recombine heads
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
        # Pre-LN residual structure
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
        max_len: int = 256, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Learnable positional embeddings (modern alternative to fixed sinusoidal)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: Tensor) -> Tensor:
        B, S = idx.shape
        device = idx.device
        
        positions = torch.arange(0, S, device=device).unsqueeze(0).expand(B, S)
        
        x = self.embedding(idx) + self.pos_emb(positions)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        logits = self.head(x)
        return logits


# ---------------------------------------------------------
# 5. Data Pipeline & Training Script
# ---------------------------------------------------------

def get_corpus(tunes):
    return [["%BOS%"] + tune.measures_as_strings + ["%EOS%"] for tune in tunes]

def init_tokenizer_and_data():
    max_seq_len = 128
    emb_dim = 128
    
    with open("./ireal_url", "r") as f:
        tunes = Tune.parse_ireal_url(f.read())
        
    chord_types = set()
    ireal_set_add(tunes, chord_types)
    corpus = get_corpus(tunes)
    
    tok = Tokenizer(WordPiece())
    trainer = WordPieceTrainer(special_tokens=["%BOS%", "%EOS%", "%PAD%"])
    tok.train_from_iterator(corpus, trainer=trainer)
    
    vocab_size = tok.get_vocab_size()
    return max_seq_len, vocab_size, chord_types, tunes, tok, corpus, emb_dim

def generate_sequence(model, tok, start_tokens, max_new_tokens=32, device="cuda"):
    model.eval()
    context = [tok.token_to_id(t) for t in start_tokens]
    x = torch.tensor([context], dtype=torch.long, device=device)
    
    generated = context.copy()
    eos_id = tok.token_to_id("%EOS%")
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(x)
            # Take logits of the last token in sequence
            next_logits = logits[:, -1, :]
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            if next_token == eos_id:
                break
                
            generated.append(next_token)
            x = torch.tensor([generated], dtype=torch.long, device=device)
            
    return [tok.id_to_token(tid) for tid in generated]

def main():
    torch.set_grad_enabled(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    max_seq_len, vocab_size, chord_types, tunes, tok, corpus, emb_dim = init_tokenizer_and_data()
    
    pad_id = tok.token_to_id("%PAD%")
    
    # Initialize modern decoder-only model
    model = ModernChordTransformer(
        vocab_size=vocab_size,
        d_model=emb_dim,
        num_layers=6,
        nhead=4,
        dim_feedforward=512,
        max_len=max_seq_len,
        dropout=0.1
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    num_epochs = 5
    
    print(f"Starting training on {len(corpus)} iRealPro chord sequences with vocab size {vocab_size}...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for i, rows in enumerate(corpus):
            row_ids = [x.ids[0] for x in tok.encode_batch(rows)]
            if len(row_ids) < 2:
                continue
                
            # Truncate or pad sequence
            if len(row_ids) > max_seq_len:
                row_ids = row_ids[:max_seq_len]
                
            input_tensor = torch.tensor(row_ids, dtype=torch.long, device=device).unsqueeze(0)
            
            # Autoregressive causal language modeling targets (shifted by 1)
            inputs = input_tensor[:, :-1]
            targets = input_tensor[:, 1:]
            
            optimizer.zero_grad()
            logits = model(inputs)
            
            # Correct cross-entropy calculation over flattened logits and targets
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size), 
                targets.reshape(-1), 
                ignore_index=pad_id
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss / len(corpus):.4f}")
        
        # Test generation sample every epoch
        sample_output = generate_sequence(model, tok, ["%BOS%", "A-"], max_new_tokens=15, device=device)
        print(f"Sample Generation: {' | '.join(sample_output)}")

if __name__ == "__main__":
    main()
