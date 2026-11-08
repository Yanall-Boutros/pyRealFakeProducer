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
# 2. DeepSeekMoE: Shared + Fine-Grained Routed Experts
# ---------------------------------------------------------

class DeepSeekMoE(nn.Module):
    """Mixture of Experts with dedicated shared experts and fine-grained routed experts."""
    def __init__(
        self, 
        d_model: int, 
        dim_feedforward: int, 
        num_shared_experts: int = 1,
        num_routed_experts: int = 6, 
        top_k: int = 2, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        
        # Always-active shared experts for universal harmonic grammar
        self.shared_experts = nn.ModuleList([
            SwiGLUFeedForward(d_model, dim_feedforward, dropout) for _ in range(num_shared_experts)
        ])
        
        # Sparsely activated fine-grained routed experts
        self.routed_experts = nn.ModuleList([
            SwiGLUFeedForward(d_model, dim_feedforward, dropout) for _ in range(num_routed_experts)
        ])
        
        self.gate = nn.Linear(d_model, num_routed_experts, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, S, C = x.shape
        x_flat = x.view(-1, C)  # [B*S, C]
        
        # 1. Process through shared experts
        shared_out = sum(expert(x_flat) for expert in self.shared_experts)
        
        # 2. Route tokens through fine-grained routed experts
        logits = self.gate(x_flat)  # [B*S, num_routed_experts]
        weights = F.softmax(logits, dim=-1)
        
        top_weights, top_indices = torch.topk(weights, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        
        routed_out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.routed_experts):
            mask = (top_indices == i)
            if not mask.any():
                continue
            expert_res = expert(x_flat)
            for k_idx in range(self.top_k):
                k_mask = mask[:, k_idx]
                if k_mask.any():
                    routed_out[k_mask] += expert_res[k_mask] * top_weights[k_mask, k_idx].unsqueeze(-1)
                    
        total_out = shared_out + routed_out
        return total_out.view(B, S, C)


# ---------------------------------------------------------
# 3. Multi-Head Latent Attention (MLA) & Logit Soft-Capping
# ---------------------------------------------------------

class MultiHeadLatentAttention(nn.Module):
    """Multi-Head Latent Attention with low-rank KV compression & logit soft-capping."""
    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        kv_latent_dim: int = 64,
        dropout: float = 0.1, 
        rope_base: float = 500000.0,
        window_size: int = 64,
        soft_cap: float = 30.0
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.window_size = window_size
        self.soft_cap = soft_cap
        
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"

        # Low-rank latent compression for KV cache (MLA discovery)
        self.kv_down_proj = nn.Linear(d_model, kv_latent_dim, bias=False)
        self.k_up_proj = nn.Linear(kv_latent_dim, d_model, bias=False)
        self.v_up_proj = nn.Linear(kv_latent_dim, d_model, bias=False)
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.dropout = dropout

    def _apply_rotary(self, x: Tensor) -> Tensor:
        B, H, S, D = x.shape
        t = torch.arange(S, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().view(1, 1, S, D)
        sin = emb.sin().view(1, 1, S, D)
        
        x1 = x[..., : D // 2]
        x2 = x[..., D // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (rotated * sin)

    def forward(self, x: Tensor) -> Tensor:
        B, S, C = x.shape
        
        q = self.q_proj(x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        
        # MLA Low-rank KV compression & decompression
        kv_latent = self.kv_down_proj(x)
        k_full = self.k_up_proj(kv_latent).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v_full = self.v_up_proj(kv_latent).view(B, S, self.nhead, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k_full)

        q = self._apply_rotary(q)
        k = self._apply_rotary(k)

        attn_mask = None
        is_causal = True
        if self.window_size > 0:
            row_idx = torch.arange(S, device=x.device).unsqueeze(1)
            col_idx = torch.arange(S, device=x.device).unsqueeze(0)
            attn_mask = (col_idx <= row_idx) & ((row_idx - col_idx) < self.window_size)
            is_causal = False

        # Scaled dot product attention
        out = F.scaled_dot_product_attention(
            q, k, v_full, 
            attn_mask=attn_mask, 
            dropout_p=self.dropout if self.training else 0.0, 
            is_causal=is_causal
        )

        out = out.transpose(1, 2).contiguous().view(B, S, C)
        return self.out_proj(out)


# ---------------------------------------------------------
# 4. Modern Transformer Decoder Block (Pre-LN with DeepSeekMoE & MLA)
# ---------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        dim_feedforward: int, 
        dropout: float = 0.1,
        rope_base: float = 500000.0,
        window_size: int = 64,
        soft_cap: float = 30.0
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadLatentAttention(d_model, nhead, kv_latent_dim=64, dropout=dropout, rope_base=rope_base, window_size=window_size, soft_cap=soft_cap)
        
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = DeepSeekMoE(d_model, dim_feedforward, num_shared_experts=1, num_routed_experts=6, top_k=2, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------
# 5. Autoregressive Chord Language Model with Logit Soft-Capping
# ---------------------------------------------------------

class ModernChordTransformer(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int = 256, 
        num_layers: int = 6, 
        nhead: int = 8, 
        dim_feedforward: int = 1024, 
        dropout: float = 0.1,
        rope_base: float = 500000.0,
        window_size: int = 64,
        soft_cap: float = 30.0
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.soft_cap = soft_cap
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout, rope_base, window_size, soft_cap)
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
        
        # Logit Soft-Capping (Gemma/Grok discovery for numerical stability)
        if self.soft_cap > 0:
            logits = self.soft_cap * torch.tanh(logits / self.soft_cap)
            
        return logits


# ---------------------------------------------------------
# 6. Dataset, DataLoader & Tokenizer Pipeline
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
# 7. Advanced Constrained Sampling with Contrastive Penalty
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
# 8. Main Training Loop with Batched DataLoader
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
        dropout=0.1,
        rope_base=500000.0,
        window_size=64,
        soft_cap=30.0
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    total_steps = len(dataloader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"Starting MLA-DeepSeekMoE batched training on {len(dataset)} chord sequences (Batch Size: {batch_size})...")
    
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
