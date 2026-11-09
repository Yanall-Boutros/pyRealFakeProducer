# pyRealFakeProducer

`pyRealFakeProducer` parses 1,460 jazz chord charts from iRealPro's database, tokenizes them using WordPiece, and trains a modern **GPT-style Causal Decoder Transformer** to generate authentic, syntactically sound jazz chord sequences.

---

## Recent Architectural & Engineering Upgrades

This repository has undergone a comprehensive refactoring from a legacy 2017 encoder-decoder setup to a production-grade generative transformer stack:

* **Causal Decoder-Only Architecture:** Replaced the legacy encoder-decoder model with an autoregressive GPT-style causal decoder, mathematically and computationally optimized for sequential token prediction.
* **FlashAttention:** Integrated native PyTorch FlashAttention kernels (`F.scaled_dot_product_attention`) to eliminate $N^2$ memory bottlenecks and drastically accelerate training throughput.
* **Rotary Position Embeddings (RoPE):** Replaced absolute learnable embeddings with native RoPE, allowing the model to effectively model relative musical distances and transpositions across extended song forms.
* **SwiGLU & RMSNorm:** Upgraded feedforward layers to SwiGLU activations and implemented pre-normalization using RMSNorm for stable gradient propagation.
* **Batched DataLoader & Padding Masking:** Moved from single-sequence iteration to efficient mini-batched data loaders (`DataLoader`) with dynamic `%PAD%` masking, stabilizing gradient descent and regularizing training.
* **Advanced Autoregressive Sampling:** Upgraded inference generation with Temperature scaling, Top-p (Nucleus) sampling, and Repetition Penalties to prevent infinite attractor loops and successfully surface `%EOS%` termination tokens.
* **Pure Nix Environment:** Completely removed `poetry2nix` and `poetry.lock` in favor of a clean, minimal, native Nix flake (`flake.nix`) managing Python, PyTorch, and CUDA integration natively.

---

## Getting Started & Development

The project is packaged to run seamlessly on CUDA-enabled Linux environments using Nix flakes. 

### 1. Enter the Development Shell
```bash
nix develop --impure
```
### 2. Train the Model
```
python 2026_cleanup_test.py
```
### 3. Project Status & Samples
* Data & Vocabulary: Scrapes 1,460 iRealPro chord sequences across a 5,560-token vocabulary.
* Inference: Use the `generate_sequence` function with custom prompt start tokens (e.g., `["%BOS%", "A-"]`) to generate jazz chord progressions with nucleus sampling.
* Samples: Cherry-picked sample output progressions and generation checkpoints are archived in the samples/ directory.
