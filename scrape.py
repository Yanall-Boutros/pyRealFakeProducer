from pyRealParser import Tune
import torch.nn as nn
import torch
import torch.nn.functional as F
import math,copy,re
import warnings
import pandas as pd
import numpy as np
import pdb
import sys
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch import Tensor
from utils import ireal_set_add
#tok = tokenizers.ByteLevelBPETokenizer()
tok = Tokenizer(BPE())
trainer = BpeTrainer()#special_tokens=["<BOS>", "<EOS>"])
#tok = Tokenizer(WordPiece())
#trainer = WordPieceTrainer()#special_tokens=["<BOS>", "<EOS>"])
#tok.pre_tokenizer = Whitespace()
from rotary_embedding_torch import RotaryEmbedding
import pytorch_model_classes as pmc

def get_corpus(tunes): return [["%BOS%"]+tune.measures_as_strings+["%EOS%"] for tune in tunes]

def embedding_to_token(embedding, embedding_layer):
  """
  Finds the closest token in the embedding layer to the input embedding.

  Args:
      embedding: Input embedding vector (tensor).
      embedding_layer: Embedding layer (nn.Embedding module).

  Returns:
      token: The closest token (string) based on cosine similarity.
  """
  # Get all embedding weights from the layer
  embedding_weights = embedding_layer.weight.detach()  # Detach weights to avoid gradients

  # Calculate cosine similarity between input embedding and all weights
  similarities = F.cosine_similarity(embedding.unsqueeze(0), embedding_weights)

  # Get the index of the most similar embedding
  most_similar_index = torch.argmax(similarities).item()
  token = tok.id_to_token(most_similar_index)
  # Retrieve the token based on the index (assuming vocabulary order matches weight order)
  return token


def generate_start_with(tokens, model, Embedding, tok):
    vocab = tok.get_vocab()
    tok_history = []
    tok_history.extend(tokens)
    print(tokens)
    timeout = 0
    while all([token in vocab for token in tok_history]) and len(tok_history) < 12:
      embedding = Embedding(Tensor([tok.token_to_id(token) for token in tok_history]).type(torch.int64).cuda())
      next_tokens = [embedding_to_token(n_emb, Embedding) for n_emb in model.decoder(embedding, model.encoder(embedding)) ]
      if next_tokens:
          if next_tokens[0]:
              tok_history.append(next_tokens[0])
              if "%EOS%" in tok_history or "%EOS%" in next_tokens[0]:
                  print("Pred: ", tok_history)
                  return
          else: return
      else: return
    print("Pred: ", tok_history)

def beam_search(inp, model, emb, tok, num_beams=10):
    beams = num_beams*[(inp, 1.0)]

def init():
    max_seq_len = 256
    emb_dim = 64
    with open("./ireal_url", "r") as f: tunes = Tune.parse_ireal_url(f.read())
    chord_types = set()
    ireal_set_add(tunes, chord_types)
    corpus =  get_corpus(tunes)
    #for tune in tunes: tok.add_tokens(list(tune.measures_as_strings)) 
    tok.add_special_tokens(["%BOS%", "%EOS%"])# "%PAD%"])#"*A", "*B", "*C", "*D", "{", "}", "(", ")", "<", ">", "T34", "T44", "T64", "T54", "T12", "T22", "[BOS]", "[EOS]", "[PAD]" ])
    #tok.add_tokens(list(chord_types))
    tok.train_from_iterator(corpus, trainer=trainer)
    src_vocab_size = target_vocab_size = tok.get_vocab_size()
    num_epochs = 5
    return max_seq_len, torch.device('cuda'), chord_types, tunes, tok, src_vocab_size,corpus, emb_dim, num_epochs

def main():
    max_seq_len, device, chord_types, tunes, tok, vocab_size, corpus, emb_dim, num_epochs = init()
    Embedding    = torch.nn.Embedding(vocab_size, emb_dim).cuda()
    RotEmbedding = RotaryEmbedding(emb_dim).cuda()
    PosEnc = pmc.PositionalEncoding(emb_dim).cuda()
    model = torch.nn.Transformer(d_model=emb_dim, batch_first=True, dim_feedforward=2**8, dropout=0.1, nhead=4, num_encoder_layers=8, num_decoder_layers=8, norm_first=False, bias=True).cuda()
    #model_test = pmc.Transformer(emb_dim, 4, 8, 12, 2**10, vocab_size)
    params = list(model.parameters())+list(Embedding.parameters()) + list(PosEnc.parameters()) + list(RotEmbedding.parameters()) #+ list(#PosEnc.parameters())
    optimizer = torch.optim.Adam(params, lr=0.0001)
    for epoch in range(num_epochs):
        for i, rows in enumerate(corpus):
            row = [tok.encode(r[0]).ids[0] for r in rows]
            model.train()
            for sss in range(len(row)-1, len(row)):
                if (sss%100 == 0): print(f"Epoch: {100*(i/len(corpus))}%")
                src = torch.concat((
                    Tensor(row).type(torch.int64).cuda()[:sss-1],
                    Tensor((max_seq_len-(sss-1))*[tok.token_to_id("%EOS%")]).type(torch.int64).cuda()
                ))
                trg = torch.concat((
                    Tensor(row).type(torch.int64).cuda()[:sss],
                    Tensor((max_seq_len-(sss))*[tok.token_to_id("%EOS%")]).type(torch.int64).cuda()
                ))# Teacher Forcing: #sub_phrase_end_index:].cuda()#src.roll(roll_i).cuda()
                tgt_mask = torch.triu(torch.ones(trg.size(0), trg.size(0)), diagonal=1).bool()
                optimizer.zero_grad()
                src_emb = PosEnc(RotEmbedding(Embedding(src).cuda()))
                trg_emb = PosEnc(RotEmbedding(Embedding(trg).cuda()))
                outputs = model(src_emb, trg_emb, src_is_causal=True, tgt_is_causal=True,
                                src_mask=model.generate_square_subsequent_mask(src.size(0)),
                                tgt_mask=model.generate_square_subsequent_mask(trg.size(0)))
                loss = F.cross_entropy(outputs.view(-1, emb_dim), trg_emb.view(-1, emb_dim))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1)
                optimizer.step()
                print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item()}")
            model.eval()
            generate_start_with(["%BOS%"], model, Embedding, tok)
        torch.save(model, f"{epoch}.pt")
        pdb.set_trace()

if __name__ == "__main__": main()
