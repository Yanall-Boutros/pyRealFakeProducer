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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=256):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(512.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def get_corpus(tunes):
    hist = {}
    rows = []
    for tune in tunes:
        rows.append(f"%BOS% {tune.chord_string} %EOS%")
        #rows.append(tune.chord_string)
        for char in tune.chord_string:
            if char in hist: hist[char] += 1
            else: hist[char] = 1
    return rows


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
  similarities = torch.nn.functional.cosine_similarity(embedding.unsqueeze(0), embedding_weights)

  # Get the index of the most similar embedding
  most_similar_index = torch.argmax(similarities).item()
  token = tok.id_to_token(most_similar_index)
  # Retrieve the token based on the index (assuming vocabulary order matches weight order)
  return token


def generate_start_with(tokens, model, Embedding, PosEnc, tok):
    vocab = tok.get_vocab()
    #pdb.set_trace()
    tok_history = []
    tok_history.extend(tokens)
    print(tokens)
    timeout = 0
    #pdb.set_trace()
    while all([token in vocab for token in tok_history]) and len(tok_history) < 12:
      embedding = Embedding(Tensor([tok.token_to_id(token) for token in tok_history]).type(torch.int64).cuda())
      next_tokens = [embedding_to_token(n_emb, Embedding) for n_emb in model.decoder(Embedding(Tensor([tok.token_to_id(token) for token in tok_history]).type(torch.int64).cuda()), model.encoder(Embedding(Tensor([tok.token_to_id(token) for token in tok_history]).type(torch.int64).cuda()))) ]
      tok_history.append(next_tokens[0])
      if "%EOS%" in tok_history:
          print("Pred: ", tok_history)
          return
    print("Pred: ", tok_history)

def beam_search(inp, model, emb, tok, num_beams=10):
    beams = num_beams*[(inp, 1.0)]

def main():
    max_seq_len = 256
    device = torch.device('cuda')
    chord_types = set()
    with open("./ireal_url", "r") as f: tunes = Tune.parse_ireal_url(f.read())
    ireal_set_add(tunes, chord_types)
    for tune in tunes: tok.add_tokens(list(tune.measures_as_strings)) 
    tok.add_special_tokens(["%BOS%", "%EOS%", "%PAD%"])#"*A", "*B", "*C", "*D", "{", "}", "(", ")", "<", ">", "T34", "T44", "T64", "T54", "T12", "T22", "[BOS]", "[EOS]", "[PAD]" ])
    tok.add_tokens(list(chord_types))
    corpus =  get_corpus(tunes)
    tok.train_from_iterator(corpus, trainer=trainer)
    src_vocab_size = target_vocab_size = tok.get_vocab_size()
    emb_dim = 64
    Embedding = torch.nn.Embedding(src_vocab_size, emb_dim)
    #RotEmbedding = RotaryEmbedding(emb_dim)
    Embedding.cuda()
    PosEnc = PositionalEncoding(emb_dim).cuda()
    #RotEmbedding.cuda()
    model = torch.nn.Transformer(d_model=emb_dim, batch_first=True, dim_feedforward=2**8, dropout=0.1, nhead=4, num_encoder_layers=8, num_decoder_layers=8, norm_first=False, bias=True)
    #mh_attention = model.encoder.layers[0].self_attn
    #model.encoder.layers[0].self_attn = RotaryAttention(mh_attention, RotEmbedding).cuda()
    model.cuda()
    params = list(model.parameters())+list(Embedding.parameters()) + list(PosEnc.parameters())
    optimizer = torch.optim.Adam(params, lr=0.00001)# + list(RotEmbedding.parameters()), lr=0.0001)
    num_epochs = 5
    for epoch in range(num_epochs):
        #pdb.set_trace()
        for i, row in enumerate(tok.encode_batch(corpus)):
#            print("Source/Target: ", [tok.id_to_token(id) for id in row.ids])
#            sss is split_sub_string, and is an index to split a target phrase in 2. The first part is the context, the last part is what should be predicted
            model.train()
            for sss in range(len(row.ids)-1, len(row.ids)):
                if (sss%100 == 0): print(f"Epoch: {100*(i/len(corpus))}%")

                src = torch.concat((Tensor(row.ids).type(torch.int64).cuda()[:sss-1], Tensor((max_seq_len-(sss-1))*[tok.token_to_id("%EOS%")]).type(torch.int64).cuda()))
                trg = torch.concat(
                    (Tensor(row.ids).type(torch.int64).cuda()[:sss], Tensor((max_seq_len-(sss))*[tok.token_to_id("%EOS%")]).type(torch.int64).cuda()))#sub_phrase_end_index:].cuda()#src.roll(roll_i).cuda()

                optimizer.zero_grad()
                src_emb = Embedding(src).cuda()
                trg_emb = Embedding(trg).cuda()
                src_emb = PosEnc(src_emb)
                trg_emb = PosEnc(trg_emb)
                #src_emb = RotEmbedding(src_emb)
                #trg_emb = RotEmbedding(trg_emb)
                outputs = model(src_emb, trg_emb)
                #loss = criterion(outputs.view(-1, 256), trg_emb.view(-1, 256))
                #pdb.set_trace()
                loss = torch.nn.functional.cross_entropy(outputs.view(-1, emb_dim), trg_emb.view(-1, emb_dim))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1)
                optimizer.step()
                print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item()}")
            model.eval()
            generate_start_with(["%BOS%", "A-"], model, Embedding, PosEnc, tok)
        torch.save(model, f"{epoch}.pt")
        pdb.set_trace()

if __name__ == "__main__": main()
