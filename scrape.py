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
#tok = Tokenizer(BPE())
#trainer = BpeTrainer()#special_tokens=["<BOS>", "<EOS>"])
tok = Tokenizer(WordPiece())
trainer = WordPieceTrainer()#special_tokens=["<BOS>", "<EOS>"])
#tok.pre_tokenizer = Whitespace()
#from rotary_embedding_torch import RotaryEmbedding
import pytorch_model_classes as pmc

def get_corpus(tunes): return [["%BOS%"]+tune.measures_as_strings+["%EOS%"] for tune in tunes]
#def get_corpus(tunes): return [f"%BOS%{t.chord_string}%EOS%" for t in tunes]

def generate_square_subsequent_mask(sz): 
    return torch.log(torch.tril(torch.ones(sz,sz)))

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

def top_k_tokens(tok_list, enc, dec, emb, posenc, linlayer, softmax, tok, k=10):
    inp_embs = emb(Tensor([tok.token_to_id(t) for t in tok_list]).type(torch.int64).cuda())
    enc_out = enc(posenc(inp_embs), mask=generate_square_subsequent_mask(inp_embs.size(0)), is_causal=True)
    #dec_out = dec(posenc(inp_embs), enc_out, tgt_mask=generate_square_subsequent_mask(enc_out.size(0)), tgt_is_causal=True)
    dec_out = dec(inp_embs, enc_out, tgt_mask=generate_square_subsequent_mask(enc_out.size(0)), tgt_is_causal=True)
    lin_out = linlayer(dec_out)
    soft_out = softmax(lin_out)
    top_k_toks = torch.topk(soft_out, k)
    top_toks = [tok.id_to_token(id) for id in top_k_toks.indices[0]]
    return top_toks, top_k_toks.values[0]

def beam_search(inp, enc, dec, emb, posenc, linlayer, softmax, tok, k=10):
    #beams = k*[([inp], [1.0])]
    beams = []
    for _ in range(k): beams.append((inp.copy(), [1.0]))
    top_toks, top_scores = top_k_tokens(inp, enc, dec, emb, posenc, linlayer, softmax, tok, k=10)
    for i, (t, scr) in enumerate(zip(top_toks, top_scores)):
        beams[i][0].append(t)
        beams[i][1][0] = scr.item()
    for _ in range(34):
        for i in range(len(beams)):
            if "%EOS%" in beams[i][0][-1]: continue
            next_possible_tokens = top_k_tokens(beams[i][0], enc, dec, emb, posenc, linlayer, softmax, tok, k=10)
            beams[i][0].append(next_possible_tokens[0][0])
            beams[i][1][0] = next_possible_tokens[1][0]
    for i in range(len(beams)): print(f"Beam {i}: {beams[i][0]}")
    return beams

def batchdata(corpus):
    max_seq_len=256
    for i, rows in enumerate(corpus):
        row = [tok.encode(r[0]).ids[0] for r in rows]
        for sss in range(1, len(row)):
            src = torch.concat((
                Tensor(row).type(torch.int64).cuda()[:sss-1],
                Tensor((max_seq_len-(sss-1))*[tok.token_to_id("%EOS%")]).type(torch.int64).cuda()
            ))
            trg = Tensor(row).type(torch.int64).cuda()[sss]
            #trg = torch.concat((
            #    Tensor(row).type(torch.int64).cuda()[:sss],
            #    Tensor((max_seq_len-(sss))*[tok.token_to_id("%EOS%")]).type(torch.int64).cuda()
            #))# Teacher Forcing: #sub_phrase_end_index:].cuda()#src.roll(roll_i).cuda()
            #tgt_mask = torch.triu(torch.ones(trg.size(0), trg.size(0)), diagonal=1).bool()
            src_emb = PosEnc(Embedding(src).cuda())
            trg_emb = Embedding(trg).cuda()

def init():
    max_seq_len = 126
    emb_dim = 16
    with open("./ireal_url", "r") as f: tunes = Tune.parse_ireal_url(f.read())
    chord_types = set()
    ireal_set_add(tunes, chord_types)
    corpus =  get_corpus(tunes)
    #pdb.set_trace()
    #for tune in tunes: tok.add_tokens(list(tune.measures_as_strings)) 
    tok.add_special_tokens(["%BOS%", "%EOS%"])# "%PAD%"])#"*A", "*B", "*C", "*D", "{", "}", "(", ")", "<", ">", "T34", "T44", "T64", "T54", "T12", "T22", "[BOS]", "[EOS]", "[PAD]" ])
    #tok.add_tokens(list(chord_types))
    tok.train_from_iterator(corpus, trainer=trainer)
    src_vocab_size = target_vocab_size = tok.get_vocab_size()
    num_epochs = 5
    return max_seq_len, torch.device('cuda'), chord_types, tunes, tok, src_vocab_size, corpus, emb_dim, num_epochs

def main():
    torch.set_grad_enabled(True)
    max_seq_len, device, chord_types, tunes, tok, vocab_size, corpus, emb_dim, num_epochs = init()
    model = pmc.Transformer_EncDec_A(emb_dim, vocab_size, num_encoder_layers=8, num_decoder_layers=8, dim_feedforward=2048, nhead=4, dropout=0.1)
    model.cuda()
    params = list(model.parameters())
    #model_test = pmc.Transformer(emb_dim, 4, 8, 12, 2**10, vocab_size).cuda()
    optimizer = torch.optim.Adam(params, lr=0.0001)
#    dataset = batchdata(corpus)
    for epoch in range(num_epochs):
        for i, rows in enumerate(corpus):
            row = [x.ids[0] for x in tok.encode_batch(rows)]

            model.train()
            for sss in range(len(row)-1, len(row)):
                #src = Tensor(row).type(torch.int64).cuda()[:sss-1]
                src = torch.concat((
                    Tensor(row).type(torch.int64).cuda()[:sss-1],
                    Tensor((max_seq_len-(sss-1))*[tok.token_to_id("%EOS%")]).type(torch.int64).cuda()
                ))
                #trg = Tensor(row).type(torch.int64).cuda()[sss]
                trg = Tensor([row[sss]]).type(torch.int64).cuda()
                trg = torch.concat((
                    Tensor(row).type(torch.int64).cuda()[:sss],
                    Tensor((max_seq_len-(sss))*[tok.token_to_id("%EOS%")]).type(torch.int64).cuda()
                ))
                out = model(src, trg)
                pdb.set_trace()

                optimizer.zero_grad()
                loss = F.cross_entropy(final_tok.view(-1), trg.type(torch.float64).view(-1))
                loss.requires_grad=True
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1)
                optimizer.step()
                print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item()}")
            model.eval()
            print("Learned from: ", "|".join(list(map(lambda x: tok.id_to_token(x), row))))
            pdb.set_trace()
            if ((i+1)%10) == 0: beam_search(["%BOS%", "A-"], transformer_encoder, transformer_decoder, Embedding, PosEnc, lin_layer, softmax, tok)
    pdb.set_trace()

if __name__ == "__main__": main()
