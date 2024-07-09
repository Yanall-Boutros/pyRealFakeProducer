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
#tok = Tokenizer(BPE())
tok = Tokenizer(WordPiece())
#tok = tokenizers.ByteLevelBPETokenizer()
#trainer = BpeTrainer()#special_tokens=["[BOS]", "[EOS]"])
trainer = WordPieceTrainer()#special_tokens=["[BOS]", "[EOS]"])
tok.pre_tokenizer = Whitespace()

# Useful for getting sense of unique chord types (~1000), max length of measures (122)
def ireal_set_add(tunes, trgt_set):
    measure_len_to_tunes = {}
    measures = set()
    for tune in tunes:
        ts = str(tune.time_signature[0])+str(tune.time_signature[1])
        measures_string = tune.measures_as_strings
        measure_len = len(measures_string)
        if measure_len in measure_len_to_tunes: measure_len_to_tunes[measure_len].append(tune)
        else: measure_len_to_tunes[measure_len] = [ tune ]
        for measure in measures_string: measures.add(measure)
        chords_string = tune.chord_string[tune.chord_string.find(ts)+2:]
        for chords in chords_string.split("|"):
            for chord in chords.split(" "):
                if chord == '': continue
                if len(chord) == 0: continue
                if any([bad_char in chord for bad_char in "{}[]*<()>TW"]): continue
                if any([bad_word in chord for bad_word in [ "edal","Q", "1st", "use", "alt", "till", "takes", "2nd", "over", "Coda", "minor", "free", "only", "chorus", "feel", "the", "eep", "in", "every", "olos", "is", "out", "on", "AABA", "ow", "time", "ops", "by", "chords", "of", "D.C.", "Miles", "or", "double","key", "Feel", "until", "CD"]]): continue
                if chord[0] == "N": chord = chord[chord.find("l")+1:]
                if chord[0] == "n": continue
                if chord[0] == "s": chord=chord[1:]
                if len(chord) == 0: continue
                if 'p' in chord[0]: chord =chord[1:]
                if len(chord) == 0: continue
                if "pps" in chord: chord=chord[3:]
                if len(chord) == 0: continue
                if "ps" in chord: chord=chord[2:]
                if len(chord) == 0: continue
                if "l" in chord[0]: chord = chord[1:]
                if len(chord) == 0: continue
                if "N" in chord: chord = chord[2:]
                if len(chord) == 0: continue
                if chord == "n": continue
                if 'p' in chord[0]: chord = chord[1:]
                if len(chord) == 0: continue
                if 'n' in chord[0]: chord = chord[1:]
                if len(chord) == 0: continue
                if 's' in chord[0]: chord = chord[1:]
                if len(chord) == 0: continue
                if 'f' in chord[0]: chord = chord[1:]
                if len(chord) == 0: continue
                if 'U' in chord[0]: chord = chord[1:]
                if len(chord) == 0: continue
                #if chord[-1] == "^": continue
                if chord[0] == "f": chord=chord[1:]
                if len(chord) == 0: continue
                if chord[0] == "l": chord=chord[1:]
                if len(chord) == 0: continue
                if 'p' in chord[0]: chord = chord[1:]
                if len(chord) == 0: continue
                if chord == "r": continue
                if chord == "x": continue
                if chord == "S": continue
                if chord[0].isnumeric(): chord = chord[1:]
                if len(chord) == 0: continue
                trgt_set.add(chord)
    return measure_len_to_tunes, measures

def get_corpus(tunes):
    hist = {}
    rows = []
    for tune in tunes:
        rows.append(tune.chord_string)
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

def main():
    device = torch.device('cuda')
    chord_types = set()
    with open("./ireal_url", "r") as f: tunes = Tune.parse_ireal_url(f.read())
    measure_len_to_tunes, measures = ireal_set_add(tunes, chord_types)    
    tok.add_tokens(list(chord_types))
    tok.add_special_tokens(["*A", "*B", "*C", "*D", "{", "}", "(", ")", "<", ">", "T34", "T44", "T64", "T54", "T12", "T22", "[BOS]", "[EOS]"])
    corpus =  get_corpus(tunes)
    tok.train_from_iterator(corpus, trainer=trainer)
    src_vocab_size = target_vocab_size = tok.get_vocab_size()
    seq_length = 256
    num_layers = 2
    Embedding = torch.nn.Embedding(src_vocab_size, 256)
    Embedding.cuda()
    model = torch.nn.Transformer(d_model=256)
    #model = Transformer(embed_dim=128, src_vocab_size=src_vocab_size, 
    #                target_vocab_size=target_vocab_size, seq_length=seq_length,
    #                num_layers=num_layers, expansion_factor=2, n_heads=2) 
    model.cuda()
    #criterion = torch.nn.NLLLoss()  # ignore padding token
    #criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        for i, row in enumerate(tok.encode_batch(corpus)):
            print(f"Epoch: {i/len(corpus)}%")
            for sub_phrase_end_index in range(1, len(row)-1):
                src = Tensor(row.ids).type(torch.int64).cuda()#.unsqueeze(0)
                trg = Tensor(row.ids).type(torch.int64)[sub_phrase_end_index:].cuda()#src.roll(roll_i).cuda()
                src_emb = Embedding(src).cuda()
                trg_emb = Embedding(trg).cuda()
                outputs = model(src_emb, trg_emb)
                #loss = criterion(outputs.view(-1, 256), trg_emb.view(-1, 256))
                #pdb.set_trace()
                loss = torch.nn.functional.cross_entropy(outputs.view(-1, 256), trg_emb.view(-1, 256))
                loss.backward()
                optimizer.step()
                print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    pdb.set_trace()

if __name__ == "__main__": main()
