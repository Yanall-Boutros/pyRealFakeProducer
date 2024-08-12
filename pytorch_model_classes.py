import torch.nn as nn
import torch
import torch.nn.functional as F
import math,copy,re
import warnings
import pandas as pd
import numpy as np
from torch import Tensor
from utils import ireal_set_add
from rotary_embedding_torch import RotaryEmbedding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=256):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len)[:, np.newaxis]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Transformer_1(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, vocab_size, dropout=0.1):  
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True),
            num_layers = num_encoder_layers)

        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True),
            num_layers=num_decoder_layers)

        # Embedding layers (example)
        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)

        # Positional encoding (example)
        #self.pos_encoder = RotaryEmbedding(d_model)#PositionalEncoding(d_model, dropout)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Output layer (example)
        self.generator = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        src = self.src_embed(src)
        src = self.pos_encoder(src)
        memory = self.encoder(src) #src_mask=src_mask, src_key_padding_mask=src_padding_mask)

        tgt = self.tgt_embed(tgt)
        tgt = self.pos_encoder(tgt)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=None, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_padding_mask)  

        output = self.generator(output)
        return output

## Example usage of attention bias:
#attention_mask = torch.zeros(batch_size, seq_len, seq_len).bool()
#attention_mask = attention_mask.masked_fill(mask_condition, True)
#attention_bias = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
#
## Example usage of nested tensors:
#nested_tensor = torch.nested_tensor(tensors)
