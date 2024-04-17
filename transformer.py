"""
- This is da transformer model (with multi-headed self attention !!)
"""

import torch
import torch.nn as nn

class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d: int, base: int = 10_000):
        super().__init__()
        self.d = d
        self.base = base
        self.cos_mat = None
        self.sin_mat = None

    def _build_cache(self, x: torch.Tensor):
        _, _, T, _ = x.size()
        i_vals = torch.arange(0, self.d//2, 1)
        exp_vals = -2 * i_vals / self.d
        theta_vals = torch.pow(self.base, exp_vals)
        theta_2d = theta_vals.repeat(T, 1)

        pos_col = torch.arange(1, T + 1, 1).unsqueeze(1)
        half_mat = theta_2d*pos_col

        full_mat = torch.cat((half_mat, half_mat), 1)
        self.cos_mat = torch.cos(full_mat)
        self.sin_mat = torch.sin(full_mat)

    def forward(self, x: torch.Tensor):
        if self.cos_mat is None:
            self._build_cache(x)
        _, _, T, _ = x.size()
        h1, h2 = torch.split(x, self.d//2, -1)
        sin_input = torch.cat((-1*h2, h1), 3)
        cos_input = torch.clone(x)

        cos_mat_trunc = self.cos_mat[:T, :]
        sin_mat_trunc = self.sin_mat[:T, :]
        out = cos_input*cos_mat_trunc + sin_input*sin_mat_trunc
        return out
    
class Mask(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask = None
        self.mask_len = -1

    def get_mask(self, len):
        if self.mask is None or self.mask_len != len:
            self.mask_len = len
            all_negs = torch.full((len, len), float('-inf'))
            self.mask = torch.triu(all_negs, diagonal=1)
        return self.mask


class Transformer(nn.Module):
    def __init__(self, n_embed, dk=512, num_heads=8, num_encode=6, num_decode=6, dropout=0.1):
        super().__init__()
        self.n_embed = n_embed
        self.dk = dk
        self.num_heads = num_heads
        self.num_encode = num_encode
        self.num_decode = num_decode
        self.dropout = dropout

        self.pos_embed = RotaryPositionalEmbeddings(dk)
        self.encoder = nn.Embedding(n_embed, dk)
        self.decoder = nn.Embedding(n_embed, dk)
        self.masker = Mask()

        self.transformer = nn.Transformer(dk, num_heads, num_encode, num_decode, dropout=dropout)
        self.linear_out = nn.Linear(dk, n_embed)

    def forward(self, input, output):
        encode_input = self.pos_embed(self.encoder(input))
        decode_output = self.pos_embed(self.decoder(output))
        transform_out = self.transformer(encode_input, decode_output, src_mask=self.masker.get_mask(len(input)), tgt_mask=self.masker.get_mask(len(output)))
        return self.linear_out(transform_out)


