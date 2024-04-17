import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from einops import einsum, rearrange
import numpy as np
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, num_layers=1, dropout=0.1):
        super(Encoder, self).__init__()
      
        self.input_dim = input_dim # D
        self.embed_dim = embed_dim # E
        self.hidden_dim = hidden_dim # H
        self.num_layers = num_layers # num_layers

        self.embedding = nn.Embedding(input_dim, self.embed_dim) 
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = (L, B, D)
        embedded = self.dropout(self.embedding(x))
        # embedded = (L, B, E)
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs = (L, B, 2 * H), hidden = (2 * num_layers, B, H)
        hidden = torch.concat((hidden[-2], hidden[-1]), dim=1)
        cell = torch.concat((cell[-2], cell[-1]), dim=1)
        # hidden = (B, H)
        return outputs, (hidden, cell) 

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, embed_dim, context_dim, dropout=0.1):
        super(Decoder, self).__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.context_dim = context_dim

        self.embedding = nn.Embedding(output_dim, self.embed_dim)
        self.lstm = nn.LSTM(self.context_dim + self.embed_dim, self.hidden_dim)
        self.out = nn.Linear(self.context_dim + self.embed_dim + self.hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
      
    def forward(self, x, hidden, context):
        # x = (B, D_out), hidden = (B, H_dec) x2, context = (1, B, H_enc)
        x = x.unsqueeze(0) # x = (1, B, D_out)
        embedded = self.dropout(self.embedding(x)) # embedded = (1, B, E_dec)
        out, (hidden, cell) = self.lstm(torch.cat((embedded, context), dim=2), hidden)
        # out = (1, B, H_dec)
        out, context, embedded = out[0], context[0], embedded[0]
        prediction = self.softmax(self.out(torch.concat((embedded, context, out), dim=1)))
        # prediction = (B, D_out), hidden = (B, H_dec)
        return prediction, (hidden[0], cell[0])

class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hidden_dim * 2) + dec_hidden_dim, dec_hidden_dim)
        self.out = nn.Linear(dec_hidden_dim, 1)

    def forward(self, hidden, encoder_outputs):
        # hidden = (B, Ho)
        # encoder_outputs = (L, B, H * 2)
        batch_size = encoder_outputs.shape[1] # B
        src_length = encoder_outputs.shape[0] # L
        hidden = hidden.unsqueeze(1).expand(-1, src_length, -1) # (B, L, Ho)
        encoder_outputs = rearrange(encoder_outputs, "l b h -> b l h") # (B, L, H*2)
        energy = torch.tanh(self.attn(torch.concat((hidden, encoder_outputs), dim=2)))
        # energy = (B, L, Ho) 
        attention = self.out(energy)[:,:,0]
        alpha = torch.softmax(attention, dim=1)
        # alpha = (B, L)
        context = alpha.unsqueeze(1) @ encoder_outputs
        # context = (B, 1, H*2)
        return alpha, rearrange(context, "b o h -> o b h")

class MultiSeq2Seq(nn.Module):
    def __init__(self, encoders, decoder, attentions, device):
        super().__init__() 
        assert len(encoders) == len(attentions)
        self.encoders = encoders
        self.decoder = decoder
        self.attentions = attentions
        self.device = device

        self.encoder_hidden_dim = sum([encoder.hidden_dim for encoder in self.encoders])
        self.decoder_hidden_dim = self.decoder.hidden_dim
        self.linear = nn.Linear(self.encoder_hidden_dim, self.decoder_hidden_dim)
 
    def forward(self, source, comment, teacher_forcing_ratio=0.5):
        # source = list of tensors of shape (L, B, D)
        # e.g. code = (Lc, B, Dc), ast = (La, B, Da), doc = (Ld, B, Dd)
        assert len(source) == len(self.encoders)
        # comment/output = (Lo, B, _)
        batch_size = source[0].shape[1]
        output_length = comment.shape[0]
        output_dim = self.decoder.output_dim
      
        # outputs = (Lo, B, Do)
        outputs = torch.zeros(output_length, batch_size, output_dim).to(self.device)

        enc_outputs, enc_hidden, enc_cell = [], [], []
        for i in range(len(self.encoders)):
            encoder = self.encoders[i]
            input = source[i]
            out, (hid, cell) = encoder(input)
            enc_outputs.append(out)
            enc_hidden.append(hid)
            enc_cell.append(cell)
 
        decoder_hidden, decoder_cell = self.linear(torch.concat(enc_hidden, dim=1)), self.linear(torch.concat(enc_cell, dim=1))
        decoder_input = comment[0,:] # SOS tokens

        for t in range(output_length): 
            contexts = []
            for i in range(len(self.encoders)):
                enc_out = enc_outputs[i]
                _, context = self.attentions[i](decoder_hidden, enc_out)
                contexts.append(context)
            context = torch.concat(contexts, dim=2)

            decoder_output, (decoder_hidden, decoder_cell) = self.decoder(decoder_input, (decoder_hidden, decoder_cell), context)
            outputs[t] = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            pred = decoder_output.argmax(1)
            decoder_input = (comment[t] if teacher_force else pred) # or decoder_output

        return outputs