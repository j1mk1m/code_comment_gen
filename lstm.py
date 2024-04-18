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
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, x):
        # print("Encoder forward")
        # x = (L, B, D)
        embedded = self.dropout(self.embedding(x))
        # print("embedded", embedded.shape)
        # embedded = (L, B, E)
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs = (L, B, 2 * H), hidden = (2 * num_layers, B, H)
        hidden = torch.concat((hidden[-2], hidden[-1]), dim=1).unsqueeze(0)
        cell = torch.concat((cell[-2], cell[-1]), dim=1).unsqueeze(0)
        # hidden, cell = (1, B, H)
        # print("hidden", hidden.shape)
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
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.softmax = nn.Softmax(dim=1)
      
    def forward(self, x, hidden, context):
        # print("Decoder forward")
        # x = (B), hidden = (1, B, H_dec) x2, context = (1, B, H_enc)
        x = x.unsqueeze(0) # x = (1, B)
        embedded = self.dropout(self.embedding(x)) # embedded = (1, B, E_dec)
        # print("embedded", embedded.shape)
        # print("context", context.shape)
        # print("hidden", hidden[0].shape, hidden[1].shape)
        out, (hidden, cell) = self.lstm(torch.cat((embedded, context), dim=2), hidden)
        # out = (1, B, H_dec)
        out, context, embedded = out[0], context[0], embedded[0]
        prediction = self.softmax(self.out(torch.concat((embedded, context, out), dim=1)))
        # prediction = (B, D_out), hidden = (B, H_dec)
        # print("prediction", prediction.shape)
        # print("hidden", hidden.shape)
        # print("cell", cell.shape)
        return prediction, (hidden, cell)

class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hidden_dim * 2) + dec_hidden_dim, dec_hidden_dim)
        self.out = nn.Linear(dec_hidden_dim, 1) 

    def forward(self, hidden, encoder_outputs):
        # hidden = (1, B, Ho)
        # encoder_outputs = (L, B, H * 2)
        # print("Attention forward")
        # print("hidden", hidden.shape)
        # print("encoder_outputs", encoder_outputs.shape)
        batch_size = encoder_outputs.shape[1] # B
        src_length = encoder_outputs.shape[0] # L
        hidden = hidden.permute(1,0,2).expand(-1, src_length, -1) # (B, L, Ho)
        # print("hidden", hidden.shape)
        encoder_outputs = rearrange(encoder_outputs, "l b h -> b l h") # (B, L, H*2)
        # print("encoder_outputs", encoder_outputs.shape)
        energy = torch.tanh(self.attn(torch.concat((hidden, encoder_outputs), dim=2)))
        # energy = (B, L, Ho) 
        # print("energy", energy.shape)
        attention = self.out(energy)[:,:,0]
        alpha = torch.softmax(attention, dim=1)
        # alpha = (B, L)
        # print("alpha", alpha.shape)
        context = alpha.unsqueeze(1) @ encoder_outputs
        # context = (B, 1, H*2)
        # print("context", context.shape)
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
        self.linear = nn.Linear(2*self.encoder_hidden_dim, self.decoder_hidden_dim)
 
    def forward(self, source, comment, teacher_forcing_ratio=1):
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
 
        decoder_hidden, decoder_cell = self.linear(torch.concat(enc_hidden, dim=2)), self.linear(torch.concat(enc_cell, dim=2))
        decoder_input = comment[0,:] # <sos> tokens
        
        for t in range(output_length): 
            contexts = []
            for i in range(len(self.encoders)):
                enc_out = enc_outputs[i]
                _, context = self.attentions[i](decoder_hidden, enc_out)
                contexts.append(context)
            context = torch.concat(contexts, dim=2)
            # print("context", context.shape)
            # print("decoder_hidden", decoder_hidden.shape)
            # print("decoder_cell", decoder_cell.shape)
            # print("decoder_input", decoder_input.shape)

            decoder_output, (decoder_hidden, decoder_cell) = self.decoder(decoder_input, (decoder_hidden, decoder_cell), context)
            outputs[t] = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            pred = decoder_output.argmax(1)
            decoder_input = (comment[t] if teacher_force else pred) # or decoder_output

        return outputs

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, attention, device):
        super().__init__() 
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.device = device

        self.decoder_hidden_dim = self.decoder.hidden_dim
        self.linear = nn.Linear(2*self.encoder.hidden_dim, self.decoder_hidden_dim) 
 
    def forward(self, source, comment, teacher_forcing_ratio=1):
        # source = (L, B, D)
        # comment/output = (Lo, B)
        batch_size = source.shape[1]
        output_length = comment.shape[0]
        output_dim = self.decoder.output_dim
      
        # outputs = (Lo, B, Do)
        outputs = torch.zeros(output_length, batch_size, output_dim).to(self.device)

        enc_out, (hid, cell) = self.encoder(source)
 
        decoder_hidden, decoder_cell = self.linear(hid), self.linear(cell)
        decoder_input = comment[0,:] # <sos> tokens
        
        for t in range(output_length): 
            _, context = self.attention(decoder_hidden, enc_out)
            # print("context", context.shape)
            # print("decoder_hidden", decoder_hidden.shape)
            # print("decoder_cell", decoder_cell.shape)
            # print("decoder_input", decoder_input.shape)

            decoder_output, (decoder_hidden, decoder_cell) = self.decoder(decoder_input, (decoder_hidden, decoder_cell), context)
            outputs[t] = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            pred = decoder_output.argmax(1)
            decoder_input = (comment[t] if teacher_force else pred) # or decoder_output

        return outputs