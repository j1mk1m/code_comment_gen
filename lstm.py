import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import os
import random

SOS_token = 0
EOS_token = 1

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, embbed_dim, dropout):
        super(Encoder, self).__init__()
      
        self.input_dim = input_dim
        self.embbed_dim = embbed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_dim, self.embbed_dim)
        self.lstm = nn.LSTM(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, hidden = self.lstm(embedded)
        return outputs, hidden 

class Decoder(nn.Module):
    def __init__(self, attention, output_dim, hidden_dim, embbed_dim, num_layers, dropout):
        super(Decoder, self).__init__()

        self.embbed_dim = embbed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.attention = attention 
        self.embedding = nn.Embedding(output_dim, self.embbed_dim)
        self.lstm = nn.LSTM(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers)
        self.out = nn.Linear(self.hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropout)
      
    def forward(self, x, h, outputs):
        x = x.unsqueeze(0)
        embedded = self.dropout(self.embedding(x))
        a = self.attention(h, outputs)
        a = a.unsqueeze(1)
        outputs = outputs.permute(1, 0, 2)
        # TODO
        out, hidden = self.lstm(embedded, h)       
        prediction = self.softmax(self.out(out[0]))
        
        return prediction, hidden

class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hidden_dim * 2) + dec_hidden_dim, dec_hidden_dim)
        self.out = nn.Linear(dec_hidden_dim, 1)

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, decoder hidden dim]
        # encoder_outputs = [source_length, batch size, encoder hidden dim * 2]
        batch_size = encoder_outputs.shape[1]
        src_length = encoder_outputs.shape[0]
        # repeat decoder hidden state src_length times
        hidden = hidden.unsqueeze(1).repeat(1, src_length, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # hidden = [batch size, src length, decoder hidden dim]
        # encoder_outputs = [batch size, src length, encoder hidden dim * 2]
        energy = torch.tanh(self.attn_fc(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch size, src length, decoder hidden dim]
        attention = self.out(energy).squeeze(2)
        # attention = [batch size, src length]
        return torch.softmax(attention, dim=1)

class Seq2Seq(nn.Module):
    def __init__(self, code_encoder, ast_encoder, doc_encoder, decoder, device):
        super().__init__() 
        self.code_encoder = code_encoder
        self.ast_encoder = ast_encoder
        self.doc_encoder = doc_encoder
        self.decoder = decoder
        self.device = device
     
    def forward(self, source, ast, doc, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[1]
        target_length = target.shape[0]
        vocab_size = self.decoder.output_dim
      
        outputs = torch.zeros(target_length, batch_size, vocab_size).to(self.device)

        code_outputs, code_hidden = self.code_encoder(source)
        enc_outputs, enc_hidden = code_outputs, code_hidden
        if self.ast_encoder is not None:
            ast_outputs, ast_hidden = self.ast_encoder(ast)
            enc_outputs = torch.concat((enc_outputs, ast_outputs), dim=0)
            enc_hidden = torch.concat((enc_hidden, ast_hidden), dim=0)
        if self.doc_encoder is not None:
            doc_outputs, doc_hidden = self.doc_encoder(doc)
            enc_outputs = torch.concat((enc_outputs, doc_outputs), dim=0)
            enc_hidden = torch.concat((enc_hidden, doc_hidden), dim=0)

        decoder_hidden = enc_hidden
        decoder_input = target[0,:] 

        for t in range(target_length):   
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, enc_outputs)
            outputs[t] = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            pred = decoder_output.argmax(1)
            decoder_input = (target[t] if teacher_force else pred)

        return outputs