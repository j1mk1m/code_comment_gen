import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd

import os
import random

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 20

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
       super().__init__()
      
       self.encoder = encoder
       self.decoder = decoder
       self.device = device
     
    def forward(self, source, target, teacher_forcing_ratio=0.5):
       input_length = source.size(0) #get the input length (number of words in sentence)
       batch_size = target.shape[1] 
       target_length = target.shape[0]
       vocab_size = self.decoder.output_dim
      
       outputs = torch.zeros(target_length, batch_size, vocab_size).to(self.device)

       for i in range(input_length):
           encoder_output, encoder_hidden = self.encoder(source[i])

       decoder_hidden = encoder_hidden.to(self.device)
  
       decoder_input = torch.tensor([SOS_token], device=self.device)

       for t in range(target_length):   
           decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
           outputs[t] = decoder_output
           teacher_force = random.random() < teacher_forcing_ratio
           topv, topi = decoder_output.topk(1)
           input = (target[t] if teacher_force else topi)
           if(teacher_force == False and input.item() == EOS_token):
               break

       return outputs