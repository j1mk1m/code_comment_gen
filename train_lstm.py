import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse
import numpy as np
import pandas as pd

from lstm import *

# overall configs
INPUT_DIM = 100
OUTPUT_DIM = 100
# LSTM configs
CODE_ENC_EMBED_DIM = 50
CODE_ENC_HIDDEN_DIM = 128
CODE_ENC_NUM_LAYERS = 1
CODE_ENC_DROPOUT = 0.5
AST_ENC_EMBED_DIM = 50
AST_ENC_HIDDEN_DIM = 128
AST_ENC_NUM_LAYERS = 1
AST_ENC_DROPOUT = 0.5
DOC_ENC_EMBED_DIM = 50
DOC_ENC_HIDDEN_DIM = 128
DOC_ENC_NUM_LAYERS = 1
DOC_ENC_DROPOUT = 0.5
DEC_EMBED_DIM = 50
DEC_HIDDEN_DIM = 128
DEC_NUM_LAYERS = 1
DEC_DROPOUT = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, epoch, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for e in range(epoch):
        # Code for training the model goes here
        pass

def test(model):
    pass

if __name__=="__main__":
    parser = argparse.ArgumentParser("Train script for Code Comment Generation")
    parser.add_argument("--epoch", type=int, default=10000, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--model", type=str, default="base", choices=["base", "AST", "Doc", "Full"], help="Model type to use (base, AST, Doc, Full)")
    args = parser.parse_args()

    enc_hidden_dim = CODE_ENC_HIDDEN_DIM
    code_encoder = Encoder(INPUT_DIM, CODE_ENC_HIDDEN_DIM, CODE_ENC_NUM_LAYERS, CODE_ENC_EMBED_DIM, CODE_ENC_DROPOUT)
    if args.model == "Full" or args.model == "AST":
        ast_encoder = Encoder(INPUT_DIM, CODE_ENC_HIDDEN_DIM, CODE_ENC_NUM_LAYERS, CODE_ENC_EMBED_DIM, CODE_ENC_DROPOUT)
        enc_hidden_dim += AST_ENC_HIDDEN_DIM
    if args.model == "Full" or args.model == "Doc":
        doc_encoder = Encoder(INPUT_DIM, CODE_ENC_HIDDEN_DIM, CODE_ENC_NUM_LAYERS, CODE_ENC_EMBED_DIM, CODE_ENC_DROPOUT)
        enc_hidden_dim += DOC_ENC_HIDDEN_DIM
    attention = Attention(enc_hidden_dim, DEC_HIDDEN_DIM)
    decoder = Decoder(attention, OUTPUT_DIM, DEC_HIDDEN_DIM, DEC_EMBED_DIM, DEC_NUM_LAYERS)
    model = Seq2Seq(code_encoder, ast_encoder, doc_encoder, decoder, device)

    train(model, args.epoch, args.learning_rate)
    test(model)

