import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse
import wandb
import tqdm

from lstm import *
from evaluation import *
from dataset import *

# TODO Template index to token function
def idx_to_token(index):
    return None

"""
- index to token (for comments)
- token sequence to sentence (for comments)
"""

# Code Encoder
CODE_DIM = 64
CODE_ENC_EMBED_DIM = 50
CODE_ENC_HIDDEN_DIM = 128
CODE_ENC_DROPOUT = 0.5
# AST Encoder
AST_DIM = 64
AST_ENC_EMBED_DIM = 50
AST_ENC_HIDDEN_DIM = 128
AST_ENC_DROPOUT = 0.5
# Doc Encoder TODO
DOC_DIM = 64
DOC_ENC_EMBED_DIM = 50
DOC_ENC_HIDDEN_DIM = 128
DOC_ENC_DROPOUT = 0.5
# Decoder
OUTPUT_DIM = 100
DEC_EMBED_DIM = 50
DEC_HIDDEN_DIM = 128
DEC_NUM_LAYERS = 1
DEC_DROPOUT = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, dataloader, epoch, learning_rate, teacher_forcing_ratio):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for e in tqdm.tqdm(range(epoch)):
        # Code for training the model goes here
        total_loss = 0
        for batch_idx, (source, target) in enumerate(dataloader):
            # Your code to loop through batches in a torch Dataloader goes here
            optimizer.zero_grad()
            # TODO might need to split source into code, ast, doc
            outputs = model(source, target, teacher_forcing_ratio)
            # outputs = (Lo, B, Do)
            outputs = outputs.view(-1, outputs.shape[-1])
            target = target.view(-1)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Average loss {avg_loss}")
        wandb.log({"epoch": e, "loss": avg_loss})
    
    torch.save(model.state_dict(), f"model_{epoch}_{learning_rate}_{teacher_forcing_ratio}.pt")

def test(model, dataloader):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    targets = []
    with torch.no_grad():
        generations = []
        for idx, (source, target) in enumerate(dataloader):
            outputs = model(source, target, 0)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), target.view(-1))
            total_loss += loss.item()
            
            generated_ids = outputs.argmax(2) # (Lo, B) let B=1
            generated_ids = generated_ids.squeeze(1) # (Lo,)
            
            generation = []
            for index in generated_ids:
                generation.append(idx_to_token(index))
            generations.append(generation)

            targets.append(target.squeeze(1))
        avg_loss = total_loss / len(dataloader)
        wandb.log({"test_loss": avg_loss})
    bleu_score = evaluate(generations, targets)
    return generations, bleu_score

if __name__=="__main__":
    parser = argparse.ArgumentParser("Train script for Code Comment Generation")
    parser.add_argument("--epoch", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    parser.add_argument("--model", type=str, default="base", choices=["base", "AST", "Doc", "Full"], help="Model type to use (base, AST, Doc, Full)")
    args = parser.parse_args()

    wandb.init(project="CodeCommentGen", 
        config={
            "learning_rate": args.learning_rate, 
            "epoch": args.epoch, 
            "model": args.model,
            "CODE_DIM": CODE_DIM,
            "CODE_ENC_EMBED_DIM": CODE_ENC_EMBED_DIM,
            "CODE_ENC_HIDDEN_DIM": CODE_ENC_HIDDEN_DIM,
            "CODE_ENC_DROPOUT": CODE_ENC_DROPOUT,
            "AST_DIM": AST_DIM,
            "AST_ENC_EMBED_DIM": AST_ENC_EMBED_DIM,
            "AST_ENC_HIDDEN_DIM": AST_ENC_HIDDEN_DIM,
            "AST_ENC_DROPOUT": AST_ENC_DROPOUT,
            "DOC_DIM": DOC_DIM,
            "DOC_ENC_EMBED_DIM": DOC_ENC_EMBED_DIM,
            "DOC_ENC_HIDDEN_DIM": DOC_ENC_HIDDEN_DIM,
            "DOC_ENC_DROPOUT": DOC_ENC_DROPOUT,
            "OUTPUT_DIM": OUTPUT_DIM,
            "DEC_EMBED_DIM": DEC_EMBED_DIM,
            "DEC_HIDDEN_DIM": DEC_HIDDEN_DIM,
            "DEC_NUM_LAYERS": DEC_NUM_LAYERS,
            "DEC_DROPOUT": DEC_DROPOUT
        })

    # MODEL
    encoders = []
    attentions = []
    context_dim = CODE_ENC_HIDDEN_DIM

    # Code Encoder
    code_encoder = Encoder(CODE_DIM, CODE_ENC_HIDDEN_DIM, CODE_ENC_EMBED_DIM, dropout=CODE_ENC_DROPOUT)
    code_attention = Attention(CODE_ENC_HIDDEN_DIM, DEC_HIDDEN_DIM)
    encoders.append(code_encoder)
    attentions.append(code_attention)

    # AST Encoder
    if args.model == "Full" or args.model == "AST":
        ast_encoder = Encoder(AST_DIM, CODE_ENC_HIDDEN_DIM, CODE_ENC_EMBED_DIM, dropout=CODE_ENC_DROPOUT)
        ast_attention = Attention(AST_ENC_HIDDEN_DIM, DEC_HIDDEN_DIM)
        encoders.append(ast_encoder)
        attentions.append(ast_attention)
        context_dim += AST_ENC_HIDDEN_DIM

    # Doc Encoder
    if args.model == "Full" or args.model == "Doc":
        doc_encoder = Encoder(DOC_DIM, CODE_ENC_HIDDEN_DIM, CODE_ENC_EMBED_DIM, dropout=CODE_ENC_DROPOUT)
        doc_attention = Attention(DOC_ENC_HIDDEN_DIM, DEC_HIDDEN_DIM)
        encoders.append(doc_encoder)
        attentions.append(doc_attention)
        context_dim += DOC_ENC_HIDDEN_DIM

    # Decoder
    decoder = Decoder(OUTPUT_DIM, DEC_HIDDEN_DIM, DEC_EMBED_DIM, context_dim, DEC_NUM_LAYERS)

    # MultiSeq2Seq Model
    model = MultiSeq2Seq(encoders, decoder, attentions, device)

    # TODO DATALOADER
    train_loader = None
    test_loader = None

    # TRAIN and EVAL
    train(model, train_loader, args.epoch, args.learning_rate, args.teacher_forcing_ratio)
    generations, score = test(model, test_loader)
    print(f"BLEU-4 Score: {score}")
    wandb.log({"bleu_score": score})
    wandb.log({"generations": generations})

    with open("output.txt", 'w') as file:
        gens = [" ".join(gen) for gen in generations]
        file.writelines(gens)

