import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pandas as pd

import argparse
# import wandb
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
# Dataloader
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 1

# Code Encoder
CODE_DIM = 10000
CODE_ENC_EMBED_DIM = 64
CODE_ENC_HIDDEN_DIM = 128
CODE_ENC_DROPOUT = 0.5
# AST Encoder
AST_DIM = 64
AST_ENC_EMBED_DIM = 50
AST_ENC_HIDDEN_DIM = 128
AST_ENC_DROPOUT = 0.5
# Doc Encoder 
DOC_DIM = 64
DOC_ENC_EMBED_DIM = 50
DOC_ENC_HIDDEN_DIM = 128
DOC_ENC_DROPOUT = 0.5
# Decoder
OUTPUT_DIM = 10000
DEC_EMBED_DIM = 64
DEC_HIDDEN_DIM = 128
DEC_NUM_LAYERS = 1
DEC_DROPOUT = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, data):
        self.unknown_token = '<unk>'
        self.data = data
        input_encoded, input_vocab, input_rev_vocab = self.get_one_hot(data['processed_func_code_tokens'], 10000)
        output_encoded, output_vocab, output_rev_vocab = self.get_one_hot(data['processed_doc_tokens'], 10000)
        self.input_encoded = input_encoded
        self.output_encoded = output_encoded
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.input_vocab_rev = input_rev_vocab
        self.output_vocab_rev = output_rev_vocab

    def get_one_hot(self, series, vocab_size):
        self.input_data = list(series.apply(lambda x: list(x)))
        
        all_words = [word for sentence in series for word in sentence]
        word_freq = Counter(all_words)
        
        most_common_words = word_freq.most_common(vocab_size)
        
        vocabulary = {word: idx for idx, (word, _) in enumerate(most_common_words)}

        vocabulary[self.unknown_token] = len(vocabulary)
        
        reverse_vocab = {id:word for word,id in vocabulary.items()}
        return [self.sentence_to_one_hot(sentence, vocabulary) for sentence in series], vocabulary, reverse_vocab

    def sentence_to_one_hot(self, sentence, vocab):
        indices = [vocab.get(word, vocab[self.unknown_token]) for word in sentence]
        # return torch.nn.functional.one_hot(torch.tensor(indices).to(torch.int64), num_classes=len(vocab))
        return torch.tensor(indices)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return [self.input_encoded[idx]], self.output_encoded[idx]
    
def get_data_loader(series, test = False):
    dataset = CustomDataset(series)
    return DataLoader(dataset, batch_size=TEST_BATCH_SIZE if test == True else TRAIN_BATCH_SIZE, shuffle=True)

def train(model, dataloader, epoch, learning_rate, teacher_forcing_ratio):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for e in tqdm.tqdm(range(epoch)):
        # Code for training the model goes here
        total_loss = 0
        for batch_idx, (source, target) in enumerate(dataloader):
            # Your code to loop through batches in a torch Dataloader goes here
            for i in range(len(source)):
                source[i] = source[i].permute(1, 0)
            target = target.permute(1, 0)
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
        # wandb.log({"epoch": e, "loss": avg_loss})
    
    torch.save(model.state_dict(), f"model_{epoch}_{learning_rate}_{teacher_forcing_ratio}.pt")

def test(model, dataloader):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    targets = []
    with torch.no_grad():
        generations = []
        for idx, (source, target) in enumerate(dataloader):
            for i in range(len(source)):
                source[i] = source[i].permute(1, 0)
            target = target.permute(1, 0)
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
        # wandb.log({"test_loss": avg_loss})
    bleu_score = evaluate(generations, targets)
    return generations, bleu_score
    
if __name__=="__main__":
    parser = argparse.ArgumentParser("Train script for Code Comment Generation")
    parser.add_argument("--epoch", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    parser.add_argument("--model", type=str, default="base", choices=["base", "AST", "Doc", "Full"], help="Model type to use (base, AST, Doc, Full)")
    args = parser.parse_args()

    # wandb.init(project="CodeCommentGen", 
    #     config={
    #         "learning_rate": args.learning_rate, 
    #         "epoch": args.epoch, 
    #         "model": args.model,
    #         "CODE_DIM": CODE_DIM,
    #         "CODE_ENC_EMBED_DIM": CODE_ENC_EMBED_DIM,
    #         "CODE_ENC_HIDDEN_DIM": CODE_ENC_HIDDEN_DIM,
    #         "CODE_ENC_DROPOUT": CODE_ENC_DROPOUT,
    #         "AST_DIM": AST_DIM,
    #         "AST_ENC_EMBED_DIM": AST_ENC_EMBED_DIM,
    #         "AST_ENC_HIDDEN_DIM": AST_ENC_HIDDEN_DIM,
    #         "AST_ENC_DROPOUT": AST_ENC_DROPOUT,
    #         "DOC_DIM": DOC_DIM,
    #         "DOC_ENC_EMBED_DIM": DOC_ENC_EMBED_DIM,
    #         "DOC_ENC_HIDDEN_DIM": DOC_ENC_HIDDEN_DIM,
    #         "DOC_ENC_DROPOUT": DOC_ENC_DROPOUT,
    #         "OUTPUT_DIM": OUTPUT_DIM,
    #         "DEC_EMBED_DIM": DEC_EMBED_DIM,
    #         "DEC_HIDDEN_DIM": DEC_HIDDEN_DIM,
    #         "DEC_NUM_LAYERS": DEC_NUM_LAYERS,
    #         "DEC_DROPOUT": DEC_DROPOUT
    #     })

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

    # DATALOADER
    df_train = pd.read_pickle('./dataprocessing/df_train_reduced.pkl').head(1000)
    df_test = pd.read_pickle('./dataprocessing/df_test_reduced.pkl').head(200)

    train_loader = get_data_loader(df_train, test=False)
    test_loader = get_data_loader(df_test, test=True) 

    # TRAIN and EVAL
    train(model, train_loader, args.epoch, args.learning_rate, args.teacher_forcing_ratio)
    generations, score = test(model, test_loader)
    print(f"BLEU-4 Score: {score}")
    # wandb.log({"bleu_score": score})
    # wandb.log({"generations": generations})

    with open("output.txt", 'w') as file:
        gens = [" ".join(gen) for gen in generations]
        file.writelines(gens)

