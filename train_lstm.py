import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse
import os
import wandb
import tqdm

from lstm import *
from evaluation import *
from dataset import *

# Code Encoder
CODE_DIM = 10000
CODE_ENC_EMBED_DIM = 64 
CODE_ENC_HIDDEN_DIM = 512
CODE_ENC_DROPOUT = 0.1
# AST Encoder
AST_DIM = 64
AST_ENC_EMBED_DIM = 64 
AST_ENC_HIDDEN_DIM = 256 
AST_ENC_DROPOUT = 0.1
# Doc Encoder 
DOC_DIM = 64
DOC_ENC_EMBED_DIM = 128 
DOC_ENC_HIDDEN_DIM = 256 
DOC_ENC_DROPOUT = 0.1
# Decoder
OUTPUT_DIM = 10000
DEC_EMBED_DIM = 64 
DEC_HIDDEN_DIM = 512 
DEC_NUM_LAYERS = 1
DEC_DROPOUT = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.1)

def train(model, dataloader, epoch, learning_rate, teacher_forcing_ratio):
    print("Training...")
    pad_tok = dataloader.dataset.input_vocab['<pad>']
    sos_tok = dataloader.dataset.input_vocab['<sos>']
    eos_tok = dataloader.dataset.input_vocab['<eos>']
    c_pad_tok = dataloader.dataset.output_vocab['<pad>']
    c_sos_tok = dataloader.dataset.output_vocab['<sos>']
    c_eos_tok = dataloader.dataset.output_vocab['<eos>']

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    best_loss = None
    for e in tqdm.tqdm(range(epoch)):
        total_loss = 0
        losses = []
        for batch_idx, (source, target) in enumerate(dataloader):
            # source, target = (B, L) => (L, B)
            # for i in range(len(source)):
                # source[i] = source[i].permute(1, 0).to(device)
            source = source.permute(1, 0).to(device)
            target = target.permute(1, 0).to(device)

            optimizer.zero_grad()
            outputs = model(source, target, teacher_forcing_ratio) # (Lo, B, Do)
            outputs = outputs[1:].view(-1, outputs.shape[-1]) # ((Lo-1) * B, Do)
            target = target[1:].reshape(-1) # ((Lo-1) * B)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            losses.append(loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Average loss {avg_loss}")
        wandb.log({"epoch": e, "loss": avg_loss})

        if best_loss is None or avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join("saved_models", f"model_{e}_{learning_rate}_{teacher_forcing_ratio}.pt"))
    
    torch.save(model.state_dict(), f"model_{epoch}_{learning_rate}_{teacher_forcing_ratio}.pt")

def test(model, dataloader):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    targets = []
    with torch.no_grad():
        generations = []
        for idx, (source, target) in enumerate(dataloader):
            # source, target = (B, L) => (L, B)
            # for i in range(len(source)):
            #     source[i] = source[i].permute(1, 0)
            source = source.permute(1, 0).to(device)
            target = target.permute(1, 0).to(device)

            outputs = model(source, target, 0) # (Lo, B, Do)

            loss = criterion(outputs.view(-1, outputs.shape[-1]), target.reshape(-1))
            total_loss += loss.item()
            
            generated_ids = outputs.argmax(2) # (Lo, B) let B=1
            generated_ids = generated_ids.squeeze(1) # (Lo,)
            
            generation = []
            for index in generated_ids:
                generation.append(dataloader.dataset.output_vocab_rev[index])
            generations.append(generation)

            targets.append(target.squeeze(1))
        avg_loss = total_loss / len(dataloader)
        wandb.log({"test_loss": avg_loss})
    print(generations)
    with open("output.txt", 'w') as file:
        gens = [" ".join(gen) for gen in generations]
        file.writelines(gens)
    bleu_score = evaluate(generations, targets)
    return generations, bleu_score
    
if __name__=="__main__":
    parser = argparse.ArgumentParser("Train script for Code Comment Generation")
    parser.add_argument("--epoch", type=int, default=200, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    parser.add_argument("--model", type=str, default="base", choices=["base", "AST", "Doc", "Full"], help="Model type to use (base, AST, Doc, Full)")
    parser.add_argument("--loadpath")
    parser.add_argument("--datapoints", type=int, default=1000)
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
    code_encoder = Encoder(CODE_DIM, CODE_ENC_HIDDEN_DIM, CODE_ENC_EMBED_DIM, dropout=CODE_ENC_DROPOUT).to(device)
    code_attention = Attention(CODE_ENC_HIDDEN_DIM, DEC_HIDDEN_DIM).to(device)
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
    decoder = Decoder(OUTPUT_DIM, DEC_HIDDEN_DIM, DEC_EMBED_DIM, context_dim * 2, DEC_NUM_LAYERS).to(device)

    # MultiSeq2Seq Model
    model = Seq2Seq(code_encoder, decoder, code_attention, device).to(device)
    # model = MultiSeq2Seq(encoders, decoder, attentions, device).to(device)

    if args.loadpath:
        model.load_state_dict(torch.load(args.loadpath))
    else: 
        model.apply(init_weights)

    print(model)

    # Train
    if not args.loadpath:
        print("Loading training data...")
        df_train = pd.read_pickle('./data/df_train_reduced.pkl').head(args.datapoints)
        train_loader = get_data_loader(df_train, test=False)
        print("Train data loaded")

        train(model, train_loader, args.epoch, args.learning_rate, args.teacher_forcing_ratio)

    # Test
    print("Loading test data...")
    df_test = pd.read_pickle('./data/df_test_reduced.pkl')
    test_loader = get_data_loader(df_test, test=True) 
    print("Test data loaded")
    
    generations, score = test(model, test_loader)
    print(f"BLEU-4 Score: {score}")
    wandb.log({"bleu_score": score})
    wandb.log({"generations": generations})

    with open("output.txt", 'w') as file:
        gens = [" ".join(gen) for gen in generations]
        file.writelines(gens)

