import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse
import os
import wandb
import tqdm
from datetime import datetime

from lstm import *
from evaluation import *
from dataset import *

# Code Encoder
CODE_DIM = 2000
CODE_ENC_EMBED_DIM = 128
CODE_ENC_HIDDEN_DIM = 1024
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
OUTPUT_DIM = 2000
DEC_EMBED_DIM = 128
DEC_HIDDEN_DIM = 1024
DEC_NUM_LAYERS = 1
DEC_DROPOUT = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.1)

def train(model, dataloader, epoch, learning_rate, teacher_forcing_ratio, resume=0):
    print("Training...")

    model.train()
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    time_id = datetime.now().strftime("%Y_%m_%d_%H_%M")
    
    best_loss = None
    for e in tqdm.tqdm(range(resume, epoch)):
        total_loss = 0
        losses = []
        for batch_idx, (source, target) in enumerate(dataloader):
            # source, target = (B, L) => (L, B)
            batch_size = target.shape[0]
            # for i in range(len(source)):
                # source[i] = source[i].permute(1, 0).to(device)
            source = source.permute(1, 0).to(device)
            target = target.permute(1, 0).to(device)

            optimizer.zero_grad()
            outputs = model(source, target, teacher_forcing_ratio) # (Lo, B, Do)
            outputs_flat = outputs[1:,:,:].view(-1, outputs.shape[-1]) # (Lo * B, Do)
            target_flat = target[1:,:].reshape(-1) # (Lo * B)
            loss = criterion(outputs_flat, target_flat)
            # zero out losses for padding
            loss = loss.masked_fill(target_flat == torch.zeros_like(target_flat), 0)
            loss = torch.mean(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            losses.append(loss.item())

            if e > 0 and e % (epoch // 10) == 0 and batch_idx == 0:
                pred = outputs[1:].permute(1, 0, 2).argmax(2)
                trg = target[1:].permute(1, 0)
                gens = []
                targets = []
                for i in range(pred.shape[0]):
                    indices = pred[i]
                    p_words = [dataloader.dataset.output_vocab_rev[p.item()] for p in pred[i]]
                    gens.append(p_words)
                    t_words = [dataloader.dataset.output_vocab_rev[p.item()] for p in trg[i]]
                    targets.append(t_words)
                    if i < 5:
                        print("Prediction")
                        print(indices)
                        print(p_words)
                        print("Target")
                        print(trg[i])
                        print(t_words)

                score1 = evaluate(gens, targets, 1)
                score2 = evaluate(gens, targets, 2)
                score3 = evaluate(gens, targets, 3)
                score4 = evaluate(gens, targets, 4)
                print("blue1:", score1)
                print("blue2:", score2)
                print("blue3:", score3)
                print("blue4:", score4)

        avg_loss = total_loss / len(dataloader)
        print(f"Average loss {avg_loss}")
        wandb.log({"epoch": e, "loss": avg_loss})

        if best_loss is None or (avg_loss < best_loss and e > (epoch // 2)):
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join("saved_models", f"{time_id}_{learning_rate}_{teacher_forcing_ratio}.pt"))
    
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

            outputs_flat = outputs.view(-1, outputs.shape[-1]) # (Lo * B, Do)
            target_flat = torch.concat((target[1:], torch.zeros((1, 1), dtype=target.dtype, device=device)), dim=0).reshape(-1) # (Lo * B)
            loss = criterion(outputs_flat, target_flat)
            loss = loss.masked_fill(target_flat == torch.zeros_like(target_flat), 0)
            loss = torch.mean(loss)
            total_loss += loss.item()
            
            pred = outputs[1:].permute(1, 0, 2).argmax(2)
            trg = target[1:].permute(1, 0)
            words = [dataloader.dataset.output_vocab_rev[p.item()] for p in pred[0]]
            generations.append(words)
            # print("Prediction")
            # print(words)
            words = [dataloader.dataset.output_vocab_rev[p.item()] for p in trg[0]]
            # print("Target")
            # print(words) 
            targets.append(words)

        avg_loss = total_loss / len(dataloader)
        wandb.log({"test_loss": avg_loss})

    bleu_score_1 = evaluate(generations, targets, 1)
    bleu_score_2 = evaluate(generations, targets, 2)
    bleu_score_3 = evaluate(generations, targets, 3)
    bleu_score_4 = evaluate(generations, targets, 4)
    return generations, bleu_score_1, bleu_score_2, bleu_score_3, bleu_score_4
    
if __name__=="__main__":
    parser = argparse.ArgumentParser("Train script for Code Comment Generation")
    parser.add_argument("--epoch", type=int, default=200, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    parser.add_argument("--model", type=str, default="base", choices=["base", "AST", "Doc", "Full"], help="Model type to use (base, AST, Doc, Full)")
    parser.add_argument("--loadpath")
    parser.add_argument("--resume", type=int, default=0)
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
    print("Loading training data...")
    df_train = pd.read_pickle('./data/df_test_reduced.pkl').head(args.datapoints)
    train_loader = get_data_loader(df_train, test=False)
    print("Train data loaded")

    train(model, train_loader, args.epoch, args.learning_rate, args.teacher_forcing_ratio, args.resume)

    # Test
    print("Loading test data...")
    df_test = pd.read_pickle('./data/df_test_reduced.pkl').head(50)
    test_loader = get_data_loader(df_test, test=True) 
    print("Test data loaded")
    
    generations, score1, score2, score3, score4 = test(model, test_loader)
    print(f"BLEU-1 Score: {score1}")
    print(f"BLEU-2 Score: {score2}")
    print(f"BLEU-3 Score: {score3}")
    print(f"BLEU-4 Score: {score4}")
    wandb.log({"bleu_score1": score1, "bleu_score2": score2, "bleu_score3": score3, "bleu_score4": score4})

    with open("output.txt", 'w') as file:
        gens = [" ".join(gen) for gen in generations]
        file.writelines(gens)

