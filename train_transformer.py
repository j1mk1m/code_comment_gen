import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformer import Transformer
from dataset import *
from tqdm import tqdm

class TrainData(Dataset):
    
    def __init__(self, ast_tokens, doc_tokens, idxs = None):
        if idxs is None:
            self.idxs = {}
        else:
            self.idxs = idxs
        
        self.doc = doc_tokens
        def parse_list(tokens):
            new_tokens = []
            idx_len = len(self.idxs)
            for sample in tokens:
                sample_list = sample.tolist()
                for s in sample_list:
                    if s not in self.idxs:
                        idx_len += 1
                        self.idxs[s] = idx_len
                    
                new_tokens.append([self.idxs[s] for s in sample_list])
            return torch.tensor(new_tokens, dtype=torch.int32)
        
        self.ast = parse_list(ast_tokens)

    def get_idxs(self):
        return self.idxs
    
    def __len__(self):
        return len(self.ast)
    
    def __getitem__(self, idx):
        return self.ast[idx], self.doc[idx]
    
    def get_idxs(self):
        return self.idxs
    

def train_epoch(model, dataloader, optimizer, lossfn, embed_size):
    model.train()
    total_loss = 0
    for src, tgt in tqdm(dataloader):
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        optimizer.zero_grad()
        output = model(src, tgt_input)
        tgt_new =  tgt_output.reshape(-1)
        loss = lossfn(output.view(-1, embed_size), tgt_new)
        loss = torch.mean(loss.masked_fill(tgt_new == torch.zeros_like(tgt_new), 0))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def test_model(model, dataloader, lossfn, embed_size, output_vocab, num_samples=0):
    model.eval()
    total_loss = 0
    total_count = 0

    with torch.no_grad():
        for idx, (src, tgt) in enumerate(dataloader):
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            output = model(src, tgt_input)
            loss = lossfn(output.view(-1, embed_size), tgt_output.reshape(-1))
            total_loss += loss.item() * tgt_output.size(0)  
            total_count += tgt_output.size(0)

            if idx < num_samples:
                predicted_ids = output.argmax(-1)
                print(f"Sample {idx+1}:")
                print("Source:", src)
                actual = []
                for aid in tgt_output[0]:
                    actual.append(output_vocab[int(aid)])
                print("Target Actual:", actual)
                predict = []
                for pid in predicted_ids:
                    predict.append(output_vocab[int(pid[0])])
                print("Target Predicted:", predict)

    average_loss = total_loss / total_count
    print(average_loss)
    return average_loss

if __name__=="__main__":
    print('TRAINING')
    train_df = pd.read_pickle('df_val_reduced.pkl')
    test_df = pd.read_pickle('df_test_reduced.pkl')
    print(train_df.head())

    train_ast_tokens = train_df['processed_func_code_tokens'].tolist()
    test_ast_tokens = test_df['processed_func_code_tokens'].tolist()

    train_custom_data = get_data_loader(train_df, test=False)
    test_custom_data = get_data_loader(test_df, test=True)

    train_dataset = TrainData(train_ast_tokens, train_custom_data.dataset.output_encoded)
    test_dataset = TrainData(test_ast_tokens, test_custom_data.dataset.output_encoded, train_dataset.idxs)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # embed_size = int(torch.max(train_dataloader))
    embed_size = len(test_dataset.idxs)
    model = Transformer(embed_size, 512, 8, 6, 6)
    lossfn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(10):
        loss = train_epoch(model, train_dataloader, optimizer, lossfn, embed_size)
        print(f"Epoch {epoch+1}: Loss = {loss}")
        test_model(model, test_dataloader, lossfn, embed_size, test_custom_data.dataset.output_vocab_rev, 10)

