import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformer import Transformer

class TrainData(Dataset):
    
    def __init__(self, ast_tokens, doc_tokens):
        def parse_list(tokens):
            new_tokens = []
            for sample in tokens:
                sample_list = sample.tolist()
                idxs = {s : i for i, s in enumerate(set(sample_list))}
                new_tokens.append([idxs[s] for s in sample_list])
            return torch.tensor(new_tokens)
        
        self.ast = parse_list(ast_tokens)
        self.doc = parse_list(doc_tokens)
    
    def __len__(self):
        return len(self.ast)
    
    def __getitem__(self, idx):
        return self.ast[idx], self.doc[idx]
    


def train_epoch(model, dataloader, optimizer, lossfn, embed_size):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        optimizer.zero_grad()
        output = model(src, tgt_input)
        loss = lossfn(output.view(-1, embed_size), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(loss.item())
    return total_loss / len(dataloader)


if __name__=="__main__":
    print('HERE')
    train_df = pd.read_pickle('df_val_reduced.pkl')
    print(train_df.head())
    ast_tokens = train_df['processed_ast_code_tokens'].tolist()
    doc_tokens = train_df['processed_doc_tokens'].tolist()

    embed_size = 10000
    model = Transformer(embed_size, 512, 8, 6, 6)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    dataset = TrainData(ast_tokens, doc_tokens)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(25):
        loss = train_epoch(model, dataloader, optimizer, loss, embed_size)
        print(f"Epoch {epoch+1}: Loss = {loss}")