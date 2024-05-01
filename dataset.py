import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from collections import Counter

# Dataloader
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 1

# 2346, 1166
def get_vocabs(data, cv_size, av_size, dv_size, ov_size):
    input_vocab, input_rev_vocab = create_vocab(data['processed_func_code_tokens'], cv_size)
    ast_vocab, ast_rev_vocab = create_vocab(data['processed_ast_code_tokens'], av_size)
    doc_vocab, doc_rev_vocab = create_vocab(data['processed_doc_tokens'], dv_size)
    output_vocab, output_rev_vocab = create_vocab(data['processed_doc_tokens'], ov_size)
    return input_vocab, input_rev_vocab, ast_vocab, ast_rev_vocab, doc_vocab, doc_rev_vocab, output_vocab, output_rev_vocab

def create_vocab(data, vocab_size):
    unknown_token = '<unk>'        
    all_words = [word for sentence in data for word in sentence]
    word_freq = Counter(all_words)
    
    most_common_words = word_freq.most_common(vocab_size-1)
    
    vocabulary = {word: idx for idx, (word, _) in enumerate(most_common_words)}

    vocabulary[unknown_token] = len(vocabulary)
    
    reverse_vocab = {id:word for word,id in vocabulary.items()}
    for i in range(len(reverse_vocab), vocab_size):
        reverse_vocab[i] = unknown_token

    return vocabulary, reverse_vocab


class CustomDataset(Dataset):
    def __init__(self, data, cvocab, avocab, dvocab, ovocab, mode):
        self.unknown_token = '<unk>'
        self.data = data
        self.input_encoded = self.parse(data["processed_func_code_tokens"], cvocab)
        self.ast_encoded = self.parse(data["processed_ast_code_tokens"], avocab)
        self.doc_encoded = self.parse(data["processed_doc_tokens"], dvocab)
        self.output_encoded = self.parse(data["processed_doc_tokens"], ovocab)
        self.mode = mode

    def parse(self, data, vocab):
        return [self.token_to_index(sentence, vocab) for sentence in data]

    def token_to_index(self, sentence, vocab):
        indices = [vocab.get(word, vocab[self.unknown_token]) for word in sentence]
        return torch.tensor(indices, dtype=torch.int64)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        inputs = []
        inputs.append(self.input_encoded[idx])
        if self.mode == 'Full' or self.mode == 'AST':
            inputs.append(self.ast_encoded[idx])
        if self.mode == 'Full' or self.mode == 'Doc':
            inputs.append(self.doc_encoded[idx])
        
        return inputs, self.output_encoded[idx]
    
def get_data_loader(series, cvocab, avocab, dvocab, ovocab, batch_size, mode):
    dataset = CustomDataset(series, cvocab, avocab, dvocab, ovocab, mode)
    return DataLoader(dataset, batch_size, shuffle=False)

if __name__=="__main__":
    df_train = pd.read_pickle('./data/df_train_reduced.pkl').head(1000)
    train_loader = get_data_loader(df_train, test=False)

    input_vocab = train_loader.dataset.input_vocab.items()
    input_vocab_rev = train_loader.dataset.input_vocab_rev.items()
    output_vocab = train_loader.dataset.output_vocab.items()
    output_vocab_rev = train_loader.dataset.output_vocab_rev.items()

    print(input_vocab_rev)
    print(output_vocab_rev)


