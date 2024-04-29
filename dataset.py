import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from collections import Counter

# Dataloader
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 1

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
        return torch.tensor(indices, dtype=torch.int64)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.input_encoded[idx], self.output_encoded[idx]
    
def get_data_loader(series, test = False):
    dataset = CustomDataset(series)
    return DataLoader(dataset, batch_size=TEST_BATCH_SIZE if test == True else TRAIN_BATCH_SIZE, shuffle=True)

if __name__=="__main__":
    df_train = pd.read_pickle('./data/df_train_reduced.pkl').head(1000)
    train_loader = get_data_loader(df_train, test=False)

    input_vocab = train_loader.dataset.input_vocab.items()
    input_vocab_rev = train_loader.dataset.input_vocab_rev.items()
    output_vocab = train_loader.dataset.output_vocab.items()
    output_vocab_rev = train_loader.dataset.output_vocab_rev.items()

    print(input_vocab_rev)
    print(output_vocab_rev)


