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

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

from evaluation import *
from dataset import *

    
if __name__=="__main__":
    pipe = pipeline("text-generation", model="NeuralNotwork/gpt2-baseline")
    # Test
    print("Loading train data...")
    df_test = pd.read_pickle('./data/df_train.pkl').head(1000)
    test_loader = get_basic_loader(df_test) 
    print("Train data loaded")
    
    total = 0
    score1 = 0
    score2 = 0
    score3 = 0
    score4 = 0
    for idx, (source, target) in enumerate(test_loader):
        pred = pipe(source, max_length=210, truncation=True)[0]["generated_text"].split()
        score1 += evaluate(pred, target.split(), 1)
        score2 += evaluate(pred, target.split(), 2)
        score3 += evaluate(pred, target.split(), 3)
        score4 += evaluate(pred, target.split(), 4)
    print(f"BLEU-1 Score: {score1 / len(test_loader)}")
    print(f"BLEU-2 Score: {score2 / len(test_loader)}")
    print(f"BLEU-3 Score: {score3 / len(test_loader)}")
    print(f"BLEU-4 Score: {score4 / len(test_loader)}")

