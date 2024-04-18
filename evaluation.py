import nltk 
from nltk.translate import bleu
from nltk.translate.bleu_score import SmoothingFunction

EPSILON = 0.1
ALPHA = 5
K = 5

def evaluate(hypothesis, references):
    sf = SmoothingFunction(EPSILON, ALPHA, K)
    total_score = 0
    for i in range(len(hypothesis)):
        hyp, ref = hypothesis[i], references[i]
        score = bleu([hyp], [ref], smoothing_function=sf.method4)
        total_score += score

    return total_score / len(hypothesis)

if __name__=="__main__":
    hyp = ["hello, my name is James"]
    ref = ["this project is cool"]
    print(evaluate(hyp, ref))