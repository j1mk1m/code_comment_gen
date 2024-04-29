from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

EPSILON = 0.1
ALPHA = 5
K = 5

def evaluate(hypothesis, reference, method):
    sf = SmoothingFunction(EPSILON, ALPHA, K)
    if method == 1:
        func = sf.method1
    elif method == 2:
        func = sf.method2
    elif method == 3:
        func = sf.method3
    elif method == 4:
        func = sf.method4
    score = sentence_bleu([reference], hypothesis, smoothing_function=func)
    return score

if __name__=="__main__":
    reference = 'it is a dog'.split()

    candidate = 'it is a dog'.split()
    print('BLEU score -> {}'.format(evaluate(candidate, reference, 2)))
