
import logging
import sys
targets = logging.StreamHandler(sys.stdout), logging.FileHandler('logs/word2vec.log')
logging.basicConfig(format='%(message)s', level=logging.INFO, handlers=targets)



import numpy as np
import random
import time
from word2vec import *
from sgd import *
from treebank import *

# Reset the random seed to make sure that everyone gets the same results
random.seed(42)
np.random.seed(42)

for label in ["PERSON","LOC","all_chapters"]:
    dataset = WebtoonSentiment(f'data/sentencebroken/{label}.csv')
    tokens = dataset.tokens()
    nWords = len(tokens)

    dimVectors = 10
    C = 5
  
    startTime = time.time()
    wordVectors = np.concatenate(
        ((np.random.rand(nWords, dimVectors) - 0.5) /
        dimVectors, np.zeros((nWords, dimVectors))),
        axis=0)
    wordVectors = sgd(
        lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C,
                                        negSamplingLossAndGradient),
        wordVectors, 0.3, 40000, None, True, PRINT_EVERY=10,label=label)

    logging.info("training took %d seconds" % (time.time() - startTime))
