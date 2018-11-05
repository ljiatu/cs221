from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors, Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np
import torch

GLOVE_DATA_SMALL_PATH = 'glove.6B.100d.txt'
WORD2VEC_OUTPUT_FILE = GLOVE_DATA_SMALL_PATH + '.model'


class Word2VecTrainer:

    def __init__(self):
        # Load pre-trained word embeddings
        glove2word2vec(GLOVE_DATA_SMALL_PATH, WORD2VEC_OUTPUT_FILE)
        self.dim = 100
        self.vectors = KeyedVectors.load_word2vec_format(WORD2VEC_OUTPUT_FILE, binary=False)

    def extract_word_vector(self, text):
        words = word_tokenize(text)
        return torch.FloatTensor(np.mean([self.vectors[w] if w in self.vectors else np.zeros(self.dim)
                                          for w in words], axis=0))
