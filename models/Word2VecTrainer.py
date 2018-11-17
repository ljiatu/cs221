from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors, Word2Vec
from nltk.tokenize import word_tokenize
import numpy as np
import torch
import csv

GLOVE_DATA_SMALL_PATH = 'glove.6B.100d.txt'
WORD2VEC_OUTPUT_FILE = GLOVE_DATA_SMALL_PATH + '.model'


class Word2VecTrainer:
    def __init__(self, train_data_path=None):
        self.dim = 100

        if train_data_path is None:
            # Load pre-trained word embeddings
            glove2word2vec(GLOVE_DATA_SMALL_PATH, WORD2VEC_OUTPUT_FILE)
            self.vectors = KeyedVectors.load_word2vec_format(WORD2VEC_OUTPUT_FILE, binary=False)
        else:
            with open(train_data_path) as data_file:
                reader = csv.reader(data_file)
                # Skip the header row.
                next(reader)
                sentences = [word_tokenize(row[1]) for row in reader if len(row) == 8]
            self.word_model = Word2Vec(sentences=sentences, size=100, window=5, min_count=3, workers=2)
            self.vectors = self.word_model.wv

    def extract_word_vector(self, text):
        words = word_tokenize(text)
        return torch.FloatTensor(np.mean([self.vectors[w] if w in self.vectors else np.zeros(self.dim)
                                          for w in words], axis=0))

