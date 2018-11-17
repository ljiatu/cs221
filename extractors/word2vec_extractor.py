import csv
import os

import numpy as np
import torch
from gensim.models import KeyedVectors, Word2Vec
from nltk.tokenize import word_tokenize

from extractors.extractor import Extractor

MODEL_FILENAME = 'word2vec_scratch.model'
DIM = 100


class Word2VecExtractor(Extractor):
    def __init__(self, train_data_path=None):
        if os.path.isfile(MODEL_FILENAME):
            self.vectors = KeyedVectors.load(MODEL_FILENAME)
        else:
            with open(train_data_path) as data_file:
                reader = csv.reader(data_file)
                # Skip the header row.
                next(reader)
                sentences = [word_tokenize(row[1]) for row in reader if len(row) == 8]
            word_model = Word2Vec(sentences=sentences, size=DIM, window=5, min_count=3, workers=2)
            word_model.save(MODEL_FILENAME)
            self.vectors = word_model.wv

    def extract(self, text: str):
        words = word_tokenize(text)
        return torch.FloatTensor(np.mean(
            [self.vectors[w] if w in self.vectors else np.zeros(DIM) for w in words], axis=0
        ))
