import csv
import os

import torch
from gensim.models import KeyedVectors, Word2Vec
from nltk.tokenize import word_tokenize

from extractors.extractor import Extractor

MODEL_FILENAME = 'word2vec_scratch.model'
DIM = 100
NUM_WORDS = 100


class Word2Vec2DExtractor(Extractor):
    """
    Extracts a 2D matrix out of each sentence. Each row in the sentence is one word.
    """
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

    def extract(self, text: str) -> torch.FloatTensor:
        words = word_tokenize(text)
        word_tensors = []
        for i in range(min(len(words), NUM_WORDS)):
            if words[i] in self.vectors:
                word_tensors.append(torch.FloatTensor(self.vectors[words[i]]))
            else:
                word_tensors.append(torch.zeros(DIM, dtype=torch.float))
        if len(word_tensors) < NUM_WORDS:
            zero_paddings = [torch.zeros(DIM, dtype=torch.float)] * (NUM_WORDS - len(word_tensors))
            word_tensors.extend(zero_paddings)

        stacked = torch.stack(word_tensors)
        return stacked.unsqueeze(0)
