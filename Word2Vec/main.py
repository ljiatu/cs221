import zipfile

# Download GloVe representations
zipFile = zipfile.ZipFile('glove.6B.zip')
zipFile.extractall()

zipFile = zipfile.ZipFile('glove.840B.300d.zip')
zipFile.extractall()

from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from nltk.tokenize import word_tokenize

X, y = [], [[], [], [], [], [], []]

# String Constants
TRAINING_FILENAME = "train.csv"

# Next, I'll import the data and look quickly at some of the attributes:

training_data_frame = pd.read_csv(TRAINING_FILENAME)

# I'll drop the id column for now and then split the comment_text column
# and toxicity classes.

class_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

training_data_frame.drop('id', axis=1, inplace=True)
train_data = training_data_frame['comment_text']
train_target = training_data_frame.iloc[:, 1:]

documents = []

for index, entry in enumerate(train_data[:100]):
    X.append(word_tokenize(entry))
    labels = []
    for i in range(len(y)):
        if train_target.values[index, i] == 1:
            labels.append(class_labels[i])
        y[i].append(train_target.values[index, i])
    documents.append(TaggedDocument(word_tokenize(entry), labels))
X, y = np.array(X), np.array(y)


# Prepare word embeddings - both the pre-trained ones and train new ones from scratch.
# GloVe file is in format: word [embedding]

all_words = set(w for words in X for w in words)


class MeanParagraphEmbeddingVectorizer(object):
    def __init__(self, w2v):
        self.word2vec = w2v
        if len(w2v) > 0:
            self.dim = 100
        else:
            self.dim = 0

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0) for words in X
        ])


document_model = Doc2Vec(documents)

word_model = Word2Vec(X, size=100, window=5, min_count=5, workers=2)
w2v = {w: vec for w, vec in zip(word_model.wv.index2word, word_model.wv.syn0)}

print(w2v)

svc_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)),
                      ("linear_svc", SVC(kernel="linear", probability=True))])

etree_w2v = Pipeline([("word2vec vectorizer", MeanParagraphEmbeddingVectorizer(w2v)),
                      ("extra trees", ExtraTreesClassifier(n_estimators=200))])

svc_w2v = Pipeline([("word2vec vectorizer", MeanParagraphEmbeddingVectorizer(w2v)),
                    ("linear_svc", SVC(kernel="linear", probability=True))])


all_models = [("svc_tfidf", svc_tfidf),
              ("w2v_etree", etree_w2v),
              ("w2v_svc", svc_w2v)]


def benchmark(model, X, y, n):
    test_size = 1 - (n / float(len(y)))
    scores = []
    for train, test in StratifiedShuffleSplit(n_splits=5, test_size=0.2).split(X, y):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        scores.append(model.fit(X_train, y_train).predict_proba(X_test))
    return np.mean(scores, axis=0)


# Need to figure out how to translate these probabilities from binary (i.e. probability its in the class vs. not)
# into a probability that it belongs to each class

# dont care so much about our accuracy or whatever, just need to now
# (1) get the classifier to work for all classes
# (2) print the predicted probability per sample per class

scores = [(name, benchmark(model, X, y, n=10)) for name, model in all_models]

for i in range(len(y)):
    print("Scores for label %s" % class_labels[i])
    unsorted_scores = [(name, benchmark(model, X, y[i], n=1000)) for name, model in all_models]
    print(unsorted_scores)
    scores = sorted(unsorted_scores, key=lambda x: -x[1])
    print(tabulate(scores, floatfmt=".4f", headers=("model", "score")))

# I think the next thing to do is to make a different version of the MeanEmbeddingVectorizer that also adds
# in the document vector.