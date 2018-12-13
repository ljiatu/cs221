import sys

import torch
from torch.utils.data import DataLoader

from datasets.kaggle_dataset_modified import KaggleTestDatasetModified
from extractors.word2vec_2d_extractor import Word2Vec2DExtractor
from extractors.window_accuracy import WindowAccuracy

TRAIN_DATA_PATH = 'data/train.csv'
TEST_DATA_PATH = 'data/processed.csv'

BATCH_SIZE = 25
DATA_LOADER_NUM_WORKERS = 5


def run_window_accuracy(model_path: str):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    test_data = _test_data()
    window_accuracy = WindowAccuracy(model_path=model_path, loader=test_data, device=device)
    window_accuracy.check_window_accuracy('outputs/test_window_two.txt')


def _test_data():
    word2vec_extractor = Word2Vec2DExtractor(train_data_path=TRAIN_DATA_PATH)
    test_dataset = KaggleTestDatasetModified(TEST_DATA_PATH, word2vec_extractor)
    loader_test = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=DATA_LOADER_NUM_WORKERS,
    )

    return loader_test


if __name__ == '__main__':
    saved_model_path = f'saved_models/{sys.argv[1]}'
    run_window_accuracy(saved_model_path)
