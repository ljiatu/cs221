from datasets.kaggle_dataset_modified import KaggleTestDatasetModified
from extractors.word2vec_2d_extractor import Word2Vec2DExtractor
from torch.utils.data import DataLoader
from extractors.window_accuracy import WindowAccuracy

TRAIN_DATA_PATH = 'data/train.csv'
TEST_DATA_PATH = 'data/processed.csv'
TEST_LABEL_PATH = 'data/test_labels.csv'

BATCH_SIZE = 25
DATA_LOADER_NUM_WORKERS = 2

def run_window_accuracy():
    test_data = _test_data()
    path = "cnn_2018-12-12T21:32:47.472484.model"
    window_accuracy = WindowAccuracy(model_path=path, loader=test_data)
    window_accuracy.check_window_accuracy('outputs/test_window_two.txt')


def _test_data():
    word2vec_extractor = Word2Vec2DExtractor(train_data_path=TRAIN_DATA_PATH)
    test_dataset = KaggleTestDatasetModified(TEST_DATA_PATH, TEST_LABEL_PATH, word2vec_extractor)
    # Reserve 10% of data for validation purposes.
    loader_test = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=DATA_LOADER_NUM_WORKERS,
    )

    return loader_test


run_window_accuracy()
