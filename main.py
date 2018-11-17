from datetime import datetime

from torch import optim
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from datasets.kaggle_dataset import KaggleTrainingDataset, KaggleTestDataset
from models.Word2VecTrainer import Word2VecTrainer
from models.linear_model import LinearModel
from models.neural_net import NeuralNet
from utils.trainer import Trainer

BATCH_SIZE = 1
DATA_LOADER_NUM_WORKERS = 1
TRAIN_DATA_PATH = 'data/train.csv'
TEST_DATA_PATH = 'data/test.csv'
TEST_LABEL_PATH = 'data/test_labels.csv'
TEST_OUTPUT_PATH = f'outputs/{datetime.now().isoformat()}.txt'


def main():
    model = NeuralNet(100, 6)
    loss_func = BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    loader_train, loader_val, loader_test = _split_data()
    trainer = Trainer(
        model, loss_func, optimizer,
        loader_train, loader_val, loader_test,
        num_epochs=1, print_every=50000,
    )
    trainer.train()
    trainer.test(TEST_OUTPUT_PATH)


def _split_data():
    word2vec_trainer = Word2VecTrainer(train_data_path=TRAIN_DATA_PATH)
    train_dataset = KaggleTrainingDataset(TRAIN_DATA_PATH, word2vec_trainer.extract_word_vector)
    val_dataset = KaggleTrainingDataset(TRAIN_DATA_PATH, word2vec_trainer.extract_word_vector)
    test_dataset = KaggleTestDataset(TEST_DATA_PATH, TEST_LABEL_PATH, word2vec_trainer.extract_word_vector)
    # Reserve 10% of data for validation purposes.
    num_train = int(len(train_dataset) * 0.9)
    num_val = int(len(val_dataset) * 0.1)
    loader_train = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=DATA_LOADER_NUM_WORKERS,
        sampler=sampler.SubsetRandomSampler(range(num_train))
    )
    loader_val = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=DATA_LOADER_NUM_WORKERS,
        sampler=sampler.SubsetRandomSampler(range(num_train, num_train + num_val))
    )
    loader_test = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=DATA_LOADER_NUM_WORKERS,
    )

    return loader_train, loader_val, loader_test


if __name__ == '__main__':
    main()
