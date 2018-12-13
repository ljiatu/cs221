from datetime import datetime

import torch
from torch import optim
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from datasets.kaggle_dataset import KaggleDataset
from extractors.word2vec_2d_extractor import Word2Vec2DExtractor
from extractors.word2vec_extractor import Word2VecExtractor
from models.cnn import CNN
from models.kim_cnn import KimCNN
from models.linear_model import LinearModel
from models.neural_net import NeuralNet
from utils.trainer import Trainer

BATCH_SIZE = 25
DATA_LOADER_NUM_WORKERS = 5
TRAIN_DATA_PATH = 'data/train.csv'
TEST_DATA_PATH = 'data/processed.csv'
TEST_OUTPUT_PATH = f'outputs/{datetime.now().isoformat()}.txt'


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device {device}')

    model = KimCNN(100, 6).to(device=device)
    loss_func = BCEWithLogitsLoss().to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    loader_train, loader_val, loader_test = _split_data()
    trainer = Trainer(
        model, loss_func, optimizer, device, 'cnn',
        loader_train, loader_val, loader_test,
        num_epochs=5, print_every=50000,
    )
    trainer.train()
    trainer.test(TEST_OUTPUT_PATH)


def _split_data():
    word2vec_extractor = Word2Vec2DExtractor(train_data_path=TRAIN_DATA_PATH)
    train_dataset = KaggleDataset(TRAIN_DATA_PATH, word2vec_extractor)
    val_dataset = KaggleDataset(TRAIN_DATA_PATH, word2vec_extractor)
    test_dataset = KaggleDataset(TEST_DATA_PATH, word2vec_extractor)
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
