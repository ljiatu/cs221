from torch import optim
from torch.nn import Linear, MultiLabelMarginLoss
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from datasets.kaggle_dataset import KaggleDataset
from utils.trainer import Trainer
from utils.word_extractor import BAD_WORDS

BATCH_SIZE = 1
DATA_LOADER_NUM_WORKERS = 1
TRAIN_DATA_PATH = 'data/train.csv'
TEST_DATA_PATH = 'data/test.csv'


def main():
    model = Linear(BATCH_SIZE, len(BAD_WORDS))
    loss_func = MultiLabelMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    loader_train, loader_val, loader_test = _split_data()
    trainer = Trainer(
        model, loss_func, optimizer,
        loader_train, loader_val, loader_test,
        num_epochs=1, print_every=50000,
    )
    trainer.train()
    # model_trainer.test()


def _split_data():
    train_dataset = KaggleDataset(TRAIN_DATA_PATH)
    val_dataset = KaggleDataset(TRAIN_DATA_PATH)
    test_dataset = KaggleDataset(TEST_DATA_PATH)
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
