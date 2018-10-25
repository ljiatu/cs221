import csv

from torch.utils.data import Dataset

from utils.label import Label
from utils.word_extractor import extract


class KaggleDataset(Dataset):
    """
    Kaggle toxic comment classification dataset.
    """

    def __init__(self, file_path: str):
        """
        :param file_path: Data file path
        """
        with open(file_path) as data_file:
            reader = csv.reader(data_file)
            # Skip the header row.
            next(reader)
            self.data = [(row[0], extract(row[1]), Label(*row[2:]).tensor()) for row in reader if len(row) == 8]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
