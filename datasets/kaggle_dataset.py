import csv

from torch.utils.data import Dataset

from extractors.extractor import Extractor
from utils.label import Label


class KaggleTrainingDataset(Dataset):
    """
    Kaggle toxic comment classification training dataset.
    """
    def __init__(self, file_path: str, extractor: Extractor):
        """
        :param file_path: Data file path
        """
        with open(file_path) as data_file:
            reader = csv.reader(data_file)
            # Skip the header row.
            next(reader)
            self.data = [(row[0], extractor.extract(row[1]), Label(*row[2:]).tensor()) for row in reader if len(row) == 8]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


class KaggleTestDataset(Dataset):
    """
    Kaggel toxic comment classification test dataset.
    """
    def __init__(self, text_file_path: str, label_file_path: str, extractor: Extractor):
        """
        :param text_file_path: Path to file containing comment text
        :param label_file_path: Path to file containing labels
        """
        with open(text_file_path) as text_file:
            reader = csv.reader(text_file)
            # Skip the header row.
            next(reader)
            self.text = {row[0]: row[1] for row in reader}

        self.labels = {}
        with open(label_file_path) as label_file:
            reader = csv.reader(label_file)
            # Skip the header row.
            next(reader)
            for row in reader:
                if row[1] != '-1':
                    self.labels[row[0]] = Label(*row[1:]).tensor()

        self.data = [(k, extractor.extract(self.text[k]), v) for k, v in self.labels.items()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
