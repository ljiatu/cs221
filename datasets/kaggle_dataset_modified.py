from nltk import word_tokenize
import csv
from torch.utils.data import Dataset
from extractors.extractor import Extractor
from utils.label import Label


class KaggleTestDatasetModified(Dataset):
    """
    Kaggel toxic comment classification tes t dataset.
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

        data = []
        window_size = 2
        for k, v in self.labels.items():
            words = word_tokenize(self.text[k])
            windows = [words[max(0, i - window_size):min(len(words) - 1, i + window_size)] for i in range(len(words))]
            for window in windows:
                if len(window) > 0:
                    window_text = ' '.join(window)
                    data.append((k, extractor.extract(window_text), v))
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]