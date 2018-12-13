from nltk import word_tokenize
import csv
from torch.utils.data import Dataset
from extractors.extractor import Extractor
from utils.label import Label
import pandas as pd

class KaggleTestDatasetModified(Dataset):
    """
    Kaggel toxic comment classification tes t dataset.
    """
    def __init__(self, file_path: str, extractor: Extractor):

        self.original_df = pd.read_csv(file_path)

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