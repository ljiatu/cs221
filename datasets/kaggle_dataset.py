import pandas as pd
from torch.utils.data import Dataset

from extractors.extractor import Extractor
from utils.label import Label


class KaggleDataset(Dataset):
    """
    Kaggle toxic comment classification training dataset.
    """
    def __init__(self, file_path: str, extractor: Extractor):
        """
        :param file_path: Data file path
        """
        self.comments_df = pd.read_csv(file_path)
        self.extractor = extractor

    def __len__(self):
        return len(self.comments_df)

    def __getitem__(self, idx: int):
        text_id = self.comments_df.iloc[idx, 0]
        word_vec = self.extractor.extract(self.comments_df.iloc[idx, 1])
        labels = Label(*self.comments_df.iloc[idx, 2:]).tensor()
        return text_id, word_vec, labels
