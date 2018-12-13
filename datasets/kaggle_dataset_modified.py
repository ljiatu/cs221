import bisect

from nltk import word_tokenize
import pandas as pd
from torch.utils.data import Dataset

from extractors.extractor import Extractor
from utils.label import Label

WINDOW_SIZE = 2


class KaggleTestDatasetModified(Dataset):
    """
    Dataset for toxicity source identification.
    """
    def __init__(self, file_path: str, extractor: Extractor):
        self.comments_df = pd.read_csv(file_path)
        self.extractor = extractor
        self.max_ranges = self.compute_max_range()

    def find_index_from_query(self, query_idx: int):
        idx = bisect.bisect(self.max_ranges, query_idx)
        if idx == 0:
            offset = query_idx
        else:
            offset = query_idx - self.max_ranges[idx - 1]
        return idx, offset

    def find_window(self, idx, offset):
        full_text = word_tokenize(self.comments_df.iloc[idx, 1])
        comment = full_text[max(0, offset - WINDOW_SIZE):min(len(full_text) - 1, offset + WINDOW_SIZE)]
        return comment

    def compute_max_range(self):
        max_ranges = []
        prev_count = 0
        for index, row in self.comments_df.iterrows():
            word_count = len(word_tokenize(row['comment_text']))
            prev_count += word_count
            max_ranges.append(prev_count)

        return max_ranges

    def __len__(self):
        return self.max_ranges[-1]

    def __getitem__(self, idx: int):
        real_idx, offset = self.find_index_from_query(idx)
        text_id = self.comments_df.iloc[real_idx, 0]
        comment_window = self.find_window(real_idx, offset)
        word_vec = self.extractor.extract(comment_window)
        labels = Label(*self.comments_df.iloc[idx, 2:]).tensor()
        return text_id, word_vec, labels
