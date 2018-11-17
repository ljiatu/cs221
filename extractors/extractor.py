from abc import ABC, abstractmethod

import torch


class Extractor(ABC):
    @abstractmethod
    def extract(self, text) -> torch.FloatTensor:
        pass
