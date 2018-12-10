import torch
from torch import Tensor
from torch.nn import Conv2d, Linear, MaxPool2d, Module

from extractors.word2vec_2d_extractor import NUM_WORDS

POOL_OUTPUT_DIM = 300
HIDDEN_LAYER_DIMENSION = 100

FILTER_SIZES = [1, 2, 5]


class KimCNN(Module):
    """
    CNN that models after Y Kim's CNN for text classification.
    """
    def __init__(self, d_in: int, d_out: int):
        super(KimCNN, self).__init__()

        # 6 filters in total, 2 for each of 1, 2 and 5 words.
        self.filter_1word_1 = Conv2d(1, 1, (FILTER_SIZES[0], d_in))
        self.filter_2words_1 = Conv2d(1, 1, (FILTER_SIZES[1], d_in))
        self.filter_5words_1 = Conv2d(1, 1, (FILTER_SIZES[2], d_in))

        self.filter_1word_2 = Conv2d(1, 1, (FILTER_SIZES[0], d_in))
        self.filter_2words_2 = Conv2d(1, 1, (FILTER_SIZES[1], d_in))
        self.filter_5words_2 = Conv2d(1, 1, (FILTER_SIZES[2], d_in))

        # 3 pooling layers. 1 for each filter above.
        # The result for each max pool is a 1 x 1 vector.
        self.pool_1word = MaxPool2d((NUM_WORDS - FILTER_SIZES[0] + 1, 1))
        self.pool_2words = MaxPool2d((NUM_WORDS - FILTER_SIZES[1] + 1, 1))
        self.pool_5words = MaxPool2d((NUM_WORDS - FILTER_SIZES[2] + 1, 1))

        self.fc_layer = Linear(6, HIDDEN_LAYER_DIMENSION)
        self.output_layer = Linear(HIDDEN_LAYER_DIMENSION, d_out)

    def forward(self, x: Tensor) -> Tensor:
        # First, apply each filter on the input.
        filter_1word_1_out = self.filter_1word_1(x)
        filter_2words_1_out = self.filter_2words_1(x)
        filter_5words_1_out = self.filter_5words_1(x)

        filter_1word_2_out = self.filter_1word_2(x)
        filter_2words_2_out = self.filter_2words_2(x)
        filter_5words_2_out = self.filter_5words_2(x)

        # Then apply pooling on each filter output.
        pool_1word_1_out = self.pool_1word(filter_1word_1_out)
        pool_2words_1_out = self.pool_2words(filter_2words_1_out)
        pool_5words_1_out = self.pool_5words(filter_5words_1_out)

        pool_1word_2_out = self.pool_1word(filter_1word_2_out)
        pool_2words_2_out = self.pool_2words(filter_2words_2_out)
        pool_5words_2_out = self.pool_5words(filter_5words_2_out)

        # Concatenate the three tensors together.
        concat_output = torch.cat(
            (
                pool_1word_1_out,
                pool_2words_1_out,
                pool_5words_1_out,
                pool_1word_2_out,
                pool_2words_2_out,
                pool_5words_2_out,
            ),
            3,
        )
        fc_out = self.fc_layer(concat_output).clamp(min=0)
        return self.output_layer(fc_out).squeeze()
