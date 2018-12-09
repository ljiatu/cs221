import torch
from torch import Tensor
from torch.nn import Conv2d, Linear, MaxPool2d, Module

from extractors.word2vec_2d_extractor import NUM_WORDS

POOL_OUTPUT_DIM = 300
HIDDEN_LAYER_DIMENSION = 100

FILTER_SIZES = [1, 2, 5]


class CNN(Module):
    def __init__(self, d_in: int, d_out: int):
        super(CNN, self).__init__()

        self.d_out = d_out

        # 3 filters in total, 1 for each of 1, 2 and 5 words.
        self.filter_1word = Conv2d(1, 1, (FILTER_SIZES[0], 1))
        self.filter_2words = Conv2d(1, 1, (FILTER_SIZES[1], 1))
        self.filter_5words = Conv2d(1, 1, (FILTER_SIZES[2], 1))

        # 3 pooling layers. 1 for each filter above.
        # The result for each max pool is a 1 x 100 vector.
        self.pool_1word = MaxPool2d((NUM_WORDS - FILTER_SIZES[0] + 1, 1))
        self.pool_2words = MaxPool2d((NUM_WORDS - FILTER_SIZES[1] + 1, 1))
        self.pool_5words = MaxPool2d((NUM_WORDS - FILTER_SIZES[2] + 1, 1))

        self.fc_layer = Linear(3 * d_in, HIDDEN_LAYER_DIMENSION)
        self.output_layer = Linear(HIDDEN_LAYER_DIMENSION, d_out)

    def forward(self, x: Tensor) -> Tensor:
        # First, apply each filter on the input.
        filter_1word_out = self.filter_1word(x)
        filter_2words_out = self.filter_2words(x)
        filter_5words_out = self.filter_5words(x)

        # Then apply pooling on each filter.
        pool_1word_out = self.pool_1word(filter_1word_out)
        pool_2words_out = self.pool_2words(filter_2words_out)
        pool_5words_out = self.pool_5words(filter_5words_out)

        # Concatenate the three tensors together.
        concat_output = torch.cat((pool_1word_out, pool_2words_out, pool_5words_out), 3)
        fc_out = self.fc_layer(concat_output).clamp(min=0)
        # Reshape the vector have the same dimension as the output.
        return self.output_layer(fc_out).reshape(1, self.d_out)
