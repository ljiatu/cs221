from torch import Tensor
from torch.nn import Linear, Module

HIDDEN_LAYER1_DIMENSION = 400
# HIDDEN_LAYER2_DIMENSION = 200
# HIDDEN_LAYER3_DIMENSION = 100


class NeuralNet(Module):
    def __init__(self, d_in: int, d_out: int):
        super(NeuralNet, self).__init__()

        self.hidden_layer1 = Linear(d_in, HIDDEN_LAYER1_DIMENSION)
        # self.hidden_layer2 = Linear(HIDDEN_LAYER1_DIMENSION, HIDDEN_LAYER2_DIMENSION)
        # self.hidden_layer3 = Linear(HIDDEN_LAYER2_DIMENSION, HIDDEN_LAYER3_DIMENSION)
        self.output_layer = Linear(HIDDEN_LAYER1_DIMENSION, d_out)

    def forward(self, x: Tensor) -> Tensor:
        h1_relu = self.hidden_layer1(x).clamp(min=0)
        # h2_relu = self.hidden_layer2(h1_relu).clamp(min=0)
        # h3_relu = self.hidden_layer3(h2_relu).clamp(min=0)
        return self.output_layer(h1_relu)
