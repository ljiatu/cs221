from torch import Tensor
from torch.nn import Linear, Module, Sigmoid


class LinearModel(Module):
    def __init__(self, d_in: int, d_out: int):
        super(LinearModel, self).__init__()

        self.linear_layer = Linear(d_in, d_out)
        self.prob_func = Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_layer(x)
        # return self.prob_func(raw_scores)
