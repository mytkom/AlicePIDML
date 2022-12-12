from typing import Optional

import torch.nn as nn
from torch import Tensor


class NeuralNet(nn.Module):

    def __init__(self,
                 layers: list[int],
                 activation: nn.Module,
                 dropout: Optional[float] = None):
        super(NeuralNet, self).__init__()
        self.layers = nn.ModuleList()
        for (in_f, out_f) in zip(layers[:-2], layers[1:-1]):
            self.layers.append(nn.Linear(in_f, out_f))
            self.layers.append(activation())
            if dropout is not None:
                self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(layers[-2], layers[-1]))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
