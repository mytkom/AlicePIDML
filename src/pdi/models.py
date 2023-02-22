""" This module contains the models used in the experiments.

Classes
------
NeuralNet
    A basic neural network with optional dropout.
NeuralNetEnsemble
    An ensemble of networks used for processing incomplete examples.
AttentionModel
    Attention-based model used for processing incomplete examples.
"""
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import one_hot

from pdi.data.constants import N_COLUMNS
from pdi.data.types import GroupID


class NeuralNet(nn.Module):
    """NeuralNet is a basic neural network with variable layer dimensions, activation function and optional dropout.
    """

    def __init__(self,
                 layers: list[int],
                 activation: nn.Module,
                 dropout: Optional[float] = None):
        """__init__

        Args:
            layers (list[int]): list of layer dimensions, including the input layer, hidden layers and output layer.
            activation (nn.Module): activation function
            dropout (Optional[float], optional): dropout rate. Defaults to None.
        """
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


class NeuralNetEnsemble(nn.Module):
    """NeuralNetEnsemble is an ensemble of networks used for processing incomplete examples.
    Each group has a separate neural network in the ensemble, responsible for processing examples belonging to that group.
    """

    def __init__(
        self,
        group_ids: list[GroupID],
        hidden_layers: list[int],
        activation: nn.Module = nn.ReLU,
        dropout: float = 0.4,
    ):
        """__init__

        Args:
            group_ids (list[GroupID]): list of ids of groups in the dataset.
                Group ids should be binary numbers based on the missing values, e.g.
                an example with 5 values, missing the 1st value, should be given id 0b11110.

            hidden_layers (list[int]): list of hidden + output layer dimensions.
                All neural networks use the same dimensions.

            activation (nn.Module, optional): Activation function. Defaults to nn.ReLU.
            dropout (float, optional): dropout rate. Defaults to 0.4.
        """
        super(NeuralNetEnsemble, self).__init__()
        self.models = nn.ModuleDict({
            str(g_id): NeuralNet([bin(g_id).count("1")] + hidden_layers,
                                 activation, dropout)
            for g_id in group_ids
        })

    def forward(self, x: Tensor, group: Tensor):
        if not torch.all(group == group[0]):
            raise Exception(
                "Not all tensors in batch belong to the same group.")
        return self.models[str(group[0].numpy()[0])](x)


class AttentionModel(nn.Module):
    """AttentionModel is an attention-based model used for processing incomplete examples.
    """

    class _AttentionPooling(nn.Module):

        def __init__(self, in_dim, pool_dim, activation):
            super(AttentionModel._AttentionPooling, self).__init__()
            self.net = NeuralNet([in_dim, pool_dim, in_dim], activation)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            alpha = self.net(x)

            attention_weights = self.softmax(alpha)
            out = torch.mul(attention_weights, x)
            out = torch.sum(out, dim=-2)
            return out

    def __init__(
        self,
        in_dim: int,
        embed_hidden: int,
        embed_dim: int,
        ff_hidden: int,
        pool_hidden: int,
        num_heads: int,
        num_blocks: int,
        activation: nn.Module = nn.ReLU,
        dropout: float = 0.4,
    ):
        """__init__

        Args:
            in_dim (int): input dimension.
            embed_hidden (int): hidden dimension in the embedding layer.
            embed_dim (int): output dimension of the embedding layer.
            ff_hidden (int): hidden dimension in the feed forward layer in the Transformer.
            pool_hidden (int): pooling hidden dimension.
            num_heads (int): number of heads in the Transformer.
            num_blocks (int): number of blocks in the Transformer.
            activation (nn.Module, optional): activation function. Defaults to nn.ReLU.
            dropout (float, optional): dropout rate. Defaults to 0.4.
        """
        super(AttentionModel, self).__init__()
        self.emb = NeuralNet([in_dim, embed_hidden, embed_dim], activation)
        self.drop = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim,
                                                   num_heads,
                                                   ff_hidden,
                                                   dropout,
                                                   activation(),
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_blocks,
                                             enable_nested_tensor=False)
        self.pool = AttentionModel._AttentionPooling(embed_dim, pool_hidden,
                                                     activation)
        self.net = NeuralNet([embed_dim, pool_hidden, 1], activation)

    def forward(self, x: Tensor, group_ids: Optional[Tensor] = None) -> Tensor:
        x = self.emb(x)
        x = self.drop(x)
        x = self.encoder(x)
        x = self.pool(x)
        x = self.net(x)
        return x

    def predict(self, incomplete_tensor: Tensor) -> Tensor:
        missing = torch.isnan(incomplete_tensor)
        non_missing_idx = torch.argwhere(~missing)

        features = one_hot(non_missing_idx, N_COLUMNS).squeeze()
        values = incomplete_tensor[non_missing_idx]

        input = torch.cat((features, values), dim=-1)
        return self.forward(input)