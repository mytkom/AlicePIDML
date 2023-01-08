from typing import Optional

import torch
import torch.nn as nn
from pdi.models.pooling import AttentionPooling
from pdi.models.utils import NeuralNet
from torch import Tensor
from torch.nn.functional import one_hot
from pdi.data.constants import N_COLUMNS


class NeuralNetEnsemble(nn.Module):

    def __init__(
        self,
        group_ids: list[int],
        hidden_layers: list[int],
        activation=nn.ReLU,
        dropout: float = 0.4,
    ):
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

    def __init__(
        self,
        in_dim: int,
        embed_hidden: int,
        embed_dim: int,
        ff_hidden: int,
        pool_hidden: int,
        num_heads: int,
        num_blocks: int,
        activation=nn.ReLU,
        dropout: float = 0.4,
    ):
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
        self.pool = AttentionPooling(embed_dim, pool_hidden, activation)
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


# if __name__ == "__main__":
#     device = torch.device("cpu")

#     model = torch.load(f"models/Proposed/pion.pt").to(device)
#     dummy_input = torch.rand(N_COLUMNS)
#     dummy_input[2] = torch.nan
#     dummy_input[5] = torch.nan

#     model.predict(dummy_input)
