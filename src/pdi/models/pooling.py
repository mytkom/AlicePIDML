import torch
import torch.linalg as linalg
import torch.nn as nn
from pdi.models.utils import NeuralNet


class MaxPooling(nn.Module):

    def __init__(self) -> None:
        super(MaxPooling, self).__init__()

    def forward(self, embeddings):
        values, _ = torch.max(embeddings, dim=1)  # , keepdim=True
        return values


def squash_function(X):
    norm = linalg.vector_norm(X, dim=1)
    norm_sq = norm * norm
    scale = norm_sq / ((1 + norm_sq) * norm)
    return X * scale.view((-1, 1))


class DynamicPooling(nn.Module):

    def __init__(self, T):
        super(DynamicPooling, self).__init__()
        self.T = T

    def forward(self, embeddings):
        B = torch.zeros(embeddings.shape[0:2], device=embeddings.device)
        softmax = nn.Softmax(dim=1)
        for t in range(self.T):
            C = softmax(B)
            Sigma = torch.bmm(embeddings.transpose(1, 2),
                              C.unsqueeze(2)).squeeze(2)
            values = squash_function(Sigma)
            B = B + torch.bmm(embeddings, values.unsqueeze(2)).squeeze(2)
        return values


class AttentionPooling(nn.Module):

    def __init__(self, in_dim, pool_dim, activation):
        super(AttentionPooling, self).__init__()
        self.net = NeuralNet([in_dim, pool_dim, in_dim], activation)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        alpha = self.net(x)

        attention_weights = self.softmax(alpha)
        out = torch.mul(attention_weights, x)
        out = torch.sum(out, dim=-2)
        return out
