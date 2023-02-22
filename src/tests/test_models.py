import unittest
import torch
import torch.nn as nn

from pdi.models import NeuralNetEnsemble, AttentionModel, NeuralNet
from pdi.data.constants import N_COLUMNS


class TestNeuralNet(unittest.TestCase):

    def test_forward(self):
        (input, hidden, output, batch) = (10, 20, 1, 64)
        net = NeuralNet([input, hidden, output], nn.ReLU)
        input = torch.rand((batch, input))

        out = net(input)
        self.assertEqual(out.size(), (batch, output))


class TestNeuralNetEnsemble(unittest.TestCase):

    def test_forward(self):
        (hidden, output, batch) = (20, 1, 64)
        inputs = [0b11111, 0b11101, 0b10000]
        sizes = [5, 4, 1]

        ensemble = NeuralNetEnsemble(inputs, [hidden, output], nn.ReLU)
        for gid, size in zip(inputs, sizes):
            inputs = torch.rand((batch, size))
            group = torch.full((batch, 1), gid)

            out = ensemble(inputs, group)
            self.assertEqual(out.size(), (batch, output))


class TestAttentionModel(unittest.TestCase):

    def test_forward(self):
        batch, input, output = (64, 10, 1)
        model = AttentionModel(input, 32, 16, 32, 32, 4, 2, nn.ReLU)
        num_features = [1, 4, 7]
        for f in num_features:
            inputs = torch.rand((batch, f, input))
            out = model(inputs)
            self.assertEqual(out.size(), (batch, output))

    def test_permutation(self):
        batch, input, output = (64, 10, 1)
        model = AttentionModel(input, 32, 16, 32, 32, 4, 2, nn.ReLU, dropout=0)
        num_features = [4, 7]
        for f in num_features:
            inputs1 = torch.rand((batch, f, input))
            out1 = model(inputs1)

            perm = torch.randperm(f)
            while perm == list(range(f)):
                perm = torch.randperm(f)

            inputs2 = inputs1[:, perm, :]

            self.assertFalse(inputs1.equal(inputs2))

            out2 = model(inputs2)
            torch.testing.assert_allclose(out1, out2)

    def test_predict(self):
        model = AttentionModel(N_COLUMNS + 1,
                               32,
                               16,
                               32,
                               32,
                               4,
                               2,
                               nn.ReLU,
                               dropout=0)

        missing = [2, 5, 7]
        not_missing = [c for c in range(N_COLUMNS) if c not in missing]

        predict_input = torch.rand(N_COLUMNS)
        predict_input[missing] = torch.nan

        I = torch.eye(N_COLUMNS)

        forward_input = I[not_missing]
        forward_input = torch.cat(
            (forward_input, predict_input[not_missing].unsqueeze(-1)), dim=1)

        out1 = model.predict(predict_input)
        out2 = model(forward_input)

        torch.testing.assert_allclose(out1, out2)