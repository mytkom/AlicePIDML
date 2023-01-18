import unittest
import torch

from pdi.data.utils import DictDataset


class TestCombinedDataLoader(unittest.TestCase):

    def setUp(self) -> None:
        self.input = [10, 20, 30]
        self.target = torch.tensor([0, 1, -1])
        self.additional1 = ["a", "b", "c"]
        self.additional2 = [[0, 1], [2, 3], [4, 5]]

        self.dataset = DictDataset(
            self.input, self.target, **{
                "string": self.additional1,
                "list": self.additional2
            })

    def test_getitem(self):
        for i in range(3):
            self.assertEqual(self.dataset[i], (self.input[i], self.target[i], {
                "string": self.additional1[i],
                "list": self.additional2[i]
            }))

    def test_len(self):
        self.assertEqual(len(self.dataset), 3)