import unittest

from pdi.data.utils import CombinedDataLoader


class TestCombinedDataLoader(unittest.TestCase):

    def setUp(self):
        self.d1 = [10, 20, 30]
        self.d2 = [-5, -15, -25]
        self.combined = CombinedDataLoader(True, self.d1, self.d2)

    def test_iter(self):
        out = list(iter(self.combined))

        for i in (self.d1 + self.d2):
            self.assertIn(i, out)

    def test_len(self):
        self.assertEqual(len(self.combined), 6)

    def test_shuffle_true(self):
        out1 = list(iter(self.combined))
        out2 = list(iter(self.combined))

        self.assertNotEqual(out1, out2)

    def test_shuffle_false(self):
        self.combined = CombinedDataLoader(False, self.d1, self.d2)

        out1 = list(iter(self.combined))
        out2 = list(iter(self.combined))

        self.assertEqual(out1, out2)