import unittest

import numpy as np
import pandas as pd
from pandas.util.testing import assert_almost_equal, assert_equal
from pdi.data.constants import (COLUMNS_TO_SCALE, MISSING_VALUES,
                                TARGET_COLUMN, TEST_SIZE, TRAIN_SIZE)
from pdi.data.types import Split
from pdi.data.utils import DataPreparation


class TestDataPreparation(unittest.TestCase):

    def setUp(self):
        self.prep = DataPreparation()

    def test_delete_unique_targets(self):
        targets = [1] * 80 + [2] * 1000
        data = pd.DataFrame({TARGET_COLUMN: targets})

        new_data = self.prep._delete_unique_targets(data)
        value_counts = new_data.value_counts()

        self.assertNotIn(1, value_counts)
        self.assertIn(2, value_counts)
        self.assertEqual(value_counts[2], 1000)

    def test_make_missing_null(self):
        data = pd.DataFrame()
        for (column, val) in MISSING_VALUES.items():
            data[column] = [val, 1, val]

        new_data = self.prep._make_missing_null(data)

        for (column, _) in MISSING_VALUES.items():
            assert_equal(new_data[column],
                         pd.Series([np.NaN, 1, np.NaN], name=column))

    def test_normalize(self):
        data = pd.DataFrame()
        for c in COLUMNS_TO_SCALE:
            data[c] = 10 * np.random.rand((100)) + 20

        new_data = self.prep._normalize_data(data)

        for c in COLUMNS_TO_SCALE:
            column_data = new_data[c]
            mean = np.mean(column_data)
            std = np.std(column_data)
            self.assertAlmostEqual(mean, 0)
            self.assertAlmostEqual(std, 1)

        assert_almost_equal(self.prep._scaling_params,
                            pd.DataFrame({
                                "column":
                                COLUMNS_TO_SCALE,
                                "mean": [25.0] * len(COLUMNS_TO_SCALE),
                                "std":
                                [10 * np.sqrt(1 / 12)] * len(COLUMNS_TO_SCALE)
                            }),
                            atol=0.5)

    def test_train_split_size(self):
        targets = [0] * 200 + [1] * 800
        data = pd.DataFrame({TARGET_COLUMN: targets})
        new_data = self.prep._test_train_split(data)

        self.assertAlmostEqual(new_data[Split.TRAIN].size, TRAIN_SIZE * 1000)
        self.assertAlmostEqual(new_data[Split.TEST].size, TEST_SIZE * 1000)
        self.assertAlmostEqual(new_data[Split.VAL].size,
                               (1 - TEST_SIZE - TRAIN_SIZE) * 1000)

    def test_train_split_stratify(self):
        targets = [0] * 200 + [1] * 800
        data = pd.DataFrame({TARGET_COLUMN: targets})
        new_data = self.prep._test_train_split(data)
        for split in new_data.values():
            counts = split.value_counts()
            ratio = counts[1] / counts[0]
            self.assertAlmostEqual(ratio, 4.0)

    def test_it_split(self):
        pass

    def test_process(self):
        pass

    def test_dataloaders(self):
        pass

    def test_posweight(self):
        pass
        self.assertEqual(self.prep.pos_weight(), 3)
