import unittest
import torch
import pandas as pd
import numpy as np
from pdi.data.constants import (COLUMNS_TO_SCALE, DROP_COLUMNS, MISSING_VALUES,
                                TARGET_COLUMN, GROUP_ID_KEY)
from pdi.data.types import Additional, GroupID, Split
from pdi.data.utils import GroupedDataPreparation

from unittest.mock import Mock


class TestGroupedDataPreparation(unittest.TestCase):

    def setUp(self):
        columns = COLUMNS_TO_SCALE + list(MISSING_VALUES.keys(
        )) + DROP_COLUMNS + [i.name for i in Additional]
        columns.remove(TARGET_COLUMN)

        self.mock_data = pd.DataFrame({
            **{
                TARGET_COLUMN: [0] * 800 + [1] * 200 + [2] * 1000,
            },
            **{c: 10 * np.random.rand((2000)) + 20
               for c in columns}
        })

        self.mock_data.loc[self.mock_data[TARGET_COLUMN] == 2,
                           columns[0]] = np.nan

        def groupby(data):
            missing = data.isnull().any(axis="columns")
            groups = data.groupby(missing.tolist(), dropna=False).groups
            return {GroupID(key): data.loc[val] for key, val in groups.items()}

        self.mock_group = groupby

    def test_posweight(self):
        prep = GroupedDataPreparation(False)
        prep._load_input_data = Mock(return_value=self.mock_data)
        prep._group_data = self.mock_group

        prep.prepare_data()
        self.assertEqual(prep.pos_weight(0), 1200 / 800)
        self.assertEqual(prep.pos_weight(1), 1800 / 200)
        self.assertEqual(prep.pos_weight(2), 1000 / 1000)

    def test_getgroupids(self):
        prep = GroupedDataPreparation(False)
        prep._load_input_data = Mock(return_value=self.mock_data)
        prep._group_data = self.mock_group

        prep.prepare_data()

        self.assertEqual(prep.get_group_ids(), [0, 1])

    def test_completeonly(self):
        prep = GroupedDataPreparation(True)
        prep.COMPLETE_GROUP_ID = 0
        prep._load_input_data = Mock(return_value=self.mock_data)
        prep._group_data = self.mock_group

        prep.prepare_data()

        (dataloader, ) = prep.prepare_dataloaders(64, 0, [Split.TRAIN])

        for _, target, _ in dataloader:
            self.assertTrue(torch.all(target != 2))

    def test_dataloaders(self):
        prep = GroupedDataPreparation(False)
        prep._load_input_data = Mock(return_value=self.mock_data)
        prep._group_data = self.mock_group

        target_count = {0: 0, 1: 0, 2: 0}

        prep.prepare_data()

        (dataloader, ) = prep.prepare_dataloaders(64, 0, [Split.TRAIN])

        train_cases = 0

        for _, target, add in dataloader:
            train_cases += target.squeeze().size()[0]
            for class_ in target_count.keys():
                target_count[class_] += torch.sum(target == class_).item()

            self.assertTrue(
                torch.all(add[GROUP_ID_KEY] == add[GROUP_ID_KEY][0]))

            if add[GROUP_ID_KEY][0] == 0:
                self.assertTrue(torch.all(target != 2))
            else:
                self.assertTrue(torch.all(target == 2))

        self.assertAlmostEqual(target_count[0] / train_cases, 800 / 2000)
        self.assertAlmostEqual(target_count[1] / train_cases, 200 / 2000)
        self.assertAlmostEqual(target_count[2] / train_cases, 1000 / 2000)
