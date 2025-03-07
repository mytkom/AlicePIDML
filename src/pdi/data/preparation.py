""" 
This module contains Classes responsible for preprocessing data.

Classes
------
DeletePreparation
    Preprocesses data by deleting incomplete cases.
MeanImputation
    Imputes missing values with mean of features, calculated on train data.
RegressionImputation
    Imputes missing values by using a regression model, fitted on always available values.
EnsemblePreparation
    Splits dataset into subsets, grouped by the combinations of missing values.
FeatureSetPreparation
    Converts incomplete vectors into sets of feature-value pairs, and groups these sets based on the number of pairs.
"""

import json
from itertools import combinations
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pdi.data.constants import (
    DROP_COLUMNS_SMALL,
    DROP_COLUMNS_BIG,
    N_COLUMNS_BIG,
    N_COLUMNS_NSIGMAS,
    NSIGMA_COLUMNS,
    PROCESSED_DIR,
    COLUMNS_DROPPED_FOR_TESTS,
)
from pdi.data.types import Additional, GroupID, InputTarget, Split, COLUMN_DETECTOR
from pdi.data.config import RUN, UNDERSAMPLE
from pdi.data.utils import DataPreparation, GroupedDataPreparation
from pdi.data.detector_helpers import columns_to_detectors_masked

class DeletePreparation(DataPreparation):
    """DeletePreparation preprocesses incomplete data by deleting incomplete examples"""

    def __init__(self, complete_only: bool = True, base_dir = PROCESSED_DIR):
        """__init__

        Args:
            complete_only (bool, optional):
                Given for consistency between different preparation classes.
                Can only be True. Raises Exception if False.

        Raises:
            Exception: raised if complete_only=False.
        """
        self.save_dir = f"{base_dir}/deleted/run{RUN}"
        super().__init__()
        if not complete_only:
            raise ValueError("DeletePreparation can only prepare complete cases.")

    def _do_preprocess_split(self, split):
        input_split = split[InputTarget.INPUT]
        targets = split[InputTarget.TARGET]

        na_rows = input_split[input_split.isnull().any(axis=1)].index

        split[InputTarget.INPUT] = input_split.drop(index=na_rows)
        split[InputTarget.TARGET] = targets.drop(index=na_rows)

        return super()._do_preprocess_split(split)


class MeanImputation(DataPreparation):
    """MeanImputation preprocesses incomplete data by imputing missing values with statistical means calculated on the train split."""

    def __init__(self, complete_only: bool = False, base_dir = PROCESSED_DIR):
        """__init__

        Args:
            complete_only (bool, optional):
                Given for consistency between different preparation classes.
                Can only be False. Raises Exception if True.
                Use DeletePreparation if only complete examples are required.

        Raises:
            Exception: raised if complete_only=True.
        """
        super().__init__()
        if complete_only:
            raise ValueError("Use DeletePreparation to only get complete cases.")
        
        self.save_dir = f"{base_dir}/mean/run{RUN}"

    def _do_process_data(self, data):
        self._mean_df = data[Split.TRAIN][InputTarget.INPUT].mean()
        return super()._do_process_data(data)

    def _do_preprocess_split(self, split):
        split[InputTarget.INPUT] = split[InputTarget.INPUT].fillna(self._mean_df)
        return super()._do_preprocess_split(split)

    def save_data(self) -> None:
        """save_data overrides base method. Additionally saves the mean values used in imputation."""
        super().save_data()
        self._mean_df.to_json(f"{self.save_dir}/column_mean.json", index=True)


class RegressionImputation(DataPreparation):
    """RegressionImputation preprocesses incomplete data by imputing missing values by using a linear regression model fitted on the train split."""

    def __init__(self, complete_only: bool = False, base_dir = PROCESSED_DIR):
        """__init__

        Args:
            complete_only (bool, optional):
                Given for consistency between different preparation classes.
                Can only be False. Raises Exception if True.
                Use DeletePreparation if only complete examples are required.

        Raises:
            Exception: raised if complete_only=True.
        """
        super().__init__()
        if complete_only:
            raise ValueError("Use DeletePreparation to only get complete cases.")
        self._regression = LinearRegression()
        self.save_dir = f"{base_dir}/regression/run{RUN}"

    def _do_process_data(self, data):
        train_input = data[Split.TRAIN][InputTarget.INPUT]

        train_input_n = train_input[
            train_input.columns[~train_input.columns.isin(COLUMNS_DROPPED_FOR_TESTS + NSIGMA_COLUMNS)]
        ]
        self.missing_features = train_input_n.isnull().any()
        print(self.missing_features.shape)
        complete = train_input_n.dropna()
        self._regression.fit(
            complete.loc[:, ~self.missing_features],
            complete.loc[:, self.missing_features],
        )
        self.regression_params = {
            "missing": list(train_input_n.columns[self.missing_features]),
            "non_missing": list(train_input_n.columns[~self.missing_features]),
            "coef": self._regression.coef_.tolist(),
        }
        return super()._do_process_data(data)

    def _do_preprocess_split(self, split):
        input_split = split[InputTarget.INPUT]
        add_dict1 = {
            column: input_split.loc[:, [column.name]].values
            for column in Additional
            if column.name in NSIGMA_COLUMNS
        }
        if len(input_split.columns) == N_COLUMNS_NSIGMAS:
            input_split.drop(columns=NSIGMA_COLUMNS, inplace=True)
        targets = split[InputTarget.TARGET]
        add_dict = {column: input_split.loc[:, [column.name]].values
                    for column in Additional if column.name not in NSIGMA_COLUMNS}
        add_dict.update(add_dict1)

        input_split.drop(columns=COLUMNS_DROPPED_FOR_TESTS, inplace=True)

        print(input_split.shape)
        pred = self._regression.predict(input_split.loc[:, ~self.missing_features])
        pred = pd.DataFrame(pred,
                            columns=input_split.columns[self.missing_features],
                            index=input_split.index)

        columns_for_training = pd.Series(input_split.columns.tolist())
        columns_for_training = columns_for_training[~columns_for_training.isin(NSIGMA_COLUMNS)]
        self._columns_for_training = columns_for_training

        input_split = input_split.fillna(pred)
        return (
            {
                InputTarget.INPUT: input_split.values,
                InputTarget.TARGET: targets.values,
            },
            add_dict,
        )

    def save_data(self) -> None:
        """save_data overrides base method. Additionally saves the coefficients of the regression model used in imputation."""
        super().save_data()
        with open(f"{self.save_dir}/regression_coef.json", "wt", encoding="UTF-8") as file:
            json.dump(self.regression_params, file)


class EnsemblePreparation(GroupedDataPreparation):
    """EnsemblePreparation groups incomplete data into separate complete datasets, based on the combinations of missing values."""

    # TODO: remove hardcoded 19 (probably change group encoding somewhere)
    COMPLETE_GROUP_ID = np.sum(2 ** np.arange(19))

    def __init__(self, complete_only: bool = False):
        """__init__

        Args:
            complete_only (bool, optional): Whether to return only the group with complete examples. Defaults to False.
        """
        super().__init__(complete_only)
        self.save_dir = f"{PROCESSED_DIR}/ensemble/run{RUN}"

    def _group_data(self, data):
        missing = data.isnull()

        drop_columns = (
            DROP_COLUMNS_BIG
            if len(data.columns) == N_COLUMNS_BIG
            else DROP_COLUMNS_SMALL
        )
        missing.drop(columns=drop_columns, inplace=True)
        if len(missing.columns) == N_COLUMNS_NSIGMAS:
            missing.drop(columns=NSIGMA_COLUMNS, inplace=True)

        groups = missing.groupby(list(missing.columns), dropna=False).groups

        return {
            GroupID(np.sum(2 ** np.arange(len(key)) * ~np.array(key))): data.loc[val]
            for key, val in groups.items()
        }

    def _do_preprocess_split(self, split):
        input_data = split[InputTarget.INPUT]
        add_data = {
            column: input_data.loc[:, [column.name]].values for column in Additional
        }
        if len(input_data.columns) == N_COLUMNS_NSIGMAS:
            input_data.drop(columns=NSIGMA_COLUMNS, inplace=True)
        targets = split[InputTarget.TARGET]
        return (
            {
                InputTarget.INPUT: input_data.dropna(axis="columns").values,
                InputTarget.TARGET: targets.values,
            },
            add_data,
        )


class FeatureSetPreparation(GroupedDataPreparation):
    """FeatureSetPreparation converts incomplete vectors into sets of feature-value pairs, which are grouped based on the number of pairs in the set."""
    
    COMPLETE_GROUP_ID = GroupID(columns_to_detectors_masked(COLUMN_DETECTOR.keys()))

    def __init__(self, complete_only: bool = False, base_dir = PROCESSED_DIR, undersample: bool = UNDERSAMPLE):
        """__init__

        Args:
            complete_only (bool, optional): Whether to return only the group with complete examples. Defaults to False.
        """
        self.undersample = undersample
        super().__init__(complete_only)
        self.save_dir: str = f"{base_dir}/feature_set/run{RUN}"

    def _group_data(self, data):
        cols = list(COLUMN_DETECTOR.keys())

        col_combinations = []
        for i in range(len(cols) + 1):
            els = [list(x) for x in combinations(cols, i)]
            col_combinations.extend(els)

        print(f"Data shape: {data.shape}")
        groups = {}
        smallest_group_size = sys.maxsize * 2 + 1
        for missing in col_combinations:
            not_missing = list(filter(lambda i: i not in missing, cols)) # pylint: disable=cell-var-from-loop
            group = data[
                data[not_missing].notnull().all(1) & data[missing].isnull().all(1)
            ]
            if len(group.index) > 0:
                key = columns_to_detectors_masked(not_missing)
                groups[key] = group
                group_size = len(group.index)
                smallest_group_size = min(smallest_group_size, group_size)
        print(f"Group count: {len(groups)}")

        # undersampling
        if self.undersample:
            for key, group in groups.items():
                to_drop = len(group.index) - smallest_group_size
                if to_drop > 0:
                    # TODO: set configurable seed for sampling to get repeatable results
                    groups[key] = groups[key].sample(frac=1).reset_index(drop=True) # shuffles data frame
                    groups[key].drop(groups[key].tail(to_drop).index, inplace=True)

        return groups

    def _do_preprocess_split(self, split):
        input_data = split[InputTarget.INPUT]
        add_data = {
            column: input_data.loc[:, [column.name]].values for column in Additional
        }
        if len(input_data.columns) == N_COLUMNS_NSIGMAS:
            input_data.drop(columns=NSIGMA_COLUMNS, inplace=True)

        input_data.drop(columns=COLUMNS_DROPPED_FOR_TESTS, inplace=True)

        columns_for_training = pd.Series(input_data.columns.tolist())
        columns_for_training = columns_for_training[~columns_for_training.isin(NSIGMA_COLUMNS)]
        self._columns_for_training = columns_for_training

        targets = split[InputTarget.TARGET]
        return (
            {InputTarget.INPUT: input_data.values, InputTarget.TARGET: targets.values},
            add_data,
        )

    def save_data(self) -> None:
        super().save_data()

        with open(f"{self.save_dir}/columns_for_training.json", "w+", encoding="UTF-8") as f:
            f.write(
                json.dumps(
                    {"columns_for_training": self._columns_for_training.tolist()}
                )
            )

