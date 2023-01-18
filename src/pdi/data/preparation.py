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

import numpy as np
import pandas as pd
from pdi.data.constants import DROP_COLUMNS, N_COLUMNS, PROCESSED_DIR
from pdi.data.types import Additional, GroupID, InputTarget, Split
from pdi.data.utils import DataPreparation, GroupedDataPreparation
from sklearn.linear_model import LinearRegression


class DeletePreparation(DataPreparation):
    """DeletePreparation preprocesses incomplete data by deleting incomplete examples
    """

    save_dir = f"{PROCESSED_DIR}/deleted"

    def __init__(self, complete_only: bool = True):
        """__init__

        Args:
            complete_only (bool, optional): 
                Given for consistency between different preparation classes.
                Can only be True. Raises Exception if False.

        Raises:
            Exception: raised if complete_only=False.
        """
        super().__init__()
        if not complete_only:
            raise Exception(
                "DeletePreparation can only prepare complete cases.")

    def _do_preprocess_split(self, split):
        input = split[InputTarget.INPUT]
        targets = split[InputTarget.TARGET]

        na_rows = input[input.isnull().any(axis=1)].index

        split[InputTarget.INPUT] = input.drop(index=na_rows)
        split[InputTarget.TARGET] = targets.drop(index=na_rows)

        return super()._do_preprocess_split(split)


class MeanImputation(DataPreparation):
    """MeanImputation preprocesses incomplete data by imputing missing values with statistical means calculated on the train split.
    """
    save_dir = f"{PROCESSED_DIR}/mean"

    def __init__(self, complete_only: bool = False):
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
            raise Exception(
                "Use DeletePreparation to only get complete cases.")

    def _do_process_data(self, data):
        self._mean_df = data[Split.TRAIN][InputTarget.INPUT].mean()
        return super()._do_process_data(data)

    def _do_preprocess_split(self, split):
        split[InputTarget.INPUT] = split[InputTarget.INPUT].fillna(
            self._mean_df)
        return super()._do_preprocess_split(split)

    def save_data(self) -> None:
        """save_data overrides base method. Additionally saves the mean values used in imputation. 
        """
        super().save_data()
        self._mean_df.to_json(f"{self.save_dir}/column_mean.json", index=True)


class RegressionImputation(DataPreparation):
    """RegressionImputation preprocesses incomplete data by imputing missing values by using a linear regression model fitted on the train split.
    """
    save_dir = f"{PROCESSED_DIR}/regression"

    def __init__(self, complete_only: bool = False):
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
            raise Exception(
                "Use DeletePreparation to only get complete cases.")
        self._regression = LinearRegression()

    def _do_process_data(self, data):
        train_input = data[Split.TRAIN][InputTarget.INPUT]

        self.missing_features = train_input.isnull().any()
        complete = train_input.dropna()
        self._regression.fit(complete.loc[:, ~self.missing_features],
                             complete.loc[:, self.missing_features])
        self.regression_params = {
            "missing": list(train_input.columns[self.missing_features]),
            "non_missing": list(train_input.columns[~self.missing_features]),
            "coef": self._regression.coef_.tolist(),
        }
        return super()._do_process_data(data)

    def _do_preprocess_split(self, split):

        input = split[InputTarget.INPUT]
        targets = split[InputTarget.TARGET]
        pred = self._regression.predict(input.loc[:, ~self.missing_features])
        pred = pd.DataFrame(pred,
                            columns=input.columns[self.missing_features],
                            index=input.index)

        input = input.fillna(pred)
        return ({
            InputTarget.INPUT: input.values,
            InputTarget.TARGET: targets.values
        }, {
            column: input.loc[:, [column.name]].values
            for column in Additional
        })

    def save_data(self) -> None:
        """save_data overrides base method. Additionally saves the coefficients of the regression model used in imputation. 
        """
        super().save_data()
        with open(f"{self.save_dir}/regression_coef.json", "wt") as file:
            json.dump(self.regression_params, file)


class EnsemblePreparation(GroupedDataPreparation):
    """EnsemblePreparation groups incomplete data into separate complete datasets, based on the combinations of missing values.
    """
    save_dir = f"{PROCESSED_DIR}/ensemble"
    COMPLETE_GROUP_ID = np.sum(2**np.arange(N_COLUMNS))

    def __init__(self, complete_only: bool = False):
        """__init__

        Args:
            complete_only (bool, optional): Whether to return only the group with complete examples. Defaults to False.
        """
        super().__init__(complete_only)

    def _group_data(self, data):
        missing = data.isnull()
        groups = missing.groupby(list(missing.columns.drop(DROP_COLUMNS)),
                                 dropna=False).groups
        return {
            GroupID(np.sum(2**np.arange(len(key)) * ~np.array(key))):
            data.loc[val]
            for key, val in groups.items()
        }

    def _do_preprocess_split(self, split):

        input_data = split[InputTarget.INPUT]
        targets = split[InputTarget.TARGET]
        add_data = {
            column: input_data.loc[:, [column.name]].values
            for column in Additional
        }
        return (
            {
                InputTarget.INPUT: input_data.dropna(axis="columns").values,
                InputTarget.TARGET: targets.values,
            },
            add_data,
        )


class FeatureSetPreparation(GroupedDataPreparation):
    """FeatureSetPreparation converts incomplete vectors into sets of feature-value pairs, which are grouped based on the number of pairs in the set.
    """
    save_dir: str = f"{PROCESSED_DIR}/feature_set"
    COMPLETE_GROUP_ID = GroupID(0)

    def __init__(self, complete_only: bool = False):
        """__init__

        Args:
            complete_only (bool, optional): Whether to return only the group with complete examples. Defaults to False.
        """
        super().__init__(complete_only)

    def _group_data(self, data):
        missing_values = data.isnull().sum(axis=1)
        groups = data.groupby(missing_values.tolist(), dropna=False).groups
        return {
            GroupID(gid): data.loc[group_idx]
            for gid, group_idx in groups.items()
        }

    def _do_preprocess_split(self, split):

        def make_feature_set(df):
            rows, cols = df.shape
            non_null_count = df.iloc[0].count()
            features = non_null_count
            is_not_null = pd.notnull(df)
            feature_indices = np.nonzero(is_not_null.values)[-1]
            one_hot_indices = np.reshape(
                np.eye(N_COLUMNS)[feature_indices],
                (rows, features, N_COLUMNS))
            feature_values = np.reshape(df.values[is_not_null],
                                        (rows, features, 1))
            feature_set = np.concatenate((one_hot_indices, feature_values), -1)
            return feature_set

        input_data = split[InputTarget.INPUT]
        targets = split[InputTarget.TARGET]
        add_data = {
            column: input_data.loc[:, [column.name]].values
            for column in Additional
        }
        return ({
            InputTarget.INPUT: make_feature_set(input_data),
            InputTarget.TARGET: targets.values
        }, add_data)
