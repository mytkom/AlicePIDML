import json
from typing import Hashable, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pdi.data.constants import N_COLUMNS, PROCESSED_DIR, DROP_COLUMNS
from pdi.data.types import (Additional, AddMap, DfOrArray, GIDMap, GroupID,
                            InputTarget, InputTargetMap, Split, SplitMap)
from pdi.data.utils import DataPreparation, GroupedDataPreparation
from sklearn.linear_model import LinearRegression  # type: ignore



class DeletePreparation(DataPreparation):
    save_dir = f"{PROCESSED_DIR}/deleted"

    def __init__(self, complete_only: bool = True):
        super().__init__()
        if not complete_only:
            raise Exception(
                "DeletePreparation can only prepare complete cases.")

    def _do_preprocess_split(
        self, split: InputTargetMap[pd.DataFrame]
    ) -> tuple[InputTargetMap[NDArray[np.float32]],
               AddMap[NDArray[np.float32]]]:

        input = split[InputTarget.INPUT]
        targets = split[InputTarget.TARGET]
        na_rows = input[input.isnull().any(axis=1)].index

        input = input.drop(index=na_rows)
        targets = targets.drop(index=na_rows)

        return ({
            InputTarget.INPUT: input.values,
            InputTarget.TARGET: targets.values
        }, {
            column: input.loc[:, [column.name]].values
            for column in Additional
        })


class MeanImputation(DataPreparation):
    save_dir = f"{PROCESSED_DIR}/mean"

    def __init__(self, complete_only: bool = False):
        super().__init__()
        if complete_only:
            raise Exception(
                "Use DeletePreparation to only get complete cases.")

    def _do_process_data(
        self, data: SplitMap[InputTargetMap[pd.DataFrame]]
    ) -> tuple[SplitMap[InputTargetMap[NDArray[np.float32]]],
               SplitMap[AddMap[NDArray[np.float32]]]]:

        self.mean_df = data[Split.TRAIN][InputTarget.INPUT].mean()
        return super()._do_process_data(data)

    def _do_preprocess_split(
        self, split: InputTargetMap[pd.DataFrame]
    ) -> tuple[InputTargetMap[NDArray[np.float32]],
               AddMap[NDArray[np.float32]]]:

        input = split[InputTarget.INPUT].fillna(self.mean_df)
        targets = split[InputTarget.TARGET]
        return ({
            InputTarget.INPUT: input.values,
            InputTarget.TARGET: targets.values
        }, {
            column: input.loc[:, [column.name]].values
            for column in Additional
        })

    def save_data(self) -> None:
        super().save_data()
        self.mean_df.to_json(f"{self.save_dir}/column_mean.json", index=True)


class RegressionImputation(DataPreparation):
    save_dir = f"{PROCESSED_DIR}/regression"

    def __init__(self, complete_only: bool = False):
        super().__init__()
        if complete_only:
            raise Exception(
                "Use DeletePreparation to only get complete cases.")
        self.regression = LinearRegression()

    def _do_process_data(
        self, data: SplitMap[InputTargetMap[pd.DataFrame]]
    ) -> tuple[SplitMap[InputTargetMap[NDArray[np.float32]]],
               SplitMap[AddMap[NDArray[np.float32]]]]:

        train_input = data[Split.TRAIN][InputTarget.INPUT]

        self.missing_features = train_input.isnull().any()
        complete = train_input.dropna()
        self.regression.fit(complete.loc[:, ~self.missing_features],
                            complete.loc[:, self.missing_features])
        self.regression_params = {
            "missing": list(train_input.columns[self.missing_features]),
            "non_missing": list(train_input.columns[~self.missing_features]),
            "coef": self.regression.coef_.tolist(),
        }
        return super()._do_process_data(data)

    def _do_preprocess_split(
        self, split: InputTargetMap[pd.DataFrame]
    ) -> tuple[InputTargetMap[NDArray[np.float32]],
               AddMap[NDArray[np.float32]]]:

        input = split[InputTarget.INPUT]
        targets = split[InputTarget.TARGET]
        pred = self.regression.predict(input.loc[:, ~self.missing_features])
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
        super().save_data()
        with open(f"{self.save_dir}/regression_coef.json", "wt") as file:
            json.dump(self.regression_params, file)


class EnsemblePreparation(GroupedDataPreparation):
    save_dir = f"{PROCESSED_DIR}/ensemble"
    COMPLETE_GROUP_ID = np.sum(2**np.arange(N_COLUMNS))

    def __init__(self, complete_only: bool = False):
        super().__init__(complete_only)

    def _group_data(self, data: pd.DataFrame) -> GIDMap[pd.DataFrame]:
        missing = data.isnull()
        groups = missing.groupby(list(missing.columns.drop(DROP_COLUMNS)), dropna=False).groups
        return {
            GroupID(
                np.sum(2**np.arange(len(cast(list[bool], key))) *
                       ~np.array(key))): data.loc[val]
            for key, val in groups.items()
        }

    def _do_preprocess_split(
        self, split: InputTargetMap[pd.DataFrame]
    ) -> tuple[InputTargetMap[NDArray[np.float32]],
               AddMap[NDArray[np.float32]]]:

        input_data = split[InputTarget.INPUT]
        targets = split[InputTarget.TARGET]
        add_data: AddMap[NDArray[np.float32]] = {
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
    save_dir: str = f"{PROCESSED_DIR}/feature_set"
    COMPLETE_GROUP_ID = GroupID(0)

    def __init__(self, complete_only: bool = False):
        super().__init__(complete_only)

    def _group_data(self, data: pd.DataFrame) -> GIDMap[pd.DataFrame]:
        missing_values = data.isnull().sum(axis=1)
        groups = data.groupby(missing_values.tolist(), dropna=False).groups
        return {
            GroupID(cast(int, gid)): data.loc[group_idx]
            for gid, group_idx in groups.items()
        }

    def _do_preprocess_split(
        self, split: InputTargetMap[pd.DataFrame]
    ) -> tuple[InputTargetMap[NDArray[np.float32]],
               AddMap[NDArray[np.float32]]]:

        def make_feature_set(df: pd.DataFrame) -> NDArray[np.float32]:
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
            feature_set: NDArray[np.float32] = np.concatenate(
                (one_hot_indices, feature_values), -1)
            return feature_set

        input_data = split[InputTarget.INPUT]
        targets = split[InputTarget.TARGET]
        add_data: AddMap[NDArray[np.float32]] = {
            column: input_data.loc[:, [column.name]].values
            for column in Additional
        }
        return ({
            InputTarget.INPUT: make_feature_set(input_data),
            InputTarget.TARGET: targets.values
        }, add_data)
