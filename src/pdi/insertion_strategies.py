from abc import abstractmethod
from typing import List
from numpy.typing import NDArray
import pandas as pd
from sklearn.linear_model import LinearRegression

from pdi.data.constants import MAX_GROUP_ID
from pdi.data.data_preparation import PreparedData
from pdi.data.group_id_helpers import group_id_to_binary_array
from pdi.data.types import GroupID, InputTarget, Split

class InsertionStrategy:
    def __init__(self, prepared_data: PreparedData) -> None:
        self._insert(prepared_data)

    @abstractmethod
    def _insert(self, prepared_data: PreparedData):
        pass

class MeanInsertion(InsertionStrategy):
    def __init__(self, prepared_data: PreparedData) -> None:
        super().__init__(prepared_data)

    def insert(self, prepared_data: PreparedData):
        ungrouped_train_data = pd.concat([group[InputTarget.INPUT] for group in prepared_data[Split.TRAIN].values()])
        means = ungrouped_train_data.mean()

        for split_data in prepared_data.values():
            split_data: dict[GroupID, dict[InputTarget, pd.DataFrame]]

            for group_data in split_data.values():
                group_data: dict[InputTarget, pd.DataFrame]

                group_data[InputTarget.INPUT].fillna(means, inplace=True)

# If we would like in the future to test more Regression models.
# This class can be for sure generalize to DRY.
class LinearRegressionInsertion(InsertionStrategy):
    def __init__(self, prepared_data: PreparedData) -> None:
        self._regressions: dict[GroupID, LinearRegression] = {}
        self._missing_features_masks: dict[GroupID, NDArray] = {}
        self._regression_params: dict[GroupID, dict[str, List[str] | List[float]]] = {}
        super().__init__(prepared_data)

    def insert(self, prepared_data: PreparedData):
        grouped_train_data = prepared_data[Split.TRAIN]
        complete_train_data = grouped_train_data[MAX_GROUP_ID][InputTarget.INPUT]

        for gid, group_data in grouped_train_data.items():
            gid: GroupID
            group_data: dict[InputTarget, pd.DataFrame]

            # For complete data, there is no regression performed (it doesn't make sense)
            if gid == MAX_GROUP_ID:
                continue

            # Decode information about missing features from GroupID
            missing_features = group_id_to_binary_array(gid) == 0
            regression = LinearRegression()
            regression.fit(
                complete_train_data.loc[:, ~missing_features],
                complete_train_data.loc[:, missing_features],
            )

            # Save resulting models
            self._missing_features_masks[gid] = missing_features
            self._regressions[gid] = regression
            self._regression_params[gid] = {
                "missing_columns": list(group_data[InputTarget.INPUT].columns[missing_features]),
                "non_missing_columns": list(group_data[InputTarget.INPUT].columns[~missing_features]),
                "regression_coefficients": regression.coef_.tolist()
            }

        for split_data in prepared_data.values():
            split_data: dict[GroupID, dict[InputTarget, pd.DataFrame]]

            for gid, group_data in split_data.items():
                gid: GroupID
                group_data: dict[InputTarget, pd.DataFrame]

                # Skip complete data like before
                if gid == MAX_GROUP_ID:
                    continue

                input_data = group_data[InputTarget.INPUT]

                # Predict missing values
                pred = self._regressions[gid].predict(input_data.loc[:, ~self._missing_features_masks[gid]])

                # Input predicted values
                input_data.loc[:, self._missing_features_masks[gid]] = pred

MISSING_DATA_STRATEGIES = {
    "mean": MeanInsertion,
    "linear regression": LinearRegressionInsertion
}
