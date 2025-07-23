""" This module contains base classes for different preparation methods, as well as other helper classes.

Classes
------
CombinedDataLoader
    Combines multiple dataloaders and shuffles the batches returned by them.
DictDataset
    A mapping dataset with items consisting of an input tensor, a target tensor, and a dict of additional tensors.
DataPreparation
    Base class for preparation techniques without grouping.
GroupedDataPreparation
    Base class for preparation techniques with grouping.
"""

import gzip
import json
import os
import pickle
import sys
from itertools import chain, combinations
from random import Random
import typing
from typing import Generic, Iterable, Iterator, TypeVar, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from pdi.data.detector_helpers import columns_to_detectors_masked

from pdi.data.constants import (
    CSV_DELIMITER,
    DROP_COLUMNS_SMALL,
    DROP_COLUMNS_BIG,
    N_COLUMNS_BIG,
    N_COLUMNS_SMALL,
    N_COLUMNS_NSIGMAS,
    N_COLUMNS_ML,
    NSIGMA_COLUMNS,
    GROUP_ID_KEY,
    MISSING_VALUES,
    PROCESSED_DIR,
    SEED,
    TARGET_COLUMN,
    TEST_SIZE,
    TRAIN_SIZE,
    DO_NOT_SCALE,
    COLUMNS_DROPPED_FOR_TESTS,
    TPC_CUT,
)
from pdi.data.types import Additional, DatasetItem, GroupID, InputTarget, Split, COLUMN_DETECTOR

T = TypeVar("T")


class DataPreparation:
    """Base class for preparation techniques without grouping.

    Methods:
        pos_weight: return the ratio between the negative and positive samples in the train split.
        prepare_data: load and preprocess the data.
        save_data: save preprocessed data.
        prepare_dataloaders: create dataloaders from preprocessed data.
    """

    def __init__(self, base_dir = PROCESSED_DIR) -> None:
        self._scaling_params = pd.DataFrame(columns=["column", "mean", "std"])
        self._columns_for_training = None
        self._input_target = {}
        self._additional = {}
        self.save_dir = f"{base_dir}/basic/run{RUN}"

    def pos_weight(self, target: int) -> float:
        """pos_weight returns the ratio between the negative and positive samples in the train split.

        Args:
            `target (int): the target code for the positive samples.

        Returns:
            float: the ratio of negative to positive samples.
        """
        self._try_load_preprocessed_data([Split.TRAIN])

        train_targets = self._input_target[Split.TRAIN][InputTarget.TARGET]
        binary_targets = train_targets == target
        pos_weight: float = (np.size(binary_targets) - np.sum(binary_targets)) / (
            np.sum(binary_targets) + np.finfo(float).eps
        )
        return pos_weight

    def prepare_data(self, input_path: str = INPUT_PATH) -> None:
        """prepare_data loads and preprocesses data.
        Data is loaded from the path provided in `pdi.data.constants`
        """
        data = self._load_input_data(input_path)
        data = data.loc[data["fTPCSignal"] < TPC_CUT, :]
        data = self._delete_unique_targets(data)
        data = self._make_missing_null(data)
        data = self._normalize_data(data)
        split_data = self._test_train_split(data)
        split_data_2 = self._input_target_split(split_data)
        self._input_target, self._additional = self._do_process_data(split_data_2)

    def _load_input_data(self, input_path: str = INPUT_PATH):
        self.csv_name: str = os.path.splitext(os.path.basename(input_path))[0]
        return pd.read_csv(input_path, sep=CSV_DELIMITER, index_col=0)

    def _delete_unique_targets(self, data):
        THRESHOLD = 400
        target_counts = data[TARGET_COLUMN].value_counts()
        for target, count in target_counts.items():
            if count < THRESHOLD:
                data = data[data[TARGET_COLUMN] != target]
        return data

    def _make_missing_null(self, data):
        for column, val in MISSING_VALUES.items():
            data.loc[data[column] == val, column] = np.NaN

        if RUN == 2:
            print("Run 2 format")

        if RUN == 3:
            data.loc[data["fTRDPattern"].isnull(), ["fTRDSignal", "fTRDPattern"]] = np.NaN
            data.loc[data["fBeta"].isnull(), ["fTOFSignal", "fBeta"]] = np.NaN
            print("Run 3 format")
        return data

    def _normalize_data(self, data):
        for column in data.columns:
            if column in DO_NOT_SCALE or column in NSIGMA_COLUMNS:
                continue
            mean = np.mean(data[column])
            std = np.std(data[column])
            if std == 0:
                mean = 0
                std = 1
            self._scaling_params = pd.concat(
                [
                    self._scaling_params,
                    pd.DataFrame(
                        {
                            "column": column,
                            "mean": mean,
                            "std": std,
                        },
                        index=[0],
                    ),
                ],
                ignore_index=True,
            )
            data[column] = (data[column] - mean) / std
        return data

    def _test_train_split(self, data):
        train_to_val_ratio = TRAIN_SIZE / (1 - TEST_SIZE)
        (data_not_test, test_data) = train_test_split(
            data,
            test_size=TEST_SIZE,
            random_state=SEED,
            stratify=data.loc[:, [TARGET_COLUMN]],
        )
        (train_data, val_data) = train_test_split(
            data_not_test,
            train_size=train_to_val_ratio,
            random_state=SEED,
            stratify=data_not_test.loc[:, [TARGET_COLUMN]],
        )
        return {Split.TRAIN: train_data, Split.VAL: val_data, Split.TEST: test_data}

    def _input_target_split(self, data):
        def do_split(data):
            targets = data.loc[:, [TARGET_COLUMN]]
            if len(data.columns) == N_COLUMNS_BIG:
                input_data = data.drop(columns=DROP_COLUMNS_BIG)
            elif len(data.columns) == N_COLUMNS_SMALL:
                input_data = data.drop(columns=DROP_COLUMNS_SMALL)
            else:
                raise ValueError(
                    "The input table has invalid number of columns. Was the PID ML producer updated?"
                )

            return {InputTarget.INPUT: input_data, InputTarget.TARGET: targets}

        split_data = {}
        for key, val in data.items():
            split_data[key] = do_split(val)

        return split_data

    def _do_process_data(self, data):
        processed_data = {}
        additional_data = {}
        for key, val in data.items():
            processed_data[key], additional_data[key] = self._do_preprocess_split(val)
        return processed_data, additional_data

    def _do_preprocess_split(self, split):
        input_split = split[InputTarget.INPUT]
        targets = split[InputTarget.TARGET]
        add_dict = {column: input_split.loc[:, [column.name]].values
                    for column in Additional}
        if len(input_split.columns) == N_COLUMNS_NSIGMAS:
            input_split.drop(columns=NSIGMA_COLUMNS, inplace=True)

        input_split.drop(columns=COLUMNS_DROPPED_FOR_TESTS, inplace=True)

        columns_for_training = pd.Series(input_split.columns.tolist())
        columns_for_training = columns_for_training[~columns_for_training.isin(NSIGMA_COLUMNS)]
        self._columns_for_training = columns_for_training

        return (
            {
                InputTarget.INPUT: input_split.values,
                InputTarget.TARGET: targets.values,
            },
            add_dict,
        )

    def save_data(self) -> None:
        """save_data saves preprocessed data, as well as scaling parameters, to disk.
        Save location is given in the class variable `save_dir`
        """

        os.makedirs(self.save_dir, exist_ok=True)
        for split, it_data in self._input_target.items():
            with gzip.open(
                f"{self.save_dir}/input_target_{split.name}.pkl", "wb"
            ) as file:
                pickle.dump(it_data, file)

        for split, add_data in self._additional.items():
            with gzip.open(
                f"{self.save_dir}/additional_{split.name}.pkl", "wb"
            ) as file:
                pickle.dump(add_data, file)

        self._scaling_params.to_json(
            f"{self.save_dir}/scaling_params.json",
            index=False,
            orient="split",
        )

        with open(f"{self.save_dir}/columns_for_training.json", "w+", encoding="UTF-8") as f:
            f.write(
                json.dumps(
                    {"columns_for_training": self._columns_for_training.tolist()}
                )
            )

        with open(f"{self.save_dir}/csv_name.txt", "w", encoding="utf-8") as file:
            file.write(self.csv_name)


    def _load_preprocessed_data(self, splits):
        for split in splits:
            with gzip.open(
                f"{self.save_dir}/input_target_{split.name}.pkl", "rb"
            ) as file:
                self._input_target[split] = pickle.load(file)
            with gzip.open(
                f"{self.save_dir}/additional_{split.name}.pkl", "rb"
            ) as file:
                self._additional[split] = pickle.load(file)
        with open(f"{self.save_dir}/csv_name.txt", "r", encoding="utf-8") as file:
            self.csv_name = file.read().strip()

    def _try_load_preprocessed_data(self, splits):
        if any(
            key not in self._input_target or key not in self._additional
            for key in splits
        ):
            self._load_preprocessed_data(splits)

    def prepare_dataloaders(
        self, batch_size: int, num_workers: int, splits: Optional[list[Split]] = None
    ) -> tuple[Iterable[DatasetItem], ...]:
        """prepare_dataloaders creates dataloaders from preprocessed data.

        Args:
            batch_size (int): batch size
            num_workers (int): number of worker processed used in loading data.
                See `torch.utils.data.DataLoader` for more info.
            splits (list[Split], optional): list of splits to create dataloaders for.
                Defaults to [Split.TRAIN, Split.VAL, Split.TEST].

        Returns:
            tuple[Iterable[DatasetItem], ...]: tuple of dataloaders, one for each split.
        """
        if splits is None:
            splits = list(Split)

        self._try_load_preprocessed_data(splits)

        def create_dataset(input_target, additional):
            return DictDataset(
                torch.tensor(input_target[InputTarget.INPUT], dtype=torch.float32),
                torch.tensor(input_target[InputTarget.TARGET], dtype=torch.float32),
                **{
                    column.name: torch.tensor(val, dtype=torch.float32)
                    for column, val in additional.items()
                },
            )

        datasets = {
            split: create_dataset(self._input_target[split], self._additional[split])
            for split in splits
        }

        dataloaders = {
            key: DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(key == Split.TRAIN),
                num_workers=num_workers,
            )
            for key, dataset in datasets.items()
        }

        return (*dataloaders.values(),)

    def load_columns(self) -> list[str]:
        cols_for_training_path = self.save_dir

        with open(os.path.join(cols_for_training_path, "columns_for_training.json"), encoding="UTF-8") as f:
            data = json.load(f)

        cols = data["columns_for_training"]
        return cols


class GroupedDataPreparation(DataPreparation):
    """Base class for preparation techniques with grouping.

    Methods:
        pos_weight: return the ratio between the negative and positive samples in the train split, in all groups.
        prepare_data: load and preprocess the data.
        save_data: save preprocessed data.
        prepare_dataloaders: create dataloaders from preprocessed data.

    """

    COMPLETE_GROUP_ID: GroupID

    def __init__(self, complete_only: bool, undersample: bool = UNDERSAMPLE):
        super().__init__()
        self.complete_only = complete_only
        self.grouped_it = {}
        self.grouped_add = {}
        self.undersample = undersample

    def pos_weight(self, target: int) -> float:
        self._try_load_preprocessed_data([Split.TRAIN])
        target_list = [
            group[Split.TRAIN][InputTarget.TARGET] for group in self.grouped_it.values()
        ]
        binary_targets = np.concatenate([targets == target for targets in target_list])
        pos_weight: float = (np.size(binary_targets) - np.sum(binary_targets)) / np.sum(
            binary_targets
        )
        return pos_weight

    def get_group_ids(self) -> list[GroupID]:
        """get_group_ids returns the ids of groups in the dataset

        Returns:
            list[GroupID]: list of group ids
        """
        self._try_load_preprocessed_data([Split.TRAIN])
        return list(self.grouped_it.keys())

    def prepare_data(self, input_path: str = INPUT_PATH) -> None:
        data = self._load_input_data(input_path)
        data = data.loc[data["fTPCSignal"] < TPC_CUT, :]
        data = self._delete_unique_targets(data)
        data = self._make_missing_null(data)
        data = self._normalize_data(data)
        group_dict = self._group_data(data)
        for gid, group_data in group_dict.items():
            split_data = self._test_train_split(group_data)
            split_data_2 = self._input_target_split(split_data)
            self.grouped_it[gid], self.grouped_add[gid] = self._do_process_data(
                split_data_2
            )

    def _group_data(self, _):
        return {}

    def save_data(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

        def save_dict(name, dictionary):
            transposed_dict = {split: {} for split in list(Split)}
            for gid, group_data in dictionary.items():
                for split, sd in group_data.items():
                    transposed_dict[split][gid] = sd

            for split, split_data in transposed_dict.items():
                with gzip.open(
                    f"{self.save_dir}/{name}_{split.name}.pkl", "wb"
                ) as file:
                    pickle.dump(split_data, file)

        save_dict("input_target", self.grouped_it)
        save_dict("additional", self.grouped_add)

        self._scaling_params.to_json(
            f"{self.save_dir}/scaling_params.json",
            index=False,
            orient="split",
        )
                    
        with open(f"{self.save_dir}/csv_name.txt", "w", encoding="utf-8") as file:
            file.write(self.csv_name)

    def _load_preprocessed_data(self, splits):
        for split in splits:
            with gzip.open(
                f"{self.save_dir}/input_target_{split.name}.pkl", "rb"
            ) as file:
                data = pickle.load(file)
                for gid, group_data in data.items():
                    self.grouped_it.setdefault(gid, {})
                    self.grouped_it[gid][split] = group_data

            with gzip.open(
                f"{self.save_dir}/additional_{split.name}.pkl", "rb"
            ) as file:
                data = pickle.load(file)
                for gid, group_data in data.items():
                    self.grouped_add.setdefault(gid, {})
                    self.grouped_add[gid][split] = group_data
            
        with open(f"{self.save_dir}/csv_name.txt", "r", encoding="utf-8") as file:
            self.csv_name = file.read().strip()

    def _try_load_preprocessed_data(self, splits):
        are_splits_loaded = [
            split in group_dict
            for group_dict in chain(self.grouped_it.values(), self.grouped_add.values())
            for split in splits
        ]

        if not all(are_splits_loaded) or len(are_splits_loaded) == 0:
            self._load_preprocessed_data(splits)

    def prepare_dataloaders(
        self,
        batch_size: int,
        num_workers: int,
        splits: list[Split] = None,
        groupId: GroupID = None
    ) -> tuple[Iterable[DatasetItem], ...]:
        if splits is None:
            splits = list(Split)

        self._try_load_preprocessed_data(splits)

        def create_datasets(input_target, additional, gid, splits):
            return {
                split: DictDataset(
                    torch.tensor(
                        input_target[split][InputTarget.INPUT], dtype=torch.float32
                    ),
                    torch.tensor(
                        input_target[split][InputTarget.TARGET], dtype=torch.float32
                    ),
                    **{
                        GROUP_ID_KEY: torch.full(
                            input_target[split][InputTarget.TARGET].shape, gid
                        ),
                        **{
                            column.name: torch.tensor(val, dtype=torch.float32)
                            for column, val in additional[split].items()
                        },
                    },
                )
                for split in splits
            }

        datasets = {
            gid: create_datasets(
                data, self.grouped_add[gid], gid, splits
            )
            for gid, data in self.grouped_it.items()
        }

        if self.complete_only:
            dataloaders = {
                split: DataLoader(
                    datasets[self.COMPLETE_GROUP_ID][split],
                    batch_size,
                    shuffle=(split == Split.TRAIN),
                    num_workers=num_workers,
                )
                for split in splits
            }
        elif groupId:
            separate_dataloaders = {
                groupId: {
                    split: DataLoader(dataset,
                                      batch_size,
                                      shuffle=(split == Split.TRAIN),
                                      num_workers=num_workers)
                    for split, dataset in datasets[groupId].items()
                }
            }
            dataloaders = {
                split: CombinedDataLoader(
                    (split == Split.TRAIN),
                    *[d[split] for d in separate_dataloaders.values()],
                    undersample=self.undersample
                )
                for split in splits
            }
        else:
            separate_dataloaders = {
                gid: {
                    split: DataLoader(
                        dataset,
                        batch_size,
                        shuffle=(split == Split.TRAIN),
                        num_workers=num_workers,
                    )
                    for split, dataset in dataset_splits.items()
                }
                for gid, dataset_splits in datasets.items()
            }

            dataloaders = {
                split: CombinedDataLoader(
                    (split == Split.TRAIN),
                    *[d[split] for d in separate_dataloaders.values()],
                    undersample=self.undersample
                )
                for split in splits
            }

        return (*dataloaders.values(),)

    def data_to_ungrouped_df(self, splits: list[Split]) -> pd.DataFrame:
        cols = self.load_columns()
        cols.append(TARGET_COLUMN)
        inputs = []
        targets = []

        for _, group in self.grouped_it.items():
            for split in splits:
                inputs.append(group[split][InputTarget.INPUT])
                targets.append(group[split][InputTarget.TARGET])

        inputs = np.concatenate(inputs)
        targets = np.concatenate(targets)

        inputs_targets = np.concatenate((inputs, targets), axis=1)
        data = pd.DataFrame(data=inputs_targets, columns=cols)

        return data

    def data_to_df_dict(self, split: Split) -> dict[typing.Any, pd.DataFrame]:
        cols = self.load_columns()
        groups = {}
        for gid, group in self.grouped_it.items():
            data = group[split][InputTarget.INPUT]
            data = pd.DataFrame(data=data, columns=cols)
            groups[gid] = data

        return groups

# Experimental data/DANN related
class ExperimentDataset(Dataset[Tensor]):
    """ExperimentDataset is a mapping dataset containing items that consist of an input tensor.

    Base Class:
        Dataset
    """

    def __init__(
        self,
        input_tensors: Tensor
    ):
        self.input_tensors = input_tensors

    def __len__(self) -> int:
        return self.input_tensors.shape[0]

    def __getitem__(self, index: int) -> Tensor:
        return self.input_tensors[index]


class ExperimentFeatureSetDataPreparation:
    """Class for preparation techniques with grouping for experimental data. Needed for DANN AttentionModel.

    Methods:
        prepare_data: load and preprocess the data.
        save_data: save preprocessed data.
        prepare_dataloaders: create dataloaders from preprocessed data.
    """

    COMPLETE_GROUP_ID = GroupID(columns_to_detectors_masked(COLUMN_DETECTOR.keys()))

    def __init__(self, base_dir = PROCESSED_DIR, complete_only = False, undersample = UNDERSAMPLE) -> None:
        self.undersample = undersample
        self._scaling_params = pd.DataFrame(columns=["column", "mean", "std"])
        self._columns_for_training = None
        self.save_dir = f"{base_dir}/experiment/run{RUN}"
        self.complete_only = complete_only
        self.grouped_it = {}

    def prepare_data(self, input_path: str = EXPERIMENTAL_INPUT_PATH) -> None:
        """prepare_data loads and preprocesses data.
        Data is loaded from the path provided in `pdi.data.constants`
        """
        data = self._load_input_data(input_path)
        data = data.loc[data["fTPCSignal"] < TPC_CUT, :]
        data = self._make_missing_null(data)
        data = self._normalize_data(data)
        group_dict = self._group_data(data)
        for gid, group_data in group_dict.items():
            group_data = self._remove_unnecessary_columns(group_data)
            self.grouped_it[gid] = self._do_process_data(group_data)

    def _load_input_data(self, input_path: str = INPUT_PATH):
        self.csv_name: str = os.path.splitext(os.path.basename(input_path))[0]
        return pd.read_csv(input_path, sep=CSV_DELIMITER, index_col=0)

    def _make_missing_null(self, data):
        for column, val in MISSING_VALUES.items():
            data.loc[data[column] == val, column] = np.NaN

        if RUN == 2:
            print("Run 2 format")

        if RUN == 3:
            data.loc[data["fTRDPattern"].isnull(), ["fTRDSignal", "fTRDPattern"]] = np.NaN
            data.loc[data["fBeta"].isnull(), ["fTOFSignal", "fBeta"]] = np.NaN
            print("Run 3 format")
        return data

    def _normalize_data(self, data):
        for column in data.columns:
            if column in DO_NOT_SCALE or column in NSIGMA_COLUMNS:
                continue
            mean = np.mean(data[column])
            std = np.std(data[column])
            if std == 0:
                mean = 0
                std = 1
            self._scaling_params = pd.concat(
                [
                    self._scaling_params,
                    pd.DataFrame(
                        {
                            "column": column,
                            "mean": mean,
                            "std": std,
                        },
                        index=[0],
                    ),
                ],
                ignore_index=True,
            )
            data[column] = (data[column] - mean) / std
        return data

    def _remove_unnecessary_columns(self, data):
        if len(data.columns) == N_COLUMNS_BIG - 2:  # -2 because of TARGET_COLUMN and IsPhysicalPrimary
            data.drop(columns=DROP_COLUMNS_BIG, inplace=True)
        elif len(data.columns) != N_COLUMNS_ML:
            raise ValueError(
                "The input table has invalid number of columns. Was the PID ML producer updated?"
            )

        return data

    def _do_process_data(self, data):
        if len(data.columns) == N_COLUMNS_NSIGMAS - 2:  # -2 because of TARGET_COLUMN and IsPhysicalPrimary
            data.drop(columns=NSIGMA_COLUMNS, inplace=True)

        data.drop(columns=COLUMNS_DROPPED_FOR_TESTS, inplace=True)

        columns_for_training = pd.Series(data.columns.tolist())
        columns_for_training = columns_for_training[~columns_for_training.isin(NSIGMA_COLUMNS)]
        self._columns_for_training = columns_for_training

        return data

    def save_data(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

        with gzip.open(
            f"{self.save_dir}/input_data.pkl", "wb"
        ) as file:
            pickle.dump(self.grouped_it, file)

        self._scaling_params.to_json(
            f"{self.save_dir}/scaling_params.json",
            index=False,
            orient="split",
        )

        with open(f"{self.save_dir}/columns_for_training.json", "w+", encoding="UTF-8") as f:
            f.write(
                json.dumps(
                    {"columns_for_training": self._columns_for_training.tolist()}
                )
            )

        with open(f"{self.save_dir}/csv_name.txt", "w", encoding="utf-8") as file:
            file.write(self.csv_name)

    
    def _load_preprocessed_data(self):
        with gzip.open(
            f"{self.save_dir}/input_data.pkl", "rb"
        ) as file:
            self.grouped_it = pickle.load(file)
        with open(f"{self.save_dir}/csv_name.txt", "r", encoding="utf-8") as file:
            self.csv_name = file.read().strip()

    def _try_load_preprocessed_data(self):
        if not self.grouped_it:
            self._load_preprocessed_data()

    def prepare_dataloaders(
        self, batch_size: int, num_workers: int, _
    ) -> Iterable[DatasetItem]:
        """prepare_dataloaders creates dataloaders from preprocessed data.

        Args:
            batch_size (int): batch size
            num_workers (int): number of worker processed used in loading data.
                See `torch.utils.data.DataLoader` for more info.

        Returns:
            Iterable[DatasetItem] dataloader with preprocessed data.
        """
        self._try_load_preprocessed_data()

        dataloaders = {
            gid: DataLoader(
                ExperimentDataset(torch.tensor(self.grouped_it[gid].values, dtype=torch.float32)),
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )
            for gid in self.grouped_it.keys()
        }

        return CombinedDataLoader(
            True,
            *dataloaders.values(),
            undersample=self.undersample
        )

    def load_columns(self) -> list[str]:
        cols_for_training_path = self.save_dir

        with open(os.path.join(cols_for_training_path, "columns_for_training.json"), encoding="UTF-8") as f:
            data = json.load(f)

        cols = data["columns_for_training"]
        return cols
    
    def get_group_ids(self) -> list[GroupID]:
        """get_group_ids returns the ids of groups in the dataset

        Returns:
            list[GroupID]: list of group ids
        """
        self._try_load_preprocessed_data()
        return list(self.grouped_it.keys())

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

        return groups



class DANNGroupedDataPreparationWrapper:
    """Base data preparation class for DANN classifier with grouping
    
    Methods are the same as in GroupedDataPreparation, but in the constuctor two other grouped preparation classes are passed.
    Their methods are sequentially called to prepare the data.
    """
    def __init__(self, source: GroupedDataPreparation, target: ExperimentFeatureSetDataPreparation):
        self.source = source
        self.target = target
    
    def pos_weight(self, target: int) -> float:
        """pos_weight returns the ratio between the negative and positive samples in the train split, in all groups."""
        return self.source.pos_weight(target)
    
    def prepare_data(self, source_input_path: str = INPUT_PATH, target_input_path: str = EXPERIMENTAL_INPUT_PATH) -> None:
        """prepare_data loads and preprocesses data.
        Data is loaded from the path provided in `pdi.data.constants`
        """
        self.source.prepare_data(source_input_path)
        self.target.prepare_data(target_input_path)

    def save_data(self) -> None:
        """save_data saves preprocessed data, as well as scaling parameters, to disk.
        Save location is given in the class variable `save_dir`
        """
        self.source.save_data()
        self.target.save_data()
    
    def prepare_dataloaders(
        self,
        batch_size: int,
        num_workers: int,
        splits: Optional[list[Split]] = None,
        groupId: GroupID = None
    ):
        """prepare separate dataloaders and return them in a dictionary with keys 'source' and 'target'."""
        if splits is None:
            splits = list(Split)

        source_dataloaders = self.source.prepare_dataloaders(batch_size, num_workers, splits, groupId)
        target_dataloaders = self.target.prepare_dataloaders(batch_size, num_workers, splits)

        return {
            "source": source_dataloaders,
            "target": target_dataloaders,
        }
    
    def get_group_ids(self) -> list[GroupID]:
        """get_group_ids returns the ids of groups in the dataset

        Returns:
            list[GroupID]: list of group ids
        """
        return self.source.get_group_ids()
