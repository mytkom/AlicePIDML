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
import os
import pickle
from itertools import chain
from random import Random
from typing import Generic, Iterable, Iterator, TypeVar

import numpy as np
import pandas as pd
import torch
from pdi.data.constants import (COLUMNS_TO_SCALE, CSV_DELIMITER, DROP_COLUMNS,
                                GROUP_ID_KEY, INPUT_PATH, MISSING_VALUES, SEED,
                                TARGET_COLUMN, TEST_SIZE, TRAIN_SIZE)
from pdi.data.types import Additional, DatasetItem, GroupID, InputTarget, Split
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

T = TypeVar("T")


class CombinedDataLoader(Generic[T]):
    """CombinedDataLoader combines multiple dataloaders, shuffling their returned batches.
    """

    def __init__(self, shuffle: bool, *dataloaders: DataLoader[T]):
        """__init__

        Args:
            shuffle (bool): whether to change item order with each iteration.
            *dataloaders (DataLoader): a list of dataloaders to combine.
        """
        self.shuffle = shuffle
        self.dataloaders = dataloaders
        self.rng = Random(SEED)

    def _reset_seed(self):
        self.rng = Random(SEED)

    def __iter__(self) -> Iterator[T]:
        if not self.shuffle:
            self._reset_seed()

        iters = [iter(d) for d in self.dataloaders]

        while iters:
            it = self.rng.choice(iters)
            try:
                yield next(it)
            except StopIteration:
                iters.remove(it)

    def __len__(self) -> int:
        return sum([len(d) for d in self.dataloaders])


class DictDataset(Dataset[DatasetItem]):
    """DictDataset is a mapping dataset containing items that consist of an input tensor, a target tensor, and a dict of additional tensors.

    Base Class:
        Dataset
    """

    def __init__(self, input_tensors: Tensor, target_tensors: Tensor,
                 **additional_tensors: Tensor):
        """__init__

        Args:
            input_tensors (Tensor): tensor containing all inputtensors
            target_tensors (Tensor): tensor containing all target tensors
            **additional_tensors (Tensor): dict of tensors containing additional information 
        """
        self.input_tensors = input_tensors
        self.target_tensors = target_tensors
        self.dict_tensors = additional_tensors

    def __len__(self) -> int:
        return self.target_tensors.shape[0]

    def __getitem__(self, index: int) -> DatasetItem:
        return (
            self.input_tensors[index],
            self.target_tensors[index],
            {key: val[index]
             for key, val in self.dict_tensors.items()},
        )


class DataPreparation:
    """ Base class for preparation techniques without grouping.

    Methods:
        pos_weight: return the ratio between the negative and positive samples in the train split.
        prepare_data: load and preprocess the data.
        save_data: save preprocessed data.
        prepare_dataloaders: create dataloaders from preprocessed data.
    """
    save_dir: str

    def __init__(self) -> None:
        self._scaling_params = pd.DataFrame(columns=["column", "mean", "std"])
        self._input_target = {}
        self._additional = {}

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
        pos_weight: float = (np.size(binary_targets) - np.sum(binary_targets)
                             ) / (np.sum(binary_targets) + np.finfo(float).eps)
        return pos_weight

    def prepare_data(self) -> None:
        """prepare_data loads and preprocesses data.
        Data is loaded from the path provided in `pdi.data.constants`
        """
        data = self._load_input_data()
        data = self._delete_unique_targets(data)
        data = self._make_missing_null(data)
        data = self._normalize_data(data)
        split_data = self._test_train_split(data)
        split_data_2 = self._input_target_split(split_data)
        self._input_target, self._additional = self._do_process_data(
            split_data_2)

    def _load_input_data(self):
        return pd.read_csv(INPUT_PATH, sep=CSV_DELIMITER, index_col=0)

    def _delete_unique_targets(self, data):
        THRESHOLD = 100
        target_counts = data[TARGET_COLUMN].value_counts()
        for target, count in target_counts.items():
            if count < THRESHOLD:
                data = data[data[TARGET_COLUMN] != target]
        return data

    def _make_missing_null(self, data):
        for column, val in MISSING_VALUES.items():
            data.loc[data[column] == val, column] = np.NaN
        return data

    def _normalize_data(self, data):
        for column in COLUMNS_TO_SCALE:
            self._scaling_params = pd.concat(
                [
                    self._scaling_params,
                    pd.DataFrame(
                        {
                            "column": column,
                            "mean": np.mean(data[column]),
                            "std": np.std(data[column]),
                        },
                        index=[0])
                ],
                ignore_index=True,
            )
            data[column] = (data[column] - np.mean(data[column])) / np.std(
                data[column])
        return data

    def _test_train_split(self, data):
        train_to_val_ratio = TRAIN_SIZE / (1 - TEST_SIZE)
        (data_not_test,
         test_data) = train_test_split(data,
                                       test_size=TEST_SIZE,
                                       random_state=SEED,
                                       stratify=data.loc[:, [TARGET_COLUMN]])
        (train_data, val_data) = train_test_split(
            data_not_test,
            train_size=train_to_val_ratio,
            random_state=SEED,
            stratify=data_not_test.loc[:, [TARGET_COLUMN]])
        return {
            Split.TRAIN: train_data,
            Split.VAL: val_data,
            Split.TEST: test_data
        }

    def _input_target_split(self, data):

        def do_split(data):
            targets = data.loc[:, [TARGET_COLUMN]]
            input_data = data.drop(columns=DROP_COLUMNS)
            return {InputTarget.INPUT: input_data, InputTarget.TARGET: targets}

        split_data = {}
        for key, val in data.items():
            split_data[key] = do_split(val)

        return split_data

    def _do_process_data(self, data):
        processed_data = {}
        additional_data = {}
        for key, val in data.items():
            processed_data[key], additional_data[
                key] = self._do_preprocess_split(val)
        return processed_data, additional_data

    def _do_preprocess_split(self, split):
        input = split[InputTarget.INPUT]
        targets = split[InputTarget.TARGET]
        return (
            {
                InputTarget.INPUT: input.values,
                InputTarget.TARGET: targets.values,
            },
            {
                column: input.loc[:, [column.name]].values
                for column in Additional
            },
        )

    def save_data(self) -> None:
        """save_data saves preprocessed data, as well as scaling parameters, to disk.
        Save location is given in the class variable `save_dir`
        """
        os.makedirs(self.save_dir, exist_ok=True)
        for split, it_data in self._input_target.items():
            with gzip.open(f"{self.save_dir}/input_target_{split.name}.pkl",
                           "wb") as file:
                pickle.dump(it_data, file)

        for split, add_data in self._additional.items():
            with gzip.open(f"{self.save_dir}/additional_{split.name}.pkl",
                           "wb") as file:
                pickle.dump(add_data, file)

        self._scaling_params.to_json(
            f"{self.save_dir}/scaling_params.json",
            index=False,
            orient="split",
        )

    def _load_preprocessed_data(self, splits):
        for split in splits:
            with gzip.open(f"{self.save_dir}/input_target_{split.name}.pkl",
                           "rb") as file:
                self._input_target[split] = pickle.load(file)
            with gzip.open(f"{self.save_dir}/additional_{split.name}.pkl",
                           "rb") as file:
                self._additional[split] = pickle.load(file)

    def _try_load_preprocessed_data(self, splits):
        if any(key not in self._input_target or key not in self._additional
               for key in splits):
            self._load_preprocessed_data(splits)

    def prepare_dataloaders(
        self,
        batch_size: int,
        num_workers: int,
        splits: list[Split] = list(Split)
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
        self._try_load_preprocessed_data(splits)

        def create_dataset(input_target, additional):
            return DictDataset(
                torch.tensor(input_target[InputTarget.INPUT],
                             dtype=torch.float32),
                torch.tensor(input_target[InputTarget.TARGET],
                             dtype=torch.float32),
                **{
                    column.name: torch.tensor(val, dtype=torch.float32)
                    for column, val in additional.items()
                },
            )

        datasets = {
            split: create_dataset(self._input_target[split],
                                  self._additional[split])
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

        return (*dataloaders.values(), )


class GroupedDataPreparation(DataPreparation):
    """ Base class for preparation techniques with grouping.

    Methods:
        pos_weight: return the ratio between the negative and positive samples in the train split, in all groups.
        prepare_data: load and preprocess the data.
        save_data: save preprocessed data.
        prepare_dataloaders: create dataloaders from preprocessed data.

    """
    COMPLETE_GROUP_ID: GroupID

    def __init__(self, complete_only: bool):
        super().__init__()
        self.complete_only = complete_only
        self.grouped_it = {}
        self.grouped_add = {}

    def pos_weight(self, target: int) -> float:
        self._try_load_preprocessed_data([Split.TRAIN])
        target_list = [
            group[Split.TRAIN][InputTarget.TARGET]
            for group in self.grouped_it.values()
        ]
        binary_targets = np.concatenate(
            [targets == target for targets in target_list])
        pos_weight: float = (np.size(binary_targets) -
                             np.sum(binary_targets)) / np.sum(binary_targets)
        return pos_weight

    def get_group_ids(self) -> list[GroupID]:
        """get_group_ids returns the ids of groups in the dataset

        Returns:
            list[GroupID]: list of group ids
        """
        self._try_load_preprocessed_data([Split.TRAIN])
        return list(self.grouped_it.keys())

    def prepare_data(self) -> None:
        data = self._load_input_data()
        data = self._delete_unique_targets(data)
        data = self._make_missing_null(data)
        data = self._normalize_data(data)
        group_dict = self._group_data(data)
        for gid, group_data in group_dict.items():
            split_data = self._test_train_split(group_data)
            split_data_2 = self._input_target_split(split_data)
            self.grouped_it[gid], self.grouped_add[
                gid] = self._do_process_data(split_data_2)

    def _group_data(self, data):
        return {}

    def save_data(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

        def save_dict(name, dict):
            transposed_dict = {split: {} for split in list(Split)}
            for gid, group_data in dict.items():
                for split, sd in group_data.items():
                    transposed_dict[split][gid] = sd

            for split, split_data in transposed_dict.items():
                with gzip.open(f"{self.save_dir}/{name}_{split.name}.pkl",
                               "wb") as file:
                    pickle.dump(split_data, file)

        save_dict("input_target", self.grouped_it)
        save_dict("additional", self.grouped_add)

        self._scaling_params.to_json(
            f"{self.save_dir}/scaling_params.json",
            index=False,
            orient="split",
        )

    def _load_preprocessed_data(self, splits):
        for split in splits:
            with gzip.open(f"{self.save_dir}/input_target_{split.name}.pkl",
                           "rb") as file:
                data = pickle.load(file)
                for gid, group_data in data.items():
                    self.grouped_it.setdefault(gid, {})
                    self.grouped_it[gid][split] = group_data

            with gzip.open(f"{self.save_dir}/additional_{split.name}.pkl",
                           "rb") as file:
                data = pickle.load(file)
                for gid, group_data in data.items():
                    self.grouped_add.setdefault(gid, {})
                    self.grouped_add[gid][split] = group_data

    def _try_load_preprocessed_data(self, splits):
        are_splits_loaded = [
            split in group_dict for group_dict in chain(
                self.grouped_it.values(), self.grouped_add.values())
            for split in splits
        ]

        if not all(are_splits_loaded) or len(are_splits_loaded) == 0:
            self._load_preprocessed_data(splits)

    def prepare_dataloaders(
        self,
        batch_size: int,
        num_workers: int,
        splits: list[Split] = list(Split)
    ) -> tuple[Iterable[DatasetItem], ...]:
        self._try_load_preprocessed_data(splits)

        def create_datasets(input_target, additional, gid, splits):
            return {
                split: DictDataset(
                    torch.tensor(input_target[split][InputTarget.INPUT],
                                 dtype=torch.float32),
                    torch.tensor(input_target[split][InputTarget.TARGET],
                                 dtype=torch.float32),
                    **{
                        GROUP_ID_KEY:
                        torch.full(
                            input_target[split][InputTarget.TARGET].shape,
                            gid),
                        **{
                            column.name: torch.tensor(val, dtype=torch.float32)
                            for column, val in additional[split].items()
                        }
                    },
                )
                for split in splits
            }

        datasets = {
            gid: create_datasets(self.grouped_it[gid], self.grouped_add[gid],
                                 gid, splits)
            for gid in self.grouped_it.keys()
        }

        if self.complete_only:
            dataloaders = {
                split: DataLoader(datasets[self.COMPLETE_GROUP_ID][split],
                                  batch_size,
                                  shuffle=(split == Split.TRAIN),
                                  num_workers=num_workers)
                for split in splits
            }
        else:
            separate_dataloaders = {
                gid: {
                    split: DataLoader(dataset,
                                      batch_size,
                                      shuffle=(split == Split.TRAIN),
                                      num_workers=num_workers)
                    for split, dataset in dataset_splits.items()
                }
                for gid, dataset_splits in datasets.items()
            }

            dataloaders = {
                split: CombinedDataLoader(
                    (split == Split.TRAIN),
                    *[d[split] for d in separate_dataloaders.values()],
                )
                for split in splits
            }

        return (*dataloaders.values(), )
