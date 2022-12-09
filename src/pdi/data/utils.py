import gzip
import os
import pickle
from itertools import chain
from typing import Any, Generic, Iterable, Iterator, TypeVar, cast

import numpy as np
import pandas as pd
import torch
from numpy.random import default_rng
from pdi.data.constants import (COLUMNS_TO_SCALE, CSV_DELIMITER, DROP_COLUMNS,
                                GROUP_ID_KEY, INPUT_PATH, MISSING_VALUES, SEED,
                                TARGET_COLUMN, TEST_SIZE, TRAIN_SIZE)
from pdi.data.types import (AddDict, Additional, DatasetItem, DfOrArray,
                            GIDDict, GroupID, InputTarget, InputTargetDict,
                            Split, SplitDict)
from sklearn.model_selection import train_test_split  # type: ignore
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

T = TypeVar("T")


class CombinedDataLoader(Generic[T]):

    def __init__(self, shuffle: bool, *dataloaders: DataLoader[T]):
        self.shuffle = shuffle
        self.dataloaders = dataloaders
        self.rng = default_rng(SEED)

    def _reset_seed(self) -> None:
        self.rng = default_rng(SEED)

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

    def __init__(self, input_tensors: Tensor, target_tensors: Tensor,
                 **additional_tensors: Tensor):
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
    save_dir: str

    def __init__(self) -> None:
        self.scaling_params = pd.DataFrame()
        self.input_target: SplitDict[InputTargetDict[np.ndarray]] = {}
        self.additional: SplitDict[AddDict[np.ndarray]] = {}

    def pos_weight(self, target: int) -> float:
        self._try_load_preprocessed_data([Split.TRAIN])

        train_targets = self.input_target[Split.TRAIN][InputTarget.TARGET]
        binary_targets = train_targets == target
        pos_weight: float = (np.size(binary_targets) - np.sum(binary_targets)
                             ) / (np.sum(binary_targets) + np.finfo(float).eps)
        return pos_weight

    def prepare_data(self) -> None:
        data = self._load_input_data()
        data = self._make_missing_null(data)
        data = self._normalize_data(data)
        split_data = self._test_train_split(data)
        split_data_2 = self._input_target_split(split_data)
        processed_data, additional_data = self._do_process_data(split_data_2)
        self.input_target = self._transform_df_to_np(processed_data)
        self.additional = self._transform_df_to_np(additional_data)

    def _load_input_data(self) -> pd.DataFrame:
        return pd.read_csv(INPUT_PATH, sep=CSV_DELIMITER, index_col=0)

    def _make_missing_null(self, data: pd.DataFrame) -> pd.DataFrame:
        for column, val in MISSING_VALUES.items():
            data.loc[data[column] == val, column] = np.NaN
        return data

    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        for column in COLUMNS_TO_SCALE:
            self.scaling_params = self.scaling_params.append(
                {
                    "column": column,
                    "mean": np.mean(data[column]),
                    "std": np.std(data[column]),
                },
                ignore_index=True,
            )
            data[column] = (data[column] - np.mean(data[column])) / np.std(
                data[column])
        return data

    def _test_train_split(self, data: pd.DataFrame) -> SplitDict[pd.DataFrame]:
        train_to_val_ratio = TRAIN_SIZE / (1 - TEST_SIZE)
        (data_not_test, test_data) = train_test_split(data,
                                                      test_size=TEST_SIZE,
                                                      random_state=SEED)
        (train_data,
         val_data) = train_test_split(data_not_test,
                                      train_size=train_to_val_ratio,
                                      random_state=SEED)
        return {
            Split.TRAIN: train_data,
            Split.VAL: val_data,
            Split.TEST: test_data
        }

    def _input_target_split(
        self, data: SplitDict[pd.DataFrame]
    ) -> SplitDict[InputTargetDict[pd.DataFrame]]:

        def do_split(data: pd.DataFrame) -> InputTargetDict[pd.DataFrame]:
            targets = data.loc[:, [TARGET_COLUMN]]
            input_data = data.drop(columns=DROP_COLUMNS)
            return {InputTarget.INPUT: input_data, InputTarget.TARGET: targets}

        split_data = {}
        for key, val in data.items():
            split_data[key] = do_split(val)

        return split_data

    def _do_process_data(
        self, data: SplitDict[InputTargetDict[pd.DataFrame]]
    ) -> tuple[SplitDict[InputTargetDict[DfOrArray]],
               SplitDict[AddDict[DfOrArray]], ]:
        processed_data = {}
        additional_data = {}
        for key, val in data.items():
            processed_data[key], additional_data[
                key] = self._do_preprocess_split(val)
        return processed_data, additional_data

    def _do_preprocess_split(
        self, split: InputTargetDict[pd.DataFrame]
    ) -> tuple[InputTargetDict[DfOrArray], AddDict[DfOrArray]]:
        input = split[InputTarget.INPUT]
        targets = split[InputTarget.TARGET]
        return (
            {
                InputTarget.INPUT: input,
                InputTarget.TARGET: targets,
            },
            {column: input.loc[:, [column.name]]
             for column in Additional},
        )

    def _transform_df_to_np(
            self, data: SplitDict[dict[T, DfOrArray]]
    ) -> SplitDict[dict[T, np.ndarray]]:
        return {
            split: {
                key: val.values if isinstance(val, pd.DataFrame) else val
                for key, val in split_data.items()
            }
            for split, split_data in data.items()
        }

    def save_data(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        for split, it_data in self.input_target.items():
            with gzip.open(f"{self.save_dir}/input_target_{split.name}.pkl",
                           "wb") as file:
                pickle.dump(it_data, file)

        for split, add_data in self.additional.items():
            with gzip.open(f"{self.save_dir}/additional_{split.name}.pkl",
                           "wb") as file:
                pickle.dump(add_data, file)

        self.scaling_params.to_json(
            f"{self.save_dir}/scaling_params.json",
            index=False,
            orient="split",
        )

    def _load_preprocessed_data(self, splits: list[Split]) -> None:
        for split in splits:
            with gzip.open(f"{self.save_dir}/input_target_{split.name}.pkl",
                           "rb") as file:
                self.input_target[split] = pickle.load(file)
            with gzip.open(f"{self.save_dir}/additional_{split.name}.pkl",
                           "rb") as file:
                self.additional[split] = pickle.load(file)

    def _try_load_preprocessed_data(self, splits: list[Split]) -> None:
        if any(key not in self.input_target or key not in self.additional
               for key in splits):
            self._load_preprocessed_data(splits)

    def prepare_dataloaders(
        self,
        batch_size: int,
        num_workers: int,
        splits: list[Split] = list(Split)
    ) -> tuple[Iterable[DatasetItem], ...]:
        self._try_load_preprocessed_data(splits)

        def create_dataset(input_target: InputTargetDict[np.ndarray],
                           additional: AddDict[np.ndarray]) -> DictDataset:
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
            split: create_dataset(self.input_target[split],
                                  self.additional[split])
            for split in splits
        }

        dataloaders: SplitDict[Iterable[DatasetItem]]

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
    COMPLETE_GROUP_ID: GroupID

    def __init__(self, complete_only: bool):
        super().__init__()
        self.complete_only = complete_only
        self.grouped_it: GIDDict[SplitDict[InputTargetDict[np.ndarray]]] = {}
        self.grouped_add: GIDDict[SplitDict[AddDict[np.ndarray]]] = {}

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

    def get_group_ids(self) -> list[int]:
        self._try_load_preprocessed_data([Split.TRAIN])
        return list(self.grouped_it.keys())

    def prepare_data(self) -> None:
        data = self._load_input_data()
        data = self._make_missing_null(data)
        data = self._normalize_data(data)
        group_dict = self._group_data(data)
        for gid, group_data in group_dict.items():
            split_data = self._test_train_split(group_data)
            split_data_2 = self._input_target_split(split_data)
            processed_data, additional_data = self._do_process_data(
                split_data_2)
            self.grouped_it[gid] = self._transform_df_to_np(processed_data)
            self.grouped_add[gid] = self._transform_df_to_np(additional_data)

    def _group_data(self, data: pd.DataFrame) -> GIDDict[pd.DataFrame]:
        return {}

    def save_data(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

        def save_dict(name: str,
                      dict: GIDDict[SplitDict[dict[T, np.ndarray]]]) -> None:
            transposed_dict: SplitDict[GIDDict[dict[T, np.ndarray]]] = {
                split: {}
                for split in list(Split)
            }
            for gid, group_data in dict.items():
                for split, sd in group_data.items():
                    transposed_dict[split][gid] = sd

            for split, split_data in transposed_dict.items():
                with gzip.open(f"{self.save_dir}/{name}_{split.name}.pkl",
                               "wb") as file:
                    pickle.dump(split_data, file)

        save_dict("input_target", self.grouped_it)
        save_dict("additional", self.grouped_add)

        self.scaling_params.to_json(
            f"{self.save_dir}/scaling_params.json",
            index=False,
            orient="split",
        )

    def _load_preprocessed_data(self, splits: list[Split]) -> None:
        for split in splits:
            with gzip.open(f"{self.save_dir}/input_target_{split.name}.pkl",
                           "rb") as file:
                data = pickle.load(file)
                for gid, group_data in data.items():
                    self.grouped_it[gid][split] = group_data

            with gzip.open(f"{self.save_dir}/additional_{split.name}.pkl",
                           "rb") as file:
                data = pickle.load(file)
                for gid, group_data in data.items():
                    self.grouped_add[gid][split] = group_data

    def _try_load_preprocessed_data(self, splits: list[Split]) -> None:
        if any([
                split not in cast(SplitDict[Any], group_dict)
                for group_dict in chain(self.grouped_it.values(),
                                        self.grouped_add.values())
                for split in splits
        ]):
            self._load_preprocessed_data(splits)

    def prepare_dataloaders(
        self,
        batch_size: int,
        num_workers: int,
        splits: list[Split] = list(Split)
    ) -> tuple[Iterable[DatasetItem], ...]:
        self._try_load_preprocessed_data(splits)

        def create_datasets(data: SplitDict[InputTargetDict[np.ndarray]],
                            gid: int) -> SplitDict[DictDataset]:
            return {
                split: DictDataset(
                    torch.tensor(split_data[InputTarget.INPUT],
                                 dtype=torch.float32),
                    torch.tensor(split_data[InputTarget.TARGET],
                                 dtype=torch.float32),
                    **{
                        GROUP_ID_KEY:
                        torch.full(split_data[InputTarget.TARGET].shape, gid),
                        **{
                            column.name: torch.tensor(val, dtype=torch.float32)
                            for column, val in self.grouped_add[gid][split].items(
                            )
                        },
                    },
                )
                for split, split_data in data.items()
            }

        datasets = {
            gid: create_datasets(data, gid)
            for gid, data in self.grouped_it.items()
        }

        dataloaders: SplitDict[Iterable[DatasetItem]]

        if self.complete_only:
            dataloaders = {
                split: DataLoader(dataset,
                                  batch_size,
                                  shuffle=(split == Split.TRAIN),
                                  num_workers=num_workers)
                for split, dataset in datasets[self.COMPLETE_GROUP_ID].items()
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
                for split in [Split.TRAIN, Split.VAL, Split.TEST]
            }

        return (*dataloaders.values(), )
