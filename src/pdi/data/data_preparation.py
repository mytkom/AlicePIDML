import gzip
import json
import os
import pickle
import uproot3
import numpy as np
import pandas as pd
import torch
import hashlib
from pathlib import Path
from random import Random
from typing import Callable, Generic, Iterable, Iterator, Tuple, TypeVar, Optional, List, MutableMapping
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from pdi.data.group_id_helpers import binary_array_to_group_id
from pdi.data.constants import (
    PROCESSED_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
    TRAIN_SIZE,
    COLUMNS_FOR_TRAINING,
    NSIGMA_COLUMNS,
)
from pdi.data.types import GroupID, InputTarget, Split
from pdi.data.config import DataConfig

def calculate_checksum(filenames: list[str]) -> str:
    hash = hashlib.md5()
    for fn in filenames:
        try:
            hash.update(Path(fn).read_bytes())
        except IsADirectoryError:
            pass
    return hash.hexdigest()

# Target code (ground truth) is available in Monte Carlo (simulated) data
# Tuple[standardized inputs tensor, target codes tensor, GroupID, undstandardized data]
MCBatchItem = Tuple[Tensor, Tensor, GroupID, MutableMapping[str, Tensor]]

# Target code (ground truth) is not available in Experimental (real) data
# Tuple[standardized inputs tensor, GroupID, undstandardized data]
ExpBatchItem = Tuple[Tensor, GroupID, MutableMapping[str, Tensor]]

T = TypeVar("T")

def is_experimental_data(table_name):
    if "mc" in table_name:
        return False
    return True

def is_extended_data(table_name):
    if "ml" in table_name:
        return False
    return True

def load_root_data(input_files: List[str]) -> Tuple[pd.DataFrame | pd.Series, bool, bool]:
    """
        Load given list of root files into pandas DataFrame. This function automatically detect,
        which out of possible tables is saved in files:
            - O2pidtracksmcml - simulated data, only columns needed for ML training (feature space and labels),
            - O2pidtracksmc - simulated data, more columns---it contains nSigma values,
            - O2pidtracksdataml - experimental data, only columns needed for training (no labels available)
            - O2pidtracksdata - experimental data, more columns---it contains nSigma values
    """
    TABLE_NAMES=["O2pidtracksmcml", "O2pidtracksmc", "O2pidtracksdataml", "O2pidtracksdata"]
    table_name: str | None = None
    dataframes = []
    for input_file in input_files:
        file = uproot3.open(input_file)
        for dirname in file:
            dirname = dirname.decode("utf-8")
            pure_dirname = dirname.split(";")[0]

            if pure_dirname.startswith("DF_"):
                if not table_name:
                    for tn in TABLE_NAMES:
                        try:
                            file[f"%s/{tn}" % (dirname)].pandas.df()
                            table_name = tn
                        except KeyError:
                            continue
                    if not table_name:
                        raise KeyError(f"Table name in ROOT file must be in {TABLE_NAMES}")

                tree_data = file[f"%s/{table_name}" % (dirname)].pandas.df()
                dataframes.append(tree_data)

    return pd.concat(dataframes, ignore_index=True), is_experimental_data(table_name), is_extended_data(table_name)

PreparedData = dict[Split, dict[GroupID, dict[InputTarget, pd.DataFrame]]]

# TODO: describe this class after implementation
class GeneralDataPreparation:
    """
    """

    COLUMNS_TO_SCALE_RUN_3 = [
        "fTPCSignal", "fTOFSignal",
        "fBeta", "fX", "fY", "fZ", "fAlpha",
        "fTPCNClsShared", "fDcaXY", "fDcaZ",
    ]

    # TRDSignal scaling in run2
    COLUMNS_TO_SCALE_RUN_2 = COLUMNS_TO_SCALE_RUN_3 + ["fTRDSignal"]


    MISSING_VALUE_INDICATORS = {
        "fBeta": -999,
        "fTOFSignal": -999,
        "fTRDPattern": 0,
        "fTRDSignal": 0
    }

    def __init__(self, config: DataConfig, input_paths: List[str], base_dir = PROCESSED_DIR) -> None:
        if len(input_paths) == 0:
            raise KeyError("You must specify at least one input_path with data!")

        # Calculate checsum for input_paths' files, so caching results will be reliable and unique for this set of files
        self._inputs_checksum = calculate_checksum(input_paths)
        self.save_dir: str = f"{base_dir}/{self._inputs_checksum}"
        self._scaling_params: pd.DataFrame = pd.DataFrame(columns=["column", "mean", "std"])
        self._input_paths: List[str] = input_paths
        self._cfg: DataConfig = config
        self._columns_for_training: List[str] = []
        self._columns_to_standardize: List[str] = self.COLUMNS_TO_SCALE_RUN_3 if config.is_run_3 else self.COLUMNS_TO_SCALE_RUN_2
        self._prepared_data: dict[Split, dict[GroupID, dict[InputTarget, pd.DataFrame]]] = {}

    def get_prepared_data(self) -> PreparedData:
        self._load_or_prepare_data([Split.TRAIN, Split.VAL, Split.TEST])

        return self._prepared_data

    # Data is stored in _prepared_data
    # Structure:
    # (Train/Validation/Test) Split: {
    #   GroupID: { Input, Target, Unstandardized (input columns + nSigma columns as dict) }
    #  }
    def prepare_data(self) -> None:
        """
            Preprocess, standardize and split data from root source file. It can be called explicitly
            to force data preparation (in case of cached data in bad state).
        """
        # It expects data in ROOT format, which is extracted using O2Physics' task PIDMLProducer.
        # This method can distinguish 4 data formats (simulated or experimental + basic or extended)
        data = self._load_data()

        # When after splits some particle specie track is unique in its group,
        # it raises an error in some point, TODO: do it cleaner way
        data = self._delete_unique_targets(data)

        # When there is no detector value in data some special
        # values indicating missing values are returned e.g. -999.0.
        # This method sets such values to NaNs
        data = self._make_missing_null(data)

        # TOF is not reliable for transverse momentum (fPt) lower than PT_CUT,
        # TPC was returning outliers (10M signal value), which were bad for
        # model performance - we filter now manually tracks with fTPCSignal > TPC_CUT
        data = self._perform_cuts(data)

        # Split dataset into Train/Validation/Test
        #   split ratios are specified in DataConfig self._cfg
        test_train_split: dict[Split, pd.DataFrame] = self._test_train_split(data)

        # Standardization parameters (mean, std) based on train split
        #   results are saved in self._scaling_params
        self._calc_scaling_params(test_train_split[Split.TRAIN])

        self._prepared_data = {}
        for split, split_data in test_train_split.items():
            split: Split
            split_data: pd.DataFrame

            self._prepared_data[split] = {}

            grouped_split_data_dict = self._group_data(split_data)
            for gid, group_data in grouped_split_data_dict.items():
                gid: GroupID
                group_data: pd.DataFrame

                # Split data into ML inputs and targets and additional unstandardized dict with nSigma columns if available
                self._prepared_data[split][gid] = self._input_target_unstandardized_split(group_data)

                # Standardize inputs using previously calculated params
                self._prepared_data[split][gid][InputTarget.INPUT] = self._standardize_data(self._prepared_data[split][gid][InputTarget.INPUT])

        # Cache prepared data
        self._save_data()

    def create_dataloaders(
        self, batch_size: int, num_workers: int, undersample: bool, seed: int, splits: Optional[list[Split]] = None
    ) -> tuple[Iterable[MCBatchItem] | Iterable[ExpBatchItem], ...]:
        """prepare_dataloaders creates dataloaders from preprocessed data.

        Args:
            batch_size (int): batch size
            num_workers (int): number of worker processed used in loading data.
                See `torch.utils.data.DataLoader` for more info.
            splits (list[Split], optional): list of splits to create dataloaders for.
                Defaults to [Split.TRAIN, Split.VAL, Split.TEST].

        Returns:
            tuple[Iterable[MCBatchItem], ...]: tuple of dataloaders, one for each split.
        """
        if splits is None:
            splits = list(Split)

        self._load_or_prepare_data(splits)

        def create_dataset(input_target_unstandardized, gid):
            if self._is_experimental:
                return ExpDataset(
                    torch.tensor(input_target_unstandardized[InputTarget.INPUT].values, dtype=torch.float32),
                    gid,
                    **{
                        column: torch.tensor(val.values, dtype=torch.float32)
                        for column, val in input_target_unstandardized[InputTarget.UNSTANDARDIZED].items()
                    },
                )

            return MCDataset(
                torch.tensor(input_target_unstandardized[InputTarget.INPUT].values, dtype=torch.float32),
                torch.tensor(input_target_unstandardized[InputTarget.TARGET].values, dtype=torch.float32),
                gid,
                **{
                    column: torch.tensor(val.values, dtype=torch.float32)
                    for column, val in input_target_unstandardized[InputTarget.UNSTANDARDIZED].items()
                },
            )

        dataloaders: dict[Split, CombinedDataLoader] = {}
        for split, grouped_data in self._prepared_data.items():
            datasets = {
                gid: create_dataset(grouped_data[gid], gid)
                for gid in grouped_data.keys()
            }
            dataloaders[split] = CombinedDataLoader(
                (split == Split.TRAIN),
                undersample,
                seed,
                *[
                    DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=(split == Split.TRAIN),
                    num_workers=num_workers,
                    )
                    for dataset in datasets.values()
                ],
            )

        return (*dataloaders.values(),)

    def transform_prepared_data(self, transform: Callable[[PreparedData], None]):
        self._load_or_prepare_data([Split.TRAIN, Split.VAL, Split.TEST])

        transform(self._prepared_data)

    def pos_weight(self, target: int) -> float:
        if self._is_experimental:
            raise RuntimeError("pos_weight is infeasible for experimental dataset!")

        self._load_or_prepare_data([Split.TRAIN])

        target_list = [
            group[InputTarget.TARGET] for group in self._prepared_data[Split.TRAIN].values()
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
        self._load_or_prepare_data([Split.TRAIN])

        return list(self._prepared_data[Split.TRAIN].keys())

    def _save_data(self) -> None:
        """save_data saves preprocessed data, as well as scaling parameters, to disk.
        Save location is given in the class variable `save_dir`
        """

        os.makedirs(self.save_dir, exist_ok=True)
        for split, it_data in self._prepared_data.items():
            with gzip.open(
                f"{self.save_dir}/prepared_data_{split.name}.pkl", "wb"
            ) as file:
                pickle.dump(it_data, file)

        self._scaling_params.to_json(
            f"{self.save_dir}/scaling_params.json",
            index=False,
            orient="split",
        )

        with open(f"{self.save_dir}/dataset_metadata.json", "w+", encoding="UTF-8") as f:
            f.write(
                json.dumps(
                    {
                        "is_experimental": self._is_experimental,
                        "is_extended": self._is_extended,
                        "input_paths": self._input_paths,
                    }
                )
            )

        with open(f"{self.save_dir}/columns_for_training.json", "w+", encoding="UTF-8") as f:
            f.write(
                json.dumps(
                    {"columns_for_training": COLUMNS_FOR_TRAINING}
                )
            )

    def _load_data(self) -> pd.DataFrame | pd.Series:
        data, self._is_experimental, self._is_extended = load_root_data(self._input_paths)

        if self._is_experimental:
            self._log("Experimental data detected!")
        else:
            self._log("Simulated (Monte Carlo) data detected!")

        if self._is_extended:
            self._log("Extended dataset detected (with nSigma columns)!")
        else:
            self._log("Basic dataset detected (without nSigma columns---only ML inputs and targets)!")

        return data

    # TODO: inspect, why it is needed---what code do not work if such targets exists
    def _delete_unique_targets(self, data):
        THRESHOLD = 400
        target_counts = data[TARGET_COLUMN].value_counts()
        for target, count in target_counts.items():
            if count < THRESHOLD:
                data = data[data[TARGET_COLUMN] != target]
        return data

    def _make_missing_null(self, data):
        for column, val in self.MISSING_VALUE_INDICATORS.items():
            data.loc[data[column] == val, column] = np.NaN

        # TODO: check why is it needed to distinguish between Run2 and Run3 here?
        if self._cfg.is_run_3:
            # TRDPattern is uint8, so cannot use NaN in producer -> need to preprocess it here
            data["fTRDPattern"].mask(np.isclose(data["fTRDPattern"], 0), inplace=True)
            data.loc[data["fTRDPattern"].isnull(), ["fTRDSignal", "fTRDPattern"]] = np.NaN
            data.loc[data["fBeta"].isnull(), ["fTOFSignal", "fBeta"]] = np.NaN
            self._log("Run 3 missing values applied!")
        else:
            self._log("Run 2 missing values applied!")

        return data

    def _perform_cuts(self, data):
        # TCPSignal must be positive
        data = data[data["fTPCSignal"] > 0]

        # TPCSignal sometimes gives huge incorrect values, which negatively impacts standardization
        # maybe outlier detection methods can handle it nicely (for now I leave it be)
        TPC_CUT = 10000
        data = data.loc[data["fTPCSignal"] < TPC_CUT, :]

        # TOF is said to be incorrect even if present if Pt is lower than 0.5 GeV/C
        PT_CUT = 0.5
        data.loc[data["fPt"] < PT_CUT, ["fTOFBeta", "fTOFSignal"]] = np.NaN

        return data

    def _test_train_split(self, data) -> dict[Split, pd.DataFrame]:
        train_to_val_ratio = TRAIN_SIZE / (1 - TEST_SIZE)

        (data_not_test, test_data) = train_test_split(
            data,
            test_size=TEST_SIZE,
            random_state=self._cfg.split_seed,
            stratify=data.loc[:, [TARGET_COLUMN]],
        )
        data_not_test = pd.DataFrame(data_not_test)

        (train_data, val_data) = train_test_split(
            data_not_test,
            train_size=train_to_val_ratio,
            random_state=self._cfg.split_seed,
            stratify=data_not_test.loc[:, [TARGET_COLUMN]],
        )

        self._log(f"Dataset has been splitted in the following ratios:")
        self._log(f"\tTrain {TRAIN_SIZE}, Validation {1 - TEST_SIZE - TRAIN_SIZE}, Test {TEST_SIZE}")

        return {
            Split.TRAIN: pd.DataFrame(train_data),
            Split.VAL: pd.DataFrame(val_data),
            Split.TEST: pd.DataFrame(test_data)
        }

    def _calc_scaling_params(self, train_split: pd.DataFrame):
        for column in train_split.columns:
            if column not in self._columns_to_standardize:
                continue
            mean = np.mean(train_split[column])
            std = np.std(train_split[column])
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

        self._log(f"Scaling (standardization) params has been calculated on training split, results:\n{self._scaling_params}")

    def _group_data(self, data):
        missing = ~data.isnull()

        groups = missing.groupby(list(COLUMNS_FOR_TRAINING), dropna=False).groups
        grouped_data = {}
        for binary_array, indices in groups.items():
            gid: GroupID = binary_array_to_group_id(np.array(list(binary_array)))
            grouped_data[gid] = data.loc[indices]

        return grouped_data

    def _input_target_unstandardized_split(self, data):
        input_data = data.loc[:, COLUMNS_FOR_TRAINING]
        if self._is_extended:
            # Add NSigma columns to unstandardized split, if they are available (extended dataset)
            unstandardized = data.loc[:, COLUMNS_FOR_TRAINING + NSIGMA_COLUMNS]
        else:
            unstandardized = data.loc[:, COLUMNS_FOR_TRAINING]

        if not self._is_experimental:
            targets = data.loc[:, [TARGET_COLUMN]]
            return {InputTarget.INPUT: input_data, InputTarget.TARGET: targets, InputTarget.UNSTANDARDIZED: unstandardized}

        return {InputTarget.INPUT: input_data, InputTarget.UNSTANDARDIZED: unstandardized}

    def _standardize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        for _, row in self._scaling_params.iterrows():
            col = row["column"]
            data[col] = (data[col] - row["mean"]) / row["std"]

        return data

    def _load_preprocessed_data(self, splits):
        for split in splits:
            with gzip.open(
                f"{self.save_dir}/prepared_data_{split.name}.pkl", "rb"
            ) as file:
                self._prepared_data[split] = pickle.load(file)

        with open(f"{self.save_dir}/dataset_metadata.json", "r", encoding="UTF-8") as f:
            metadata = json.load(f)
            self._is_experimental = metadata["is_experimental"]
            self._is_extended = metadata["is_extended"]

    def _try_load_preprocessed_data(self, splits) -> bool:
        if any(
            key not in self._prepared_data
            for key in splits
        ):
            try:
                self._load_preprocessed_data(splits)
            except:
                return False

        return True

    def _load_or_prepare_data(self, splits):
        if not self._try_load_preprocessed_data(splits):
            self.prepare_data()

    def _log(self, message: str):
        print(f"[DataPreparation] {message}")


class CombinedDataLoader(Generic[T]):
    """CombinedDataLoader combines multiple dataloaders, shuffling their returned batches."""

    def __init__(self, shuffle: bool, undersample: bool, seed: int, *dataloaders: DataLoader[T]):
        """__init__

        Args:
            shuffle (bool): whether to change item order with each iteration.
            *dataloaders (DataLoader): a list of dataloaders to combine.
        """
        self.shuffle = shuffle
        self.dataloaders = dataloaders
        self.rng = Random(seed)
        self.seed = seed

    def _reset_seed(self):
        self.rng = Random(self.seed)

    # TODO: test undersampling implementation (it was not tested)
    def __iter__(self) -> Iterator[T]:
        if not self.shuffle:
            self._reset_seed()

        self.len = sum(len(dl) for dl in self.dataloaders)

        iters = [iter(d) for d in self.dataloaders]

        while iters:
            it = self.rng.choice(iters)
            try:
                yield next(it)
            except StopIteration:
                iters.remove(it)

    def __len__(self) -> int:
        return sum(len(d) for d in self.dataloaders)


class MCDataset(Dataset[MCBatchItem]):
    """MCDataset is a mapping for Monte Carlo (simulated) dataset containing items that consist of an input tensor, a target tensor, group id and a dict of unstandardized training or nSigma values tensors.
    """

    def __init__(
        self,
        input: Tensor,
        target: Tensor,
        group_id: GroupID, # ID of missing detectors group (binary representation of available columns)
        **unstandardized: Tensor,
    ):
        """__init__

        Args:
            input (Tensor): tensor containing all inputtensors
            target (Tensor): tensor containing all target tensors
            group_id (GroupID): group id of missing detectors group
            **unstandardized (Tensor): dict of tensors containing unstandardized training columns (+ nSigma columns if available) information
        """
        self._input = input
        self._target = target
        self._group_id = group_id
        self._unstandardized = unstandardized

    def __len__(self) -> int:
        return self._target.shape[0]

    def __getitem__(self, index: int) -> MCBatchItem:
        return (
            self._input[index],
            self._target[index],
            self._group_id,
            {key: val[index] for key, val in self._unstandardized.items()},
        )


class ExpDataset(Dataset[ExpBatchItem]):
    """MCDataset is a mapping for Monte Carlo (simulated) dataset containing items that consist of an input tensor, a target tensor, group id and a dict of unstandardized training or nSigma values tensors.
    """

    def __init__(
        self,
        input: Tensor,
        group_id: GroupID, # ID of missing detectors group (binary representation of available columns)
        **unstandardized: Tensor,
    ):
        """__init__

        Args:
            input (Tensor): tensor containing all inputtensors
            group_id (GroupID): group id of missing detectors group
            **unstandardized (Tensor): dict of tensors containing unstandardized training columns (+ nSigma columns if available) information
        """
        self._input = input
        self._group_id = group_id
        self._unstandardized = unstandardized

    def __len__(self) -> int:
        return self._input.shape[0]

    def __getitem__(self, index: int) -> ExpBatchItem:
        return (
            self._input[index],
            self._group_id,
            {key: val[index] for key, val in self._unstandardized.items()},
        )

