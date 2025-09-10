import dataclasses
import gzip
import json
import os
import pickle
import warnings
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler
import uproot3
import numpy as np
import pandas as pd
import torch
import hashlib
from pathlib import Path
from random import Random
from typing import (
    Callable,
    Generic,
    Iterator,
    Tuple,
    TypeVar,
    Optional,
    List,
    MutableMapping,
)
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from pdi.constants import TARGET_CODE_TO_PART_NAME, TARGET_CODES
from pdi.data.group_id_helpers import binary_array_to_group_id, group_id_to_detectors_available
from pdi.data.constants import (
    PART_DICT,
    PROCESSED_DIR,
    TARGET_COLUMN,
    COLUMNS_FOR_TRAINING,
    NSIGMA_COLUMNS,
)
from pdi.data.outlier_filtering import build_outlier_filtering
from pdi.data.types import GroupID, InputTarget, Split
from pdi.config import DataConfig
from pdi.results_and_metrics import TestResults

# Target code (ground truth) is available in Monte Carlo (simulated) data
# Tuple[standardized inputs tensor, target codes tensor, GroupID, undstandardized data]
MCBatchItem = tuple[Tensor, Tensor, GroupID, MutableMapping[str, Tensor]]
# GroupID -> Tensor in typing https://github.com/pytorch/pytorch/issues/119123
MCBatchItemOut = tuple[Tensor, Tensor, Tensor, MutableMapping[str, Tensor]]

# Target code (ground truth) is not available in Experimental (real) data
# Tuple[standardized inputs tensor, GroupID, undstandardized data]
ExpBatchItem = tuple[Tensor, GroupID, MutableMapping[str, Tensor]]
# GroupID -> Tensor in typing https://github.com/pytorch/pytorch/issues/119123
ExpBatchItemOut = tuple[Tensor, Tensor, MutableMapping[str, Tensor]]


class MCDataset(Dataset[MCBatchItem]):
    """
    MCDataset is a mapping for Monte Carlo (simulated) dataset containing items
    that consist of an input tensor, a target tensor, group id and a dict of
    unstandardized training columns and nSigma columns (if available).
    """

    def __init__(
        self,
        input: Tensor,
        target: Tensor,
        group_id: GroupID,
        **unstandardized: Tensor,
    ):
        """__init__

        Args:
            input (Tensor): tensor containing all inputtensors
            target (Tensor): tensor containing all target tensors
            group_id (GroupID): ID of missing detectors group (binary representation of available columns)
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

    def to_df(self) -> pd.DataFrame:
        # Convert data to dictionary format
        batch_dict = {
            TARGET_COLUMN: self._target.numpy().squeeze(),
            "GroupID": self._group_id,
        }

        # Add unstandardized data to the batch dictionary
        for k, v in self._unstandardized.items():
            batch_dict[k] = v.numpy()

        # Convert batch data to a DataFrame and append
        return pd.DataFrame(batch_dict)


class ExpDataset(Dataset[ExpBatchItem]):
    """
    ExpDataset is a mapping for experimental dataset containing items that
    consist of an input tensor, group id and a dict of unstandardized
    training columns and nSigma columns (if available).
    """

    def __init__(
        self,
        input: Tensor,
        group_id: GroupID,
        **unstandardized: Tensor,
    ):
        """__init__

        Args:
            input (Tensor): tensor containing all inputtensors
            group_id (GroupID): ID of missing detectors group (binary representation of available columns)
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

    def to_df(self) -> pd.DataFrame:
        # Convert data to dictionary format
        batch_dict = {
            "GroupID": self._group_id,
        }

        # Add unstandardized data to the batch dictionary
        for k, v in self._unstandardized.items():
            batch_dict[k] = v.numpy()

        # Convert batch data to a DataFrame and append
        return pd.DataFrame(batch_dict)


InT = TypeVar("InT")
OutT = TypeVar("OutT")


class CombinedDataLoader(Generic[InT, OutT]):
    """
    CombinedDataLoader combines multiple dataloaders, during iteration there
    is random choice of DataLoader performed. If DataLoader is empty it is removed
    from dataloaders set and iteration is continued until all data is yielded.
    If undersample flag is specified then each dataloader will be yielded the
    same number of times---choice is sequential, not random like in default setting.
    """

    def __init__(
        self, shuffle: bool, undersample: bool, seed: int, *dataloaders: DataLoader[InT]
    ):
        """__init__

        Args:
            shuffle (bool): whether to change item order with each iteration.
            *dataloaders (DataLoader): a list of dataloaders to combine.
        """
        self.shuffle = shuffle
        self.dataloaders = dataloaders
        self.undersample = undersample
        self.rng = Random(seed)

    def __iter__(self) -> Iterator[OutT]:
        iters = [iter(d) for d in self.dataloaders]

        # Round Robin iteration over dataloaders
        end_iteration = False
        if self.undersample:
            while True:
                for it in iters:
                    try:
                        yield next(it)
                    except StopIteration:
                        end_iteration = True
                if end_iteration:
                    break
            return

        if self.shuffle:
            # Random choice in every iteration
            while iters:
                it = self.rng.choice(iters)
                try:
                    yield next(it)
                except StopIteration:
                    iters.remove(it)
        else:
            # When shuffle is off, we want batch_size and num_workers
            # not to change validation/test dataset ordering of batches.
            # This way we can cache minimal TestResults.
            it = iters[0]
            while iters:
                try:
                    yield next(it)
                except StopIteration:
                    iters.remove(it)
                    if iters:
                        it = iters[0]

    def __len__(self) -> int:
        if self.undersample:
            return len(self.dataloaders) * min([len(d) for d in self.dataloaders])
        return sum(len(d) for d in self.dataloaders)

    def unwrap(self) -> pd.DataFrame:
        """
        Unwraps the CombinedDataLoader by concatenating the datasets from all dataloaders into a single DataFrame.
        If undersampling is enabled, all datasets are truncated to the length of the smallest dataset.
        """
        if self.shuffle:
            warnings.warn(
                "Unwrapping shuffled CombinedDataLoader. The resulting data may differ from iterating over the dataloader."
            )

        # Collect data from each dataloader
        data_records = []
        min_len = float("inf")  # Initialize to infinity for finding the minimum length

        for dataloader in self.dataloaders:
            dataset = dataloader.dataset
            if isinstance(dataset, (MCDataset, ExpDataset)):
                data_records.append(dataset.to_df())
                min_len = min(min_len, len(dataset))
            else:
                raise AttributeError(
                    "Cannot unwrap CombinedDataLoader. Unexpected Dataset class."
                )

        # Apply undersampling if enabled
        if self.undersample:
            data_records = [
                df.iloc[self.rng.sample(range(len(df)), int(min(len(df), min_len)))]
                for df in data_records
            ]

        # Concatenate all data into a single DataFrame
        return pd.concat(data_records, ignore_index=True)


def calculate_checksum(
    filenames: list[str],
    config: DataConfig,
    seed: int,
    scaling_params: Optional[pd.DataFrame] = None,
) -> str:
    """
    Calculate a checksum based on the contents of the given files, a config object and seed.

    Args:
        filenames (list[str]): List of file paths to include in the checksum.
        config (object): A dataclass instance representing the configuration.

    Returns:
        str: The calculated checksum as a hexadecimal string.
    """
    hash = hashlib.md5()

    # Include file contents in the checksum
    for fn in filenames:
        try:
            hash.update(Path(fn).read_bytes())
        except IsADirectoryError:
            pass

    # Serialize the config object to JSON and include it in the checksum
    dict_config = dataclasses.asdict(config)
    dict_config["seed"] = seed
    if scaling_params is not None:
        dict_config["scaling_params"] = scaling_params.to_dict()
    config_json = json.dumps(dict_config, sort_keys=True)
    hash.update(config_json.encode("utf-8"))

    return hash.hexdigest()


def is_experimental_data(table_name):
    if "mc" in table_name:
        return False
    return True


def is_extended_data(table_name):
    if "ml" in table_name:
        return False
    return True


def load_root_data(input_files: List[str]) -> Tuple[pd.DataFrame, bool, bool]:
    """
    Load given list of root files into pandas DataFrame. This function automatically detect,
    which out of possible tables is saved in files:
        - O2pidtracksmcml - simulated data, only columns needed for ML training (feature space and labels),
        - O2pidtracksmc - simulated data, more columns---it contains nSigma values,
        - O2pidtracksdataml - experimental data, only columns needed for training (no labels available)
        - O2pidtracksdata - experimental data, more columns---it contains nSigma values

    Args:
        input_files (List[str]): List of paths to ROOT files to be loaded.

    Returns:
        data_with_metadata (Tuple[pd.DataFrame, bool, bool]):
            - A pandas DataFrame containing the concatenated data from the ROOT files.
            - A boolean indicating whether the data is experimental (True) or simulated (False).
            - A boolean indicating whether the data is extended (True) or basic (False).
    """
    MISSING_VALUE_INDICATORS = {
        "fBeta": -999,
        "fTOFSignal": -999,
        "fTRDPattern": 0,
        "fTRDSignal": 0,
    }
    TABLE_NAMES = [
        "O2pidtracksmcml",
        "O2pidtracksmc",
        "O2pidtracksdataml",
        "O2pidtracksdata",
    ]

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
                        raise KeyError(
                            f"Table name in ROOT file must be in {TABLE_NAMES}"
                        )

                tree_data = file[f"%s/{table_name}" % (dirname)].pandas.df()
                dataframes.append(tree_data)

    data = pd.concat(dataframes, ignore_index=True)

    # When there is no detector value in data, some special values indicating missing values
    # are returned from PIDMLProducer task of O2Physics (e.g. -999.0.). This method sets such values to NaNs.
    for column, val in MISSING_VALUE_INDICATORS.items():
        data.loc[data[column] == val, column] = np.NaN

    # TRDPattern is uint8, so cannot use NaN in producer -> need to preprocess it here
    data["fTRDPattern"].mask(np.isclose(data["fTRDPattern"], 0), inplace=True)
    data.loc[data["fTRDPattern"].isnull(), ["fTRDSignal", "fTRDPattern"]] = np.NaN
    data.loc[data["fBeta"].isnull(), ["fTOFSignal", "fBeta"]] = np.NaN
    data.loc[data["fTOFSignal"].isnull(), ["fTOFSignal", "fBeta"]] = np.NaN

    return data, is_experimental_data(table_name), is_extended_data(table_name)


PreparedData = dict[Split, dict[GroupID, dict[InputTarget, pd.DataFrame]]]


class DataPreparation:
    """
    DataPreparation is a class, which for provided DataConfig object handles all
    the following stages of the data preparation (all related to data):
        - loading data from ROOT files and extracting metadata,
        - preprocessing: setting NaNs, cuts on fP and fTCPSignal,
        - splitting into train/validation/test datasets,
        - standardization using mean and std calculated on train dataset,
        - grouping by missing values combinations,
        - creation of data loaders.
    For provided DataConfig, seed and input_paths the checksum is being calculated
    and preprocessed data is being saved for future use in `base_dir/{checksum}` directory.
    Class can also handle loading cached preprocessed data just by constructing new
    DataPreparation object and using desired method, e.g. for obtaining data loaders.

    The structure of the self._prepared_data (where the preprocessed data is stored):
    {
        (Split.TRAIN/Split.VAL/Split.TEST): {
            GroupID: {
                InputTarget.INPUT: pd.DataFrame with standardized input data,
                InputTarget.TARGET: pd.DataFrame with target column (particle species' PDG code),
                InputTarget.UNSTANDARDIZED: pd.DataFrame with unstandardized input columns and nSigma columns (if available)
            }
        }
    }

    Files saved in `base_dir/{checksum}`:
        - for each TRAIN/VAL/TEST splits: "prepared_data_{split}.pkl" with prepared data dictionary
        - "dataset_metadata.json" with data config dumped, seed, input paths and dataset metadata (is_experimental, is_extended).
        - "columns_for_training.json" with array of column labels ordered the same way during training and inference.
        - "scaling_params.json" with standardization params calculated on train dataset.

    DataPreparation contains member constants: COLUMNS_TO_SCAL_RUN_{run number}, which can be
    configured to change the behaviour of the class.
    """

    COLUMNS_TO_SCALE_RUN_3 = [
        "fTPCSignal",
        "fTOFSignal",
        "fBeta",
        "fX",
        "fY",
        "fZ",
        "fAlpha",
        "fTPCNClsShared",
        "fDcaXY",
        "fDcaZ",
    ]

    # TRDSignal scaling in run2
    COLUMNS_TO_SCALE_RUN_2 = COLUMNS_TO_SCALE_RUN_3 + ["fTRDSignal"]

    DEFAULT_NSIGMA_THRESHOLD = 3.0

    def __init__(
        self,
        config: DataConfig,
        input_paths: List[str],
        seed: int,
        base_dir=PROCESSED_DIR,
        scaling_params: Optional[pd.DataFrame] = None,
    ) -> None:
        if len(input_paths) == 0:
            raise KeyError("You must specify at least one input_path with data!")

        # Calculate checsum for input_paths' files, so caching results will be reliable and unique for this set of files
        self._log("Calculating input_paths + configuration checksum:")
        self._inputs_checksum = calculate_checksum(
            input_paths, config, seed, scaling_params
        )
        self._log(f"\tresulting checksum: {self._inputs_checksum}")
        self.save_dir: str = f"{base_dir}/{self._inputs_checksum}"
        if scaling_params is not None:
            self._scaling_params: pd.DataFrame = scaling_params
        else:
            self._scaling_params: pd.DataFrame = pd.DataFrame(
                columns=["column", "mean", "std"]
            )
        self._cfg: DataConfig = config
        self._input_paths: List[str] = input_paths
        self._seed = seed
        self._columns_for_training: List[str] = []
        self._columns_to_standardize: List[str] = (
            self.COLUMNS_TO_SCALE_RUN_3
            if config.is_run_3
            else self.COLUMNS_TO_SCALE_RUN_2
        )
        self._prepared_data: dict[
            Split, dict[GroupID, dict[InputTarget, pd.DataFrame]]
        ] = {}

    def prepare_data(self) -> None:
        """
        Read, preprocess, standardize, split and group data from ROOT source files. It can be called explicitly
        to force data preparation (in case of cached data in bad state).
        """
        # It expects data in ROOT format, which is extracted using O2Physics' task PIDMLProducer.
        # This method can distinguish 4 data formats (simulated or experimental + basic or extended)
        data = self._load_data()
        self._log(f"Number of observations in complete loaded dataset {data.shape[0]}")

        # Get the subset of all data before further processing
        if self._cfg.subset_size:
            data = data.sample(n=self._cfg.subset_size, random_state=self._seed)
            self._log(f"Number of observations after subset filtering {data.shape[0]}")

        # TODO: investigate, why is it needed
        # Later, after splits and grouping some observation can be the only one of its class,
        # such unique observation raises an error at some point in the code.
        if not self._is_experimental:
            data = self._delete_unique_targets(data)

        # TOF is not reliable for transverse momentum (fPt) lower than PT_CUT,
        # TPC was returning outliers (10M signal value), which were bad for standardization parameters
        # calculation - we filter now manually tracks with fTPCSignal > TPC_CUT
        data = self._perform_cuts(data)

        # Split dataset into Train/Validation/Test
        #   split ratios are specified in DataConfig self._cfg
        test_train_split: dict[Split, pd.DataFrame] = self._test_train_split(data)

        # Outlier filtering
        if self._cfg.outlier_filtering_method:
            test_train_split[Split.TRAIN] = self._filter_outliers(test_train_split[Split.TRAIN])

        # Standardization parameters (mean, std) based on train split
        #   results are saved in self._scaling_params
        if not self._is_experimental:
            self._calc_scaling_params(test_train_split[Split.TRAIN])
        else:
            if not self._scaling_params.size > 0:
                raise AttributeError(
                    "[DataPreparation] For experimental data scaling params must be set in constructor!"
                )

        self._prepared_data = {}
        for split, split_data in test_train_split.items():
            split: Split
            split_data: pd.DataFrame

            self._prepared_data[split] = {}

            # Group data by missing detectors; GroupID is binary representation of missing columns,
            # binary 1 mean column is present in group, 0 indicates that it is missing.
            grouped_split_data_dict = self._group_data(split_data)
            for gid, group_data in grouped_split_data_dict.items():
                gid: GroupID
                group_data: pd.DataFrame

                # Split data into inputs, targets and additional unstandardized inputs with nSigma columns if available
                self._prepared_data[split][gid] = (
                    self._input_target_unstandardized_split(group_data)
                )

                # Standardize inputs using previously calculated params
                self._prepared_data[split][gid][InputTarget.INPUT] = (
                    self._standardize_data(
                        self._prepared_data[split][gid][InputTarget.INPUT]
                    )
                )

        # Cache prepared data using checksum calculated in constructor.
        self._save_data()

    def get_prepared_data(self, splits: List[Split] = list(Split)) -> PreparedData:
        """
        Returns PreparedData object. If splits passed to this function are not loaded to self._prepared_data
        it is first loaded. If additional splits are already loaded, they are being returned too.
        """
        self._load_or_prepare_data(splits)

        return self._prepared_data

    def get_nsigma_test_results(
        self, target_code: int, threshold_unscaled: float = DEFAULT_NSIGMA_THRESHOLD
    ) -> TestResults:
        self._load_or_prepare_data([Split.TEST])

        if not self._is_extended:
            raise AttributeError(
                "Data must be extended (contain nSigma columns) to calculate nSigma test results."
            )

        targets = (
            pd.concat([
                v[InputTarget.TARGET] for v in self._prepared_data[Split.TEST].values()
            ])
            .to_numpy()
            .squeeze()
        )
        unstandardized = pd.concat([
            v[InputTarget.UNSTANDARDIZED]
            for v in self._prepared_data[Split.TEST].values()
        ])

        nsigma_normalized_all = self._calc_nsigma_normalized(
            unstandardized, self.DEFAULT_NSIGMA_THRESHOLD
        )
        nsigma_normalized_predictions, scaler = nsigma_normalized_all[target_code]
        threshold_scaled = (
            1 - scaler.transform(np.array([[threshold_unscaled]])).squeeze()
        )

        return TestResults(
            targets,
            nsigma_normalized_predictions,
            threshold_scaled,
            target_code,
        )

    def create_dataloaders(
        self,
        batch_size: dict[Split, int],
        num_workers: dict[Split, int],
        undersample_missing_detectors: bool,
        undersample_pions: bool,
    ) -> tuple[
        CombinedDataLoader[MCBatchItem, MCBatchItemOut]
        | CombinedDataLoader[MCBatchItem, ExpBatchItemOut],
        ...,
    ]:
        """prepare_dataloaders creates dataloaders from preprocessed data.

        Args:
            batch_size (dict[Split, int]): A dictionary mapping each data split (e.g., TRAIN, VAL, TEST)
                to its corresponding batch size.
            num_workers (dict[Split, int]): A dictionary mapping each data split (e.g., TRAIN, VAL, TEST)
                to the number of worker processes used for loading data. See `torch.utils.data.DataLoader`
                for more information.
            undersample_missing_detectors (bool): If true, give the same amount of batches from
                each missing detector combination group of observations in CombinedDataLoader. (Only in train split)
            undersample_pions (bool): If true, undersample pions to the next majority class. (Only in train split)

        Raises:
            AttributeError: If the keys of `batch_size` and `num_workers` dictionaries do not match.
                This ensures that both dictionaries define the same splits.

        Returns:
            tuple[CombinedDataLoader[MCBatchItem, MCBatchItemOut] | CombinedDataLoader[MCBatchItem, ExpBatchItemOut], ...]:
                A tuple of combined dataloaders, one for each split.
        """
        if batch_size.keys() == num_workers.keys():
            splits = batch_size.keys()
        else:
            raise AttributeError(
                "batch_size and num_workers split keys must be the same!"
            )

        self._load_or_prepare_data(splits)

        def create_dataset(
            input_target_unstandardized: dict[InputTarget, pd.DataFrame], gid, split
        ):
            if self._is_experimental:
                return ExpDataset(
                    torch.tensor(
                        input_target_unstandardized[InputTarget.INPUT].values,
                        dtype=torch.float32,
                    ),
                    gid,
                    **{
                        str(column): torch.tensor(val.values, dtype=torch.float32)
                        for column, val in input_target_unstandardized[
                            InputTarget.UNSTANDARDIZED
                        ].items()
                    },
                )

            # Undersample (anti)pions to the next majority group in the training split
            # for simulated data.
            if split == Split.TRAIN and undersample_pions:
                input_target_unstandardized = self._undersample_pions(
                    input_target_unstandardized
                )

            return MCDataset(
                torch.tensor(
                    input_target_unstandardized[InputTarget.INPUT].values,
                    dtype=torch.float32,
                ),
                torch.tensor(
                    input_target_unstandardized[InputTarget.TARGET].values,
                    dtype=torch.float32,
                ),
                gid,
                **{
                    str(column): torch.tensor(val.values, dtype=torch.float32)
                    for column, val in input_target_unstandardized[
                        InputTarget.UNSTANDARDIZED
                    ].items()
                },
            )

        dataloaders: dict[Split, CombinedDataLoader] = {}
        for split in splits:
            grouped_data = self._prepared_data[split]
            datasets = {
                gid: create_dataset(grouped_data[gid], gid, split)
                for gid in grouped_data.keys()
            }
            dataloaders[split] = CombinedDataLoader(
                (split == Split.TRAIN),
                (split == Split.TRAIN and undersample_missing_detectors),
                self._seed,
                *[
                    DataLoader(
                        dataset,
                        batch_size=batch_size[split],
                        shuffle=(split == Split.TRAIN),
                        num_workers=num_workers[split],
                        pin_memory=True,
                    )
                    for dataset in datasets.values()
                ],
            )

        return (*dataloaders.values(),)

    def transform_prepared_data(
        self, transform: Callable[[PreparedData], PreparedData]
    ):
        """
        Handful interface for transforming self._prepared_data from outside. This method can
        be used to e.g. handle missing data imputation strategies.
        """
        self._load_or_prepare_data([Split.TRAIN, Split.VAL, Split.TEST])

        self._prepared_data = transform(self._prepared_data)

    def pos_weight(self, target: int) -> float:
        """
        Another way of handling imbalanced classes is to use weighted loss function during training.
        This function returns weight of positive class for given target class (all other targets are negative class).
        It is calculated solely on train split.

        Args:
            target (int): PDG code of targetted particle specie.

        Returns:
            positive_class_weight (float):
        """
        if self._is_experimental:
            raise RuntimeError("pos_weight is infeasible for experimental dataset!")

        self._load_or_prepare_data([Split.TRAIN])

        target_list = [
            group[InputTarget.TARGET]
            for group in self._prepared_data[Split.TRAIN].values()
        ]
        binary_targets = np.concatenate([targets == target for targets in target_list])
        pos_weight: float = (np.size(binary_targets) - np.sum(binary_targets)) / np.sum(
            binary_targets
        )
        return pos_weight

    def get_group_ids(self) -> list[GroupID]:
        """
        Returns the ids of missing columns groups in the dataset.
        """
        self._load_or_prepare_data([Split.TRAIN])

        return list(self._prepared_data[Split.TRAIN].keys())

    def _save_data(self) -> None:
        """
        Saves preprocessed data, scaling parameters, columns for training and dataset
        metadata in checksum named subdirectory in processed data directory.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        for split, it_data in self._prepared_data.items():
            with gzip.open(
                f"{self.save_dir}/prepared_data_{split.name}.pkl", "wb"
            ) as file:
                pickle.dump(it_data, file)

        self.save_dataset_metadata(self.save_dir)

    def save_dataset_metadata(self, dir: str):
        self._scaling_params.to_json(
            f"{dir}/scaling_params.json",
            index=False,
            orient="split",
        )

        with open(f"{dir}/dataset_metadata.json", "w+", encoding="UTF-8") as f:
            f.write(
                json.dumps({
                    "is_experimental": self._is_experimental,
                    "is_extended": self._is_extended,
                    "input_paths": self._input_paths,
                    "data_config": dataclasses.asdict(self._cfg),
                    "seed": self._seed,
                })
            )

        with open(f"{dir}/columns_for_training.json", "w+", encoding="UTF-8") as f:
            f.write(json.dumps({"columns_for_training": COLUMNS_FOR_TRAINING}))

    def _load_data(self) -> pd.DataFrame:
        """
        Loads ROOT files in self._input_paths. Exracts metadata about dataset: a) if is experimental
        b) if is extended.
        """
        data, self._is_experimental, self._is_extended = load_root_data(
            self._input_paths
        )

        if self._is_experimental:
            self._log("Experimental data detected!")
        else:
            self._log("Simulated (Monte Carlo) data detected!")

        if self._is_extended:
            self._log("Extended dataset detected (with nSigma columns)!")
        else:
            self._log(
                "Basic dataset detected (without nSigma columns---only ML inputs and targets)!"
            )

        return data

    # TODO: inspect, why it is needed---what code do not work if such targets exists
    def _delete_unique_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Delete targets with less than THRESHOLD of observations.
        """
        THRESHOLD = 400
        target_counts = data[TARGET_COLUMN].value_counts()
        for target, count in target_counts.items():
            if count < THRESHOLD:
                data = data.loc[data[TARGET_COLUMN] != target, :]
        return data

    def _perform_cuts(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform simple preprocessing cuts on dataset.
        """
        # TCPSignal must be positive
        data = data.loc[data["fTPCSignal"] > 0, :]

        # TPCSignal sometimes gives huge incorrect values, which negatively impacts standardization
        # maybe outlier detection methods can handle it nicely (for now I leave it be)
        TPC_CUT = 10000
        data = data.loc[data["fTPCSignal"] < TPC_CUT, :]

        # TOF is said to be incorrect even if present if Pt is lower than 0.5 GeV/C
        PT_CUT = 0.5
        data.loc[data["fPt"] < PT_CUT, ["fBeta", "fTOFSignal"]] = np.NaN

        return data

    def _test_train_split(self, data: pd.DataFrame) -> dict[Split, pd.DataFrame]:
        """
        Split dataset into train/validation/test splits. Their ratio is specified in DataConfig object
        in self._cfg. It uses sklearn's method for splitting. If it is simulated data, the split also
        uses statify strategy by target codes. This way particle species' ratios should be the same in
        the resulting splits.
        """
        train_to_val_ratio = self._cfg.train_size / (1 - self._cfg.test_size)

        (data_not_test, test_data) = train_test_split(
            data,
            test_size=self._cfg.test_size,
            random_state=self._seed,
            stratify=None if self._is_experimental else data.loc[:, [TARGET_COLUMN]],
        )
        data_not_test = pd.DataFrame(data_not_test)

        (train_data, val_data) = train_test_split(
            data_not_test,
            train_size=train_to_val_ratio,
            random_state=self._seed,
            stratify=(
                None if self._is_experimental else data_not_test.loc[:, [TARGET_COLUMN]]
            ),
        )

        self._log("Dataset has been splitted in the following ratios:")
        self._log(
            f"\tTrain {self._cfg.train_size}, Validation {1 - self._cfg.test_size - self._cfg.train_size:.2f}, Test {self._cfg.test_size}"
        )

        return {
            Split.TRAIN: pd.DataFrame(train_data),
            Split.VAL: pd.DataFrame(val_data),
            Split.TEST: pd.DataFrame(test_data),
        }

    def _filter_outliers(self, train_data: pd.DataFrame) -> pd.DataFrame:
        """
        Filters outliers from the training data by grouping the data by missing columns
        and target codes, then applying the outlier filtering logic.

        Args:
            train_data (pd.DataFrame): The training data to filter.

        Returns:
            pd.DataFrame: The filtered training data.
        """
        final_indices = []
        # Group by missing columns
        for gid, missing_group_data in self._group_data(train_data).items():
            # Group data by the target column
            target_groups = missing_group_data.groupby(
                TARGET_COLUMN, dropna=False
            )
            for target_code, target_group_ids in target_groups.groups.items():
                self._log(f"Filtering outliers of {TARGET_CODE_TO_PART_NAME[target_code]} for GID {gid}...")
                outlier_filtering = build_outlier_filtering(self._cfg, self._seed)
                part_df = missing_group_data.loc[target_group_ids, self._cfg.outlier_filtering_methods.columns].dropna(axis=1)
                inliers_mask = outlier_filtering(part_df)
                final_indices.extend(part_df.index[inliers_mask])

        train_data_len = train_data.shape[0]
        outliers_count = train_data_len - len(final_indices)
        self._log(f"Filtered {outliers_count} outliers ({float(outliers_count)/train_data_len:.4f} % of train data)")
        return train_data.loc[final_indices]

    def _calc_nsigma_normalized(
        self, unstd: pd.DataFrame, threshold: float
    ) -> dict[int, tuple[NDArray, MinMaxScaler]]:
        """
        Scale nsigma values to [0,1] and reverse meaning (1 - score). After such transformation predictions
        are behaving the same way as torch model predictions and data is transformed so provided threshold (e.g. < 3.0)
        is approx. >=0.5 after transformation.
        """
        nsigma_normalized: dict[int, tuple[NDArray, MinMaxScaler]] = {}
        for target_code in TARGET_CODES:
            tpc_n_sigmas = unstd[f"fTPCNSigma{PART_DICT[abs(target_code)]}"]
            tof_n_sigmas = unstd[f"fTOFNSigma{PART_DICT[abs(target_code)]}"]

            # Apply nsigma formula
            n_sigma_predictions = np.where(
                np.isnan(unstd["fTOFSignal"]),
                np.abs(tpc_n_sigmas),
                np.sqrt(tpc_n_sigmas**2 + tof_n_sigmas**2),
            )

            # minmax_scaler = MinMaxScaler(feature_range=(0,1))
            minmax_scaler = MinMaxScaler(feature_range=(0, 1))
            # make transformed threshold ~0.5
            n_sigma_predictions = np.where(
                n_sigma_predictions > threshold, threshold * 2, n_sigma_predictions
            )
            # make nsigmas to be in range 0 to 1
            n_sigma_predictions_normalized = minmax_scaler.fit_transform(
                n_sigma_predictions.reshape(-1, 1)
            ).squeeze()
            # reverse it so lower nSigma is higher result
            n_sigma_predictions_normalized = 1 - n_sigma_predictions_normalized
            # bad sign
            nsigma_normalized[target_code] = (
                np.where(
                    unstd["fSign"] != np.sign(target_code),
                    0,
                    n_sigma_predictions_normalized,
                ),
                minmax_scaler,
            )

        return nsigma_normalized

    def _undersample_pions(
        self, input_target_unstandardized: dict[InputTarget, pd.DataFrame]
    ) -> dict[InputTarget, pd.DataFrame]:
        """
        Undersample pions (211) and anti-pions (-211) to match the maximum count of other particle groups.
        """
        rng = Random(self._seed)  # Seed for reproducibility

        # Group data by the target column
        groups = input_target_unstandardized[InputTarget.TARGET].groupby(
            TARGET_COLUMN, dropna=False
        )

        # Calculate the maximum count of non-pion groups
        non_pion_counts = groups.size().drop([211, -211], errors="ignore")
        max_count = max(non_pion_counts.values.astype(int))

        # Print group sizes before undersampling
        self._log(
            "Sizes of particle groups in training split before undersampling of pions:"
        )
        self._log(str(groups.size().to_dict()))

        # Undersample pions and anti-pions
        sampled_indices = []
        for target, group_indices in groups.groups.items():
            if target in [211, -211]:  # Pions and anti-pions
                sampled_indices.extend(
                    rng.sample(list(group_indices), min(len(group_indices), max_count))
                )
            else:  # Keep all other particles
                sampled_indices.extend(group_indices)

        # Create a new DataFrame with the sampled indices
        input_target_unstandardized[InputTarget.INPUT] = input_target_unstandardized[
            InputTarget.INPUT
        ].loc[sampled_indices]
        input_target_unstandardized[InputTarget.TARGET] = input_target_unstandardized[
            InputTarget.TARGET
        ].loc[sampled_indices]
        input_target_unstandardized[InputTarget.UNSTANDARDIZED] = (
            input_target_unstandardized[InputTarget.UNSTANDARDIZED].loc[sampled_indices]
        )

        # Print group sizes after undersampling
        self._log(
            "Sizes of particle groups in training split after undersampling of pions:"
        )
        self._log(
            str(
                input_target_unstandardized[InputTarget.TARGET]
                .groupby(TARGET_COLUMN, dropna=False)
                .size()
                .to_dict()
            )
        )

        return input_target_unstandardized

    def _calc_scaling_params(self, train_split: pd.DataFrame):
        """
        Calculate mean and standard deviation for each column, which is in self._columns_to_standardize.
        Those values are saved as a member variable self._scaling_params and later used for standardization.
        """
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

        self._log(
            f"Scaling (standardization) params has been calculated on training split, results:\n{self._scaling_params}"
        )

    def _group_data(self, data):
        """
        Groups data by missing columns. GroupID is binary mask of missing columns (1 - column present, 0 - column missing), order
        of columns in a mask determined by COLUMNS_FOR_TRAINING array. There are helper functions provided in project to handle it easily.
        """
        missing = ~data.isnull()

        groups = missing.groupby(list(COLUMNS_FOR_TRAINING), dropna=False).groups
        grouped_data = {}
        for binary_array, indices in groups.items():
            gid: GroupID = binary_array_to_group_id(np.array(list(binary_array)))
            grouped_data[gid] = data.loc[indices]

        return grouped_data

    def _input_target_unstandardized_split(
        self, data: pd.DataFrame
    ) -> dict[InputTarget, pd.DataFrame]:
        """
        Split data into inputs (standardized inputs for ML model), targets (ground truth, if not experimental) and additional
        information with unstandardized input columns and possibly additional data like nSigma columns.
        """
        input_data = data.loc[:, COLUMNS_FOR_TRAINING]
        if self._is_extended:
            # Add NSigma columns to unstandardized split, if they are available (extended dataset)
            unstandardized = data.loc[
                :, COLUMNS_FOR_TRAINING + NSIGMA_COLUMNS + ["fPt"]
            ]
        else:
            unstandardized = data.loc[:, COLUMNS_FOR_TRAINING + ["fPt"]]

        if not self._is_experimental:
            targets = data.loc[:, [TARGET_COLUMN]]
            return {
                InputTarget.INPUT: input_data,
                InputTarget.TARGET: targets,
                InputTarget.UNSTANDARDIZED: unstandardized,
            }

        return {
            InputTarget.INPUT: input_data,
            InputTarget.UNSTANDARDIZED: unstandardized,
        }

    def _standardize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Make data mean equal 0 and standard deviation equal 1, similarly to standard normal distribution. Then
        it is better conditioned for ML model.
        """
        for _, row in self._scaling_params.iterrows():
            col = row["column"]
            data[col] = (data[col] - row["mean"]) / row["std"]

        return data

    def _try_load_preprocessed_data(self, splits) -> bool:
        """
        Check if data is already preprocessed, returns boolean indicating success.
        """
        if any(key not in self._prepared_data.keys() for key in splits):
            try:
                self._load_preprocessed_data(splits)
            except FileNotFoundError:
                return False

        return True

    def _load_or_prepare_data(self, splits):
        """
        Try to load already cached preprocessed data, if it is not available, prepare it from scratch.
        """
        if not self._try_load_preprocessed_data(splits):
            self._log("Cannot load preprocessed data, preparing it from scratch:")
            self.prepare_data()

    def _load_preprocessed_data(self, splits: List[Split]):
        """
        Load preprocessed data, only expected splits.
        """
        for split in splits:
            with gzip.open(
                f"{self.save_dir}/prepared_data_{split.name}.pkl", "rb"
            ) as file:
                self._prepared_data[split] = pickle.load(file)

        with open(f"{self.save_dir}/dataset_metadata.json", "r", encoding="UTF-8") as f:
            metadata = json.load(f)
            self._is_experimental = metadata["is_experimental"]
            self._is_extended = metadata["is_extended"]

        self._scaling_params = pd.read_json(
            f"{self.save_dir}/scaling_params.json", orient="split"
        )

        self._log(
            "Successfuly loaded preprocessed data! No need for from scratch preparation."
        )

    def _log(self, message: str):
        print(f"[DataPreparation] {message}")
