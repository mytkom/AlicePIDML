from abc import abstractmethod
from ast import Tuple
import gzip
import os
import pickle
from typing import Iterable, List
from numpy.typing import NDArray
from pandas.io.parsers.readers import csv
from torch import nn, onnx
import torch
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import wandb
from pdi.constants import PARTICLES_DICT
from pdi.config import Config
from pdi.data.data_preparation import CombinedDataLoader, MCBatchItem, MCBatchItemOut

# (training losses array), (validation losses array)
TrainResults = tuple[List[float], List[float]]

# (metrics and optimal threshold dictionary), (inputs, targets, predictions and unstandardized data dictionary)
TestResults = tuple[dict[str, float], dict[str, NDArray]]

class BaseEngine:
    def __init__(self, cfg: Config) -> None:
        self._epoch_num = 0
        self._epochs_since_last_progress = 0
        self._cfg = cfg
        self._best_metric = 0.0
        # It there are many runs on the same config, there
        # should be subdirectories with number for this run
        run_number = 1
        while os.path.exists(os.path.join(cfg.log_dir, cfg.project_dir, f"run_{run_number}")):
            run_number += 1
        self._base_dir = os.path.join(cfg.log_dir, cfg.project_dir, f"run_{run_number}")

    @abstractmethod
    def train(self, target_code: int) -> TrainResults:
        pass

    @abstractmethod
    def test(self, target_code: int) -> TestResults:
        pass

    def _save_model(self, model: nn.Module, target_code: int, epoch: int):
        dirpath = os.path.join(self._base_dir, "models", f"{PARTICLES_DICT[target_code]}")
        os.makedirs(dirpath, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(dirpath, f"epoch_{epoch}.pt"))

    def _save_best_model(self, skeleton_model: nn.Module, target_code: int, best_epoch: int):
        self._load_model(skeleton_model, target_code, best_epoch)

        # TODO: auto onnx export
        # onnx.export(skeleton_model)


    def _load_model(self, skeleton_model: nn.Module, target_code: int, epoch: int):
        path = os.path.join(self._base_dir, "models", f"{PARTICLES_DICT[target_code]}", f"epoch_{epoch}.pt")
        skeleton_model.load_state_dict(torch.load(path, weights_only=True))

    # returns if progress was made
    def _early_stopping_step(self, val_loss: float, min_loss: float) -> bool:
        if (1 - val_loss / min_loss) > self._cfg.training.early_stopping_progress_threshold:
            self._epochs_since_last_progress += 1
            return False
        else:
            self._epochs_since_last_progress = 0
            return True

    def _should_early_stop(self):
        if self._cfg.training.early_stopping_epoch_count == 0:
            return False

        return self._epochs_since_last_progress >= self._cfg.training.early_stopping_epoch_count

    def _log_results(self, metrics: dict, target_code: int, csv_name: str, offline: bool = False, step: int | None = None):
        """
        Logs metrics and saves them to a CSV file.

        Args:
            metrics (dict): Dictionary of metrics to log.
            step (int): Current validation step.
            csv_path (str): Path to the CSV file for saving metrics.
        """

        if not offline:
            # Log metrics using the accelerator
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)

        # Save metrics to CSV
        dir_path = os.path.join(self._base_dir, PARTICLES_DICT[target_code])
        os.makedirs(dir_path, exist_ok=True)
        csv_path = os.path.join(dir_path, csv_name)

        file_exists = os.path.exists(csv_path)
        with open(csv_path, mode="a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["step"] + list(metrics.keys()) if step else list(metrics.keys()))
            if not file_exists:
                writer.writeheader()  # Write header if file doesn't exist
            writer.writerow({"step": step, **metrics} if step else metrics)

    def _save_test_results(self, results: dict[str, NDArray], target_code: int, filename="test_prediction_results_with_inputs"):
        path = os.path.join(self._base_dir, PARTICLES_DICT[target_code])
        os.makedirs(path, exist_ok=True)

        with gzip.open(
            os.path.join(path, f"{filename}.pkl"), "wb"
        ) as file:
            pickle.dump(results, file)

