from abc import abstractmethod
from ast import Tuple
import gzip
import os
import pickle
from typing import Iterable, List
from numpy.typing import NDArray
from pandas.io.parsers.readers import csv
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import wandb
from pdi.constants import PARTICLES_DICT
from pdi.data.config import Config
from pdi.data.data_preparation import MCBatchItem

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
        self._base_dir = os.path.join(cfg.log_dir, cfg.project_dir)

    def _early_stopping_step(self, val_loss: float, min_loss: float):
        if (1 - val_loss / min_loss) > self._cfg.training.early_stopping_progress_threshold:
            self._epochs_since_last_progress += 1
        else:
            self._epochs_since_last_progress = 0

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
        csv_path = os.path.join(self._base_dir, PARTICLES_DICT[target_code], csv_name)
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

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def _train_one_epoch(self, target_code: int, model: nn.Module, optimizer: Optimizer, loss_func: _Loss, dataloader: Iterable[MCBatchItem]) -> float:
        pass

    @abstractmethod
    def _evaluate(self, target_code: int, model: nn.Module, loss: _Loss, dataloader: Iterable[MCBatchItem]) -> TestResults:
        pass

    @abstractmethod
    def train(self, target_code: int) -> TrainResults:
        pass

    @abstractmethod
    def test(self, target_code: int) -> TestResults:
        pass
