from abc import abstractmethod
import dataclasses
import gzip
import json
import os
import pickle
from typing import List, Optional
from numpy.typing import NDArray
from pandas.io.parsers.readers import csv
from torch import nn
import torch
import wandb
from pdi.constants import TARGET_CODE_TO_PART_NAME
from pdi.config import Config

# (training losses array), (validation losses array)
TrainResults = tuple[List[float], List[float]]

# (metrics and optimal threshold dictionary), (inputs, targets, predictions and unstandardized data dictionary)
TestResults = tuple[dict[str, float], dict[str, NDArray]]

class BaseEngine:
    def __init__(self, cfg: Config, target_code: int) -> None:
        self._target_code = target_code
        self._epoch_num = 0
        self._epochs_since_last_progress = 0
        self._cfg = cfg
        self._best_f1 = 0.0
        # It there are many runs on the same config, there
        # should be subdirectories with number for this run
        run_number = 1
        project_target_path = os.path.join(cfg.log_dir, cfg.project_dir, TARGET_CODE_TO_PART_NAME[self._target_code])
        while os.path.exists(os.path.join(project_target_path, f"run_{run_number}")):
            run_number += 1
        os.makedirs(os.path.join(project_target_path, f"run_{run_number}"))
        self._base_dir = os.path.join(project_target_path, f"run_{run_number}")

        # Dump config to base_dir
        with open(os.path.join(self._base_dir, "config.json"), "w") as config_file:
            config_dict = dataclasses.asdict(self._cfg)
            json.dump(config_dict, config_file, indent=4)

    @abstractmethod
    def train(self) -> TrainResults:
        pass

    @abstractmethod
    def test(self, model_dirpath: Optional[str] = None) -> TestResults:
        pass

    def _save_best_model(self, model: nn.Module, epoch: int, threshold: float):
        dirpath = os.path.join(self._base_dir, "model_weights")
        os.makedirs(dirpath, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(dirpath, "best.pt"))
        with open(os.path.join(dirpath, f"metadata.json"), "w") as metadata_file:
            json.dump({
                "threshold": str(threshold),
                "epoch": str(epoch),
                "best_f1": str(self._best_f1),
                "model_class": model.__class__.__name__,
            }, metadata_file, indent=4)

    def _load_model(self, skeleton_model: nn.Module, dirpath: Optional[str] = None) -> tuple[nn.Module, float]:
        if not dirpath:
            dirpath = os.path.join(self._base_dir, "model_weights")
        skeleton_model.load_state_dict(torch.load(os.path.join(dirpath, "best.pt"), weights_only=True))
        with open(os.path.join(dirpath, f"metadata.json"), "r") as metadata_file:
            metadata = json.load(metadata_file)

        return skeleton_model, metadata["threshold"]

    # returns if progress was made
    def _early_stopping_step(self, model: nn.Module, threshold: float, epoch: int, val_loss: float, min_loss: float, val_f1: float) -> bool:
        if (1 - val_loss / min_loss) > self._cfg.training.early_stopping_progress_threshold:
            self._epochs_since_last_progress += 1
            if self._best_f1 < val_f1:
                self._best_f1 = val_f1
                self._save_best_model(model, epoch, threshold)
            return False
        else:
            # TODO: save as best model
            self._epochs_since_last_progress = 0
            return True

    def _should_early_stop(self):
        if self._cfg.training.early_stopping_epoch_count == 0:
            return False

        return self._epochs_since_last_progress >= self._cfg.training.early_stopping_epoch_count

    def _log_results(self, metrics: dict, csv_name: str, offline: bool = False, step: int | None = None):
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
        dir_path = os.path.join(self._base_dir)
        os.makedirs(dir_path, exist_ok=True)
        csv_path = os.path.join(dir_path, csv_name)

        file_exists = os.path.exists(csv_path)
        with open(csv_path, mode="a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["step"] + list(metrics.keys()) if step else list(metrics.keys()))
            if not file_exists:
                writer.writeheader()  # Write header if file doesn't exist
            writer.writerow({"step": step, **metrics} if step else metrics)

    def _save_test_results(self, results: dict[str, NDArray], filename="test_prediction_results_with_inputs"):
        path = os.path.join(self._base_dir)
        os.makedirs(path, exist_ok=True)

        with gzip.open(
            os.path.join(path, f"{filename}.pkl"), "wb"
        ) as file:
            pickle.dump(results, file)

