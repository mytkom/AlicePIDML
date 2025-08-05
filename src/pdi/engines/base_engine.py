from abc import abstractmethod
import dataclasses
import gzip
import json
import os
import pickle
from typing import List, Optional
from joblib.pool import np
from numpy import float32
from numpy.typing import NDArray
from pandas.io.parsers.readers import csv
from torch import nn
import torch
import wandb
from pdi.constants import TARGET_CODE_TO_PART_NAME
from pdi.config import Config
from pdi.data.constants import COLUMNS_FOR_TRAINING
from pdi.evaluate import maximize_f1
from sklearn.metrics import precision_score, recall_score, f1_score

class ValidationMetrics:
    """
    Represents evaluation metrics for training/validation, calculated dynamically using optimal threshold for F1 score.
    """
    def __init__(self, targets: NDArray, predictions: NDArray, loss: float):
        self.f1, self.precision, self.rGecall, self.threshold = maximize_f1(targets, predictions)
        self.loss = loss

    def to_dict(self) -> dict:
        """
        Converts the validation metrics to a dictionary for logging or serialization.
        """
        return {
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "loss": self.loss,
            "threshold": self.threshold,
        }

class TestMetrics:
    """
    Represents evaluation metrics for test evaluation, calculated using the optimal threshold from validation.
    """
    def __init__(self, targets: NDArray, predictions: NDArray, threshold: float, target_code: int, loss: Optional[float] = None):
        binary_targets = targets == target_code
        binary_predictions = predictions >= threshold
        self.f1 = f1_score(binary_targets, binary_predictions, average="binary")
        self.precision = precision_score(binary_targets, binary_predictions, average="binary")
        self.recall = recall_score(binary_targets, binary_predictions, average="binary")
        self.loss = loss
        self.threshold = threshold
        self.target_code = target_code

    def to_dict(self) -> dict:
        """
        Converts the test metrics to a dictionary for logging or serialization.
        """
        return {
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "loss": self.loss,
            "threshold": self.threshold,
            "target_code": self.target_code,
        }

class TrainResults:
    """
    Represents training results, including validation metrics and loss data.
    """
    def __init__(self, train_losses: List[float], val_losses: List[float]):
        self.train_losses = train_losses
        self.val_losses = val_losses

    def to_dict(self) -> dict:
        """
        Converts the training results to a dictionary for logging or serialization.
        """
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }


class TestResults:
    """
    Represents test results, including test metrics and data used for evaluation.
    """
    def __init__(self, inputs: NDArray, targets: NDArray, predictions: NDArray, unstandardized: dict[str, NDArray], test_metrics: TestMetrics):
        self.inputs = inputs
        self.targets = targets
        self.predictions = predictions
        self.unstandardized = unstandardized
        self.test_metrics = test_metrics

    @classmethod
    def from_file(cls, filepath: str) -> "TestResults":
        """
        Initializes the TestResults object by loading data from a pickle file.

        Args:
            filepath (str): Path to the pickle file containing serialized test results.

        Returns:
            TestResults: An instance of TestResults initialized with the loaded data.
        """
        with gzip.open(filepath, "rb") as file:
            data = pickle.load(file)

        test_metrics = TestMetrics(
            targets=np.array(data["targets"]),
            predictions=np.array(data["predictions"]),
            threshold=data["test_metrics"]["threshold"],
            target_code=data["test_metrics"]["target_code"],
            loss=data["test_metrics"]["loss"],
        )

        return cls(
            inputs=np.array(data["inputs"]),
            targets=np.array(data["targets"]),
            predictions=np.array(data["predictions"]),
            unstandardized={key: np.array(value) for key, value in data["unstandardized"].items()},
            test_metrics=test_metrics,
        )

    def to_dict(self) -> dict:
        """
        Converts the test results to a dictionary for logging or serialization.
        """
        return {
            "inputs": self.inputs.tolist(),
            "targets": self.targets.tolist(),
            "predictions": self.predictions.tolist(),
            "unstandardized": {key: value.tolist() for key, value in self.unstandardized.items()},
            "test_metrics": self.test_metrics.to_dict(),
        }

class BaseEngine:
    """
    BaseEngine is a class, which is a super class of all the engines. Engine is an object, which knows
    how to run data preparation, training and testing of a specific model and how to save files regarding it.
    Child classes of BaseEngine needs to implement train() and test() methods and can use helper methods
    to extract common behaviour like logging.

    Function build_engine(), which knows what engine is suitable for given config is in __init__.py file
    of this (engines) module.
    """
    def __init__(self, cfg: Config, target_code: int) -> None:
        self._target_code = target_code
        self._epoch_num = 0
        self._epochs_since_last_progress = 0
        self._cfg = cfg
        self._best_f1 = 0.0
        # It there are many runs on the same config, there
        # should be subdirectories with number for this run
        run_number = 1
        project_target_path = os.path.join(cfg.results_dir, cfg.project_dir, TARGET_CODE_TO_PART_NAME[self._target_code])
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
        """
        Helper function suitable to save best model of pytorch training engine.
        If your engine is not training pytorch model, unfortunatelly this helper
        cannot be used.

        It saves two files in subdirectory "model_weights":
            - best.pt with model weights
            - metadata.json with model metadata
        """
        dirpath = os.path.join(self._base_dir, "model_weights")
        os.makedirs(dirpath, exist_ok=True)

        model.to("cpu")
        torch.save(model.state_dict(), os.path.join(dirpath, "best.pt"))
        with open(os.path.join(dirpath, f"metadata.json"), "w") as metadata_file:
            json.dump({
                "threshold": str(threshold),
                "epoch": str(epoch),
                "best_f1": str(self._best_f1),
                "model_class": model.__class__.__name__,
            }, metadata_file, indent=4)

        # Export ONNX
        onnx_path = os.path.join(dirpath, "model.onnx")
        dummy_input = torch.tensor(np.random.rand(1, len(COLUMNS_FOR_TRAINING)), dtype=torch.float32)
        model_with_sigmoid = nn.Sequential(model, nn.Sigmoid())
        torch.onnx.export(model_with_sigmoid, dummy_input, onnx_path,
                          export_params=True,
                          opset_version=14,
                          do_constant_folding=True,
                          input_names=["input"],
                          output_names=["output"],
                          dynamic_axes={"input": {0: 'batch size'}})
        
        model.to(self._cfg.training.device)

    def _load_model(self, skeleton_model: nn.Module, dirpath: Optional[str] = None) -> tuple[nn.Module, float]:
        """
        Loads weights for pytorch model from dirpath according to _save_best_model() file naming convention.
        """
        if not dirpath:
            dirpath = os.path.join(self._base_dir, "model_weights")
        skeleton_model.load_state_dict(torch.load(os.path.join(dirpath, "best.pt"), weights_only=True, map_location=self._cfg.training.device))
        with open(os.path.join(dirpath, f"metadata.json"), "r") as metadata_file:
            metadata = json.load(metadata_file)

        return skeleton_model, float(metadata["threshold"])

    # returns if progress was made
    def _early_stopping_step(self, model: nn.Module, threshold: float, epoch: int, val_loss: float, min_loss: float, val_f1: float) -> bool:
        if (1 - val_loss / min_loss) > self._cfg.training.early_stopping_progress_threshold:
            self._epochs_since_last_progress = 0
            if self._best_f1 < val_f1:
                self._best_f1 = val_f1
                self._save_best_model(model, epoch, threshold)
            return True
        else:
            self._epochs_since_last_progress += 1
            return False

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

        if wandb.run is not None and not offline:
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


