from abc import abstractmethod
import dataclasses
import json
import os
from typing import Optional
from joblib.pool import np
from pandas.io.parsers.readers import csv
from torch import nn
import torch
import wandb
from pdi.constants import TARGET_CODE_TO_PART_NAME
from pdi.config import Config
from pdi.data.constants import COLUMNS_FOR_TRAINING
from pdi.data.data_preparation import DataPreparation
from pdi.results_and_metrics import TrainResults, TestResults

class BaseEngine:
    """
    BaseEngine is a class, which is a super class of all the engines. Engine is an object, which knows
    how to run data preparation, training and testing of a specific model and how to save files regarding it.
    Child classes of BaseEngine needs to implement train(), test() and get_data_prep() methods and can use helper methods
    to extract common behaviour like logging.

    Function build_engine(), which knows what engine is suitable for given config is in __init__.py file
    of this (engines) module.
    """
    def __init__(self, cfg: Config, target_code: int, base_dir: str | None) -> None:
        self._target_code = target_code
        self._epoch_num = 0
        self._epochs_since_last_progress = 0
        self._cfg = cfg
        self._best_f1 = 0.0
        # It there are many runs on the same config, there
        # should be subdirectories with number for this run
        if base_dir and os.path.exists(base_dir):
            self._base_dir = base_dir
        else:
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
    def get_data_prep(self) -> DataPreparation:
        pass

    @abstractmethod
    def train(self) -> TrainResults:
        pass

    @abstractmethod
    def _test(self, model_dirpath: Optional[str] = None) -> TestResults:
        pass

    def test(self, model_dirpath: Optional[str] = None) -> TestResults:
        base_dir = model_dirpath if model_dirpath is not None else self._base_dir
        test_results_path = os.path.join(base_dir, "test_results.pkl")

        # Try to load cached TestResults
        if os.path.exists(test_results_path):
            try:
                return TestResults.from_file(test_results_path)
            except:
                print("Cannot use precomputed TestResults, computing from scratch:")

        results = self._test(model_dirpath)

        self._log_results({f"test/{k}": v for k,v in results.test_metrics.to_dict().items()}, csv_name=f"test_metrics.csv")

        print("Test results:")
        print(results.test_metrics.to_dict())

        # Save TestResults
        results.save(test_results_path)

        return results

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

        artifact = wandb.Artifact("best_model", type="model")

        model.to("cpu")
        model_path = os.path.join(dirpath, "best.pt")
        torch.save(model.state_dict(), model_path)

        artifact.add_file(model_path)

        metadata_path = os.path.join(dirpath, f"metadata.json")
        with open(metadata_path, "w") as metadata_file:
            json.dump({
                "threshold": str(threshold),
                "epoch": str(epoch),
                "best_f1": str(self._best_f1),
                "model_class": model.__class__.__name__,
            }, metadata_file, indent=4)

        artifact.add_file(metadata_path)

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

        artifact.add_file(onnx_path)

        model.to(self._cfg.training.device)
        wandb.log_artifact(artifact)

    def _load_model(self, skeleton_model: nn.Module, dirpath: Optional[str] = None) -> tuple[nn.Module, float]:
        """
        Loads weights for pytorch model from dirpath according to _save_best_model() file naming convention.
        """
        if not dirpath:
            dirpath = self._base_dir
        dirpath = os.path.join(dirpath, "model_weights")

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


