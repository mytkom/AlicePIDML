# pylint: disable=duplicate-code

import os
import time
from typing import Optional, cast
from joblib.pool import np
from numpy.typing import NDArray
import torch
from torch.functional import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from tqdm import tqdm

from pdi.config import Config
from pdi.data.data_preparation import (
    CombinedDataLoader,
    ExpBatchItem,
    ExpBatchItemOut,
    MCBatchItem,
    DataPreparation,
    MCBatchItemOut,
)
from pdi.data.types import GroupID
from pdi.engines.base_engine import TorchBaseEngine
from pdi.results_and_metrics import (
    TestResults,
    ValidationMetrics,
)
from pdi.losses import build_loss
from pdi.models import build_model
from pdi.optimizers import build_optimizer
from pdi.lr_schedulers import build_lr_scheduler


class DomainAdaptationEngine(TorchBaseEngine):
    """
    Engine suitable for DANN (Domain Adversarial Neural Network) training. It handles both
    simulated data and experimental data.
    """

    def __init__(
        self, cfg: Config, target_code: int, base_dir: str | None = None
    ) -> None:
        super().__init__(cfg, target_code, base_dir)
        self._sim_data_prep = DataPreparation(cfg.data, cfg.sim_dataset_paths, cfg.seed)

        self._sim_train_dl, self._sim_val_dl, self._sim_test_dl = self.setup_dataloaders(self._cfg, self._sim_data_prep) 
        if self._sim_data_prep._is_experimental:
            raise RuntimeError(
                "DomainAdaptationEngine: Expected simulated data, but got experimental data in cfg.sim_dataset_paths!"
            )

        # Experimental
        self._exp_data_prep = DataPreparation(
            cfg.data,
            cfg.exp_dataset_paths,
            cfg.seed,
            scaling_params=self._sim_data_prep._scaling_params,
        )
        self._exp_train_dl, self._exp_val_dl, self._exp_test_dl = self.setup_dataloaders(self._cfg, self._exp_data_prep) 
        if not self._exp_data_prep._is_experimental:
            raise RuntimeError(
                "DomainAdaptationEngine: Expected experiments data, but got simulated data in cfg.exp_dataset_paths!"
            )

        self._sim_data_prep.save_dataset_metadata(self._base_dir)

    def get_data_prep(self) -> DataPreparation:
        return self._sim_data_prep

    def train(self):
        model = build_model(
            self._cfg.model, group_ids=self._sim_data_prep.get_group_ids()
        )
        model.to(self._cfg.training.device)

        pos_weight = None
        # TODO: check if it works as expected, compare results with it and without
        if self._cfg.training.weight_particles_species:
            pos_weight = torch.tensor(
                self._sim_data_prep.pos_weight(self._target_code)
            ).to(self._cfg.training.device)
        loss_func_class: _Loss = build_loss(self._cfg.training, pos_weight=pos_weight)
        loss_func_domain: _Loss = build_loss(self._cfg.training)

        optimizer = build_optimizer(self._cfg.training, model)
        scheduler = build_lr_scheduler(self._cfg.training, optimizer)

        min_loss = torch.inf
        loss_arr = []
        val_loss_arr = []

        for epoch in range(self._cfg.training.max_epochs):
            # One epoch of training
            train_loss = self._train_one_epoch(
                model=model,
                epoch=epoch,
                optimizer=optimizer,
                loss_func_class=loss_func_class,
                loss_func_domain=loss_func_domain,
                sim_dataloader=cast(
                    CombinedDataLoader[MCBatchItem, MCBatchItemOut], self._sim_train_dl
                ),
                exp_dataloader=cast(
                    CombinedDataLoader[ExpBatchItem, ExpBatchItemOut],
                    self._exp_train_dl,
                ),
            )

            # New learning rate for the next
            scheduler.step()

            loss_arr.append(train_loss)

            if epoch % self._cfg.validation.validate_every == 0:
                # Validation
                class_val_metrics, domain_val_metrics = self._evaluate(
                    model=model,
                    loss_func_class=loss_func_class,
                    loss_func_domain=loss_func_domain,
                    sim_dataloader=cast(
                        CombinedDataLoader[MCBatchItem, MCBatchItemOut],
                        self._sim_val_dl,
                    ),
                    exp_dataloader=cast(
                        CombinedDataLoader[ExpBatchItem, ExpBatchItemOut],
                        self._exp_val_dl,
                    ),
                )

                # Threshold for posterior probability to identify as positive
                # it is optimized for f1 metric
                model.thres = torch.tensor(np.array(class_val_metrics.threshold)).to(
                    self._cfg.training.device
                )

                val_loss = class_val_metrics.loss
                val_loss_arr.append(val_loss)

                self._early_stopping_step(
                    model=model,
                    threshold=class_val_metrics.threshold,
                    epoch=epoch,
                    val_loss=val_loss,
                    min_loss=min_loss,
                    val_f1=class_val_metrics.f1,
                )
                min_loss = min(min_loss, val_loss)

                # Log validation metrics
                self._log_results(
                    metrics={
                        "epoch": epoch,
                        "scheduled_lr": scheduler.get_last_lr()[0],
                        "val/f1_best": self._best_f1,
                        **{
                            f"val/{k}": v
                            for k, v in class_val_metrics.to_dict().items()
                        },
                        **{
                            f"val/domain/{k}": v
                            for k, v in domain_val_metrics.to_dict().items()
                        },
                    },
                    csv_name="validation_metrics.csv",
                )
                print(
                    f"Epoch: {epoch}, F1: {class_val_metrics.f1:.4f}, Domain F1: {domain_val_metrics.f1:.4f}, Loss: {train_loss:.4f}, Val_Loss:{val_loss:.4f}"
                )

                if self._should_early_stop():
                    print(f"Finishing training early at epoch: {epoch}")
                    break

        self._model = model

    def _train_one_epoch(
        self,
        model: torch.nn.Module,
        epoch: int,
        optimizer: Optimizer,
        loss_func_class: _Loss,
        loss_func_domain: _Loss,
        sim_dataloader: CombinedDataLoader[MCBatchItem, MCBatchItemOut],
        exp_dataloader: CombinedDataLoader[ExpBatchItem, ExpBatchItemOut],
    ) -> float:
        model.train()
        group_losses: dict[GroupID, NDArray] = {}
        loss_run_sum = 0
        final_loss = 0.0
        count = 0

        # Undersample to minority from source/target dataloaders
        loader_len = min(len(sim_dataloader), len(exp_dataloader))

        sim_iter = iter(sim_dataloader)
        exp_iter = iter(exp_dataloader)

        start_time = time.time()

        # TODO: Consider training common GroupID missing detector combination for
        # sim and exp in each iteration. Maybe it will improve results---distributions
        # between groups can differ.
        for i in tqdm(range(1, loader_len + 1), desc="Training DANN", total=loader_len):
            sim_inputs, sim_targets, _, _ = next(sim_iter)
            # sim_gid: GroupID = cast(GroupID, sim_gids[0])
            exp_inputs, _, _ = next(exp_iter)

            sim_inputs = sim_inputs.to(self._cfg.training.device)
            exp_inputs = exp_inputs.to(self._cfg.training.device)

            sim_binary_targets = (
                (sim_targets == self._target_code)
                .type(torch.float)
                .to(self._cfg.training.device)
            )

            with torch.autocast(
                device_type=self._cfg.training.device,
                dtype=torch.float16,
                enabled=self._cfg.mixed_precision,
            ):
                # Model returns Tensor with 0: positive class posterior, 1: target (exp) domain posterior
                sim_out: Tensor = model(sim_inputs)
                sim_class_out = sim_out[:, 0].unsqueeze(dim=1)
                sim_domain_out = sim_out[:, 1].unsqueeze(dim=1)

                exp_out = model(exp_inputs)
                exp_domain_out = exp_out[:, 1].unsqueeze(dim=1)

                loss_source_class = loss_func_class(sim_class_out, sim_binary_targets)
                loss_source_domain = loss_func_domain(
                    sim_domain_out, torch.zeros_like(sim_domain_out)
                )
                loss_target_domain = loss_func_domain(
                    exp_domain_out, torch.ones_like(exp_domain_out)
                )

            loss = loss_source_class + loss_source_domain + loss_target_domain
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Handle metrics
            loss_run_sum += loss.item()
            final_loss += loss.item()
            count += 1

            self.log_training_metrics(
                log_results_func=self._log_results,
                group_losses=group_losses,
                loss_run_sum=loss_run_sum,
                step=i,
                loader_len=loader_len,
                epoch=epoch,
                steps_to_log=self._cfg.training.steps_to_log,
            )

        epoch_duration = time.time() - start_time
        self._log_results(metrics={ "epoch_duration": epoch_duration }, csv_name="training_durations.csv", offline=False, step=None)

        return final_loss / count

    def _evaluate(
        self,
        model: torch.nn.Module,
        loss_func_class: _Loss,
        loss_func_domain: _Loss,
        sim_dataloader: CombinedDataLoader[MCBatchItem, MCBatchItemOut],
        exp_dataloader: CombinedDataLoader[ExpBatchItem, ExpBatchItemOut],
    ) -> tuple[ValidationMetrics, ValidationMetrics]:
        model.eval()

        class_results = {
            "targets": [],
            "predictions": [],
        }
        domain_results = {
            "targets": [],
            "predictions": [],
        }
        # input_data_tensors = []
        val_loss = 0.0
        count = 0

        # Undersample to minority from source/target dataloaders
        loader_len = min(len(sim_dataloader), len(exp_dataloader))

        sim_iter = iter(sim_dataloader)
        exp_iter = iter(exp_dataloader)

        # TODO: Consider training common GroupID missing detector combination for
        # sim and exp in each iteration. Maybe it will improve results---distributions
        # between groups can differ.
        for _ in tqdm(range(loader_len), desc="Training DANN", total=loader_len):
            sim_inputs, sim_targets, _, _ = next(sim_iter)

            class_results["targets"].extend(sim_targets.cpu().detach().numpy())

            exp_inputs, _, _ = next(exp_iter)

            sim_inputs = sim_inputs.to(self._cfg.training.device)
            exp_inputs = exp_inputs.to(self._cfg.training.device)

            sim_binary_targets = (
                (sim_targets == self._target_code)
                .type(torch.float)
                .to(self._cfg.training.device)
            )

            with torch.autocast(
                device_type=self._cfg.training.device,
                dtype=torch.float16,
                enabled=self._cfg.mixed_precision,
            ):
                # Model returns Tensor with 0: positive class posterior, 1: target (exp) domain posterior
                sim_out = model(sim_inputs)
                sim_class_out = sim_out[:, 0].unsqueeze(dim=1)
                sim_domain_out = sim_out[:, 1].unsqueeze(dim=1)

                exp_out = model(exp_inputs)
                exp_domain_out = exp_out[:, 1].unsqueeze(dim=1)

                loss_source_class = loss_func_class(sim_class_out, sim_binary_targets)
                loss_source_domain = loss_func_domain(
                    sim_domain_out, torch.zeros_like(sim_domain_out)
                )
                loss_target_domain = loss_func_domain(
                    exp_domain_out, torch.ones_like(exp_domain_out)
                )

            loss = loss_source_class + loss_source_domain + loss_target_domain
            loss.backward()

            # Handle metrics
            val_loss += loss.item()
            count += 1

            predict_class = torch.sigmoid(sim_class_out)
            class_results["predictions"].extend(predict_class.cpu().detach().numpy())

            predict_domain = torch.sigmoid(
                torch.cat((sim_domain_out, exp_domain_out), dim=0)
            )
            domain_targets = torch.cat(
                (torch.zeros_like(sim_domain_out), torch.ones_like(exp_domain_out)),
                dim=0,
            )
            domain_results["targets"].extend(domain_targets.cpu().detach().numpy())
            domain_results["predictions"].extend(predict_domain.cpu().detach().numpy())

            del exp_out
            del sim_inputs
            del sim_targets
            del exp_inputs

        if count == 0:
            count = 1

        # Common loss
        val_loss = val_loss / count

        # Class label classification
        squeezed_binary_targets_class = (
            np.array(class_results["targets"], dtype=np.float32).squeeze()
            == self._target_code
        )
        squeezed_predictions_class = np.array(class_results["predictions"]).squeeze()
        class_validation_metrics = ValidationMetrics(
            squeezed_binary_targets_class, squeezed_predictions_class, val_loss
        )

        # Domain label classification
        squeezed_binary_targets_domain = np.array(
            domain_results["targets"], dtype=np.float32
        ).squeeze()
        squeezed_predictions_domain = np.array(domain_results["predictions"]).squeeze()
        domain_validation_metrics = ValidationMetrics(
            squeezed_binary_targets_domain, squeezed_predictions_domain, val_loss
        )

        return class_validation_metrics, domain_validation_metrics

    def _test(
        self,
        model_dirpath: Optional[str] = None,
    ) -> TestResults:
        DOMAIN_CLASSIFIER_THRESHOLD = 0.5

        loss_func_class: _Loss = build_loss(self._cfg.training)
        loss_func_domain: _Loss = build_loss(self._cfg.training)
        model: torch.nn.Module = build_model(
            self._cfg.model, group_ids=self._sim_data_prep.get_group_ids()
        )
        model, threshold = self._load_model(model, model_dirpath)
        model.to(self._cfg.training.device)

        sim_dataloader = cast(
            CombinedDataLoader[MCBatchItem, MCBatchItemOut], self._sim_test_dl
        )
        exp_dataloader = cast(
            CombinedDataLoader[ExpBatchItem, ExpBatchItemOut], self._exp_test_dl
        )

        # Undersample to minority from source/target dataloaders
        loader_len = min(len(sim_dataloader), len(exp_dataloader))

        sim_iter = iter(sim_dataloader)
        exp_iter = iter(exp_dataloader)

        class_results = {
            "targets": [],
            "predictions": [],
        }
        domain_results = {
            "targets": [],
            "predictions": [],
        }
        # input_data_tensors = []
        test_loss = 0.0
        count = 0

        model.eval()

        # TODO: Consider training common GroupID missing detector combination for
        # sim and exp in each iteration. Maybe it will improve results---distributions
        # between groups can differ.
        for _ in tqdm(range(loader_len), desc="Training DANN", total=loader_len):
            sim_inputs, sim_targets, _, _ = next(sim_iter)

            exp_inputs, _, _ = next(exp_iter)

            sim_inputs = sim_inputs.to(self._cfg.training.device)
            exp_inputs = exp_inputs.to(self._cfg.training.device)

            sim_binary_targets = (
                (sim_targets == self._target_code)
                .type(torch.float)
                .to(self._cfg.training.device)
            )

            with torch.autocast(
                device_type=self._cfg.training.device,
                dtype=torch.float16,
                enabled=self._cfg.mixed_precision,
            ):
                # Model returns Tensor with 0: positive class posterior, 1: target (exp) domain posterior
                sim_out = model(sim_inputs)
                sim_class_out = sim_out[:, 0].unsqueeze(dim=1)
                sim_domain_out = sim_out[:, 1].unsqueeze(dim=1)

                exp_out = model(exp_inputs)
                exp_domain_out = exp_out[:, 1].unsqueeze(dim=1)

                loss_source_class = loss_func_class(sim_class_out, sim_binary_targets)
                loss_source_domain = loss_func_domain(
                    sim_domain_out, torch.zeros_like(sim_domain_out)
                )
                loss_target_domain = loss_func_domain(
                    exp_domain_out, torch.ones_like(exp_domain_out)
                )

            loss = loss_source_class + loss_source_domain + loss_target_domain
            loss.backward()

            # Handle metrics
            test_loss += loss.item()
            count += 1

            class_results["targets"].extend(sim_targets.cpu().detach().numpy())
            predict_class = torch.sigmoid(sim_class_out)
            class_results["predictions"].extend(predict_class.cpu().detach().numpy())

            domain_targets = torch.cat(
                (torch.zeros_like(sim_domain_out), torch.ones_like(exp_domain_out)),
                dim=0,
            )
            domain_results["targets"].extend(domain_targets.cpu().detach().numpy())
            predict_domain = torch.sigmoid(
                torch.cat((sim_domain_out, exp_domain_out), dim=0)
            )
            domain_results["predictions"].extend(predict_domain.cpu().detach().numpy())

            del exp_out
            del sim_inputs
            del sim_targets
            del exp_inputs

        if count == 0:
            count = 1

        # Common test loss
        test_loss = test_loss / count

        # Class classification
        squeezed_targets_class = np.array(
            class_results["targets"], dtype=np.float32
        ).squeeze() == self._target_code
        squeezed_predictions_class = np.array(class_results["predictions"]).squeeze()

        class_test_results = TestResults(
            targets=squeezed_targets_class,
            predictions=squeezed_predictions_class,
            threshold=threshold,
            target_code=self._target_code,
            loss=test_loss,
        )

        # Domain classification
        squeezed_targets_domain = np.array(
            domain_results["targets"], dtype=np.float32
        ).squeeze()
        squeezed_predictions_domain = np.array(domain_results["predictions"]).squeeze()
        domain_test_results = TestResults(
            targets=squeezed_targets_domain,
            predictions=squeezed_predictions_domain,
            threshold=DOMAIN_CLASSIFIER_THRESHOLD,
            target_code=self._target_code,
            loss=test_loss,
        )

        self._log_results(
            {
                f"test/domain/{k}": v
                for k, v in domain_test_results.test_metrics.to_dict().items()
            },
            csv_name="test_domain_metrics.csv",
        )

        print("Domain label test results:")
        print(domain_test_results.test_metrics.to_dict())

        domain_test_path = os.path.join(
            self._base_dir if model_dirpath is None else model_dirpath,
            "domain_test_results.pkl",
        )
        domain_test_results.save(domain_test_path)

        return class_test_results
