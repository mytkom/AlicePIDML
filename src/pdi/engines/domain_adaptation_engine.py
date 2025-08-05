from typing import Optional, cast
from joblib.pool import np
from numpy.typing import NDArray
from torch.functional import Tensor
import torch
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from tqdm import tqdm
from pdi.config import Config
from pdi.data.data_preparation import CombinedDataLoader, ExpBatchItem, ExpBatchItemOut, MCBatchItem, DataPreparation, MCBatchItemOut
from pdi.data.types import GroupID, Split
from pdi.engines.base_engine import BaseEngine, TestResults, TrainResults, ValidationMetrics, TestMetrics
from pdi.losses import build_loss
from pdi.models import build_model
from pdi.optimizers import build_optimizer
from pdi.lr_schedulers import build_lr_scheduler

class DomainAdaptationEngine(BaseEngine):
    """
    Engine suitable for DANN (Domain Adversarial Neural Network) training. It handles both
    simulated data and experimental data.
    """
    def __init__(self, cfg: Config, target_code: int) -> None:
        super().__init__(cfg, target_code)
        self._sim_data_prep = DataPreparation(cfg.data, cfg.sim_dataset_paths, cfg.seed)
        (self._sim_train_dl, self._sim_val_dl, self._sim_test_dl) = self._sim_data_prep.create_dataloaders(
            batch_size={
                    Split.TRAIN: self._cfg.training.batch_size,
                    Split.VAL: self._cfg.validation.batch_size,
                    Split.TEST: self._cfg.validation.batch_size,
            },
            num_workers={
                    Split.TRAIN: self._cfg.training.num_workers,
                    Split.VAL: self._cfg.validation.num_workers,
                    Split.TEST: self._cfg.validation.num_workers,
            },
            undersample_missing_detectors=self._cfg.training.undersample_missing_detectors,
            undersample_pions=self._cfg.training.undersample_pions,
        )
        if self._sim_data_prep._is_experimental:
            raise RuntimeError("DomainAdaptationEngine: Expected simulated data, but got experimental data in cfg.sim_dataset_paths!")

        self._exp_data_prep = DataPreparation(cfg.data, cfg.exp_dataset_paths, cfg.seed, scaling_params=self._sim_data_prep._scaling_params)
        (self._exp_train_dl, self._exp_val_dl, self._exp_test_dl) = self._exp_data_prep.create_dataloaders(
            batch_size={
                    Split.TRAIN: self._cfg.training.batch_size,
                    Split.VAL: self._cfg.validation.batch_size,
                    Split.TEST: self._cfg.validation.batch_size,
            },
            num_workers={
                    Split.TRAIN: self._cfg.training.num_workers,
                    Split.VAL: self._cfg.validation.num_workers,
                    Split.TEST: self._cfg.validation.num_workers,
            },
            undersample_missing_detectors=self._cfg.training.undersample_missing_detectors,
            undersample_pions=self._cfg.training.undersample_pions,
        )
        if not self._exp_data_prep._is_experimental:
            raise RuntimeError("DomainAdaptationEngine: Expected experiments data, but got simulated data in cfg.exp_dataset_paths!")

        self._sim_data_prep.save_dataset_metadata(self._base_dir)


    def train(self) -> TrainResults:
        model = build_model(self._cfg.model, group_ids=self._sim_data_prep.get_group_ids())
        model.to(self._cfg.training.device)

        pos_weight = None
        # TODO: check if it works as expected, compare results with it and without
        if self._cfg.training.weight_particles_species:
            pos_weight = torch.tensor(self._sim_data_prep.pos_weight(self._target_code)).to(self._cfg.training.device)
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
                sim_dataloader=cast(CombinedDataLoader[MCBatchItem, MCBatchItemOut], self._sim_train_dl),
                exp_dataloader=cast(CombinedDataLoader[ExpBatchItem, ExpBatchItemOut], self._exp_train_dl),
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
                    sim_dataloader=cast(CombinedDataLoader[MCBatchItem, MCBatchItemOut], self._sim_val_dl),
                    exp_dataloader=cast(CombinedDataLoader[ExpBatchItem, ExpBatchItemOut], self._exp_val_dl),
                )

                # Threshold for posterior probability to identify as positive
                # it is optimized for f1 metric
                model.thres = torch.tensor(np.array(class_val_metrics.threshold)).to(self._cfg.training.device)

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
                if  val_loss < min_loss:
                    min_loss = val_loss

                # Log validation metrics
                self._log_results(
                    metrics = {
                        "epoch": epoch,
                        "scheduled_lr": scheduler.get_last_lr()[0],
                        "val/f1_best": self._best_f1,
                        **{f"val/{k}": v for k,v in class_val_metrics.to_dict().items()},
                        **{f"val/domain/{k}": v for k,v in domain_val_metrics.to_dict().items()},
                    },
                    csv_name = f"validation_metrics.csv"
                )
                print(
                    f"Epoch: {epoch}, F1: {class_val_metrics.f1:.4f}, Domain F1: {domain_val_metrics.f1:.4f}, Loss: {train_loss:.4f}, Val_Loss:{val_loss:.4f}"
                )

                if self._should_early_stop():
                    print(f"Finishing training early at epoch: {epoch}")
                    break

        self._model = model

        return TrainResults(train_losses=loss_arr, val_losses=val_loss_arr)

    def test(self, model_dirpath: Optional[str] = None) -> TestResults:
        loss_class: _Loss = build_loss(self._cfg.training)
        loss_domain: _Loss = build_loss(self._cfg.training)
        model: torch.nn.Module = build_model(self._cfg.model, group_ids=self._sim_data_prep.get_group_ids())
        model, threshold = self._load_model(model, model_dirpath)
        model.to(self._cfg.training.device)

        class_results, domain_results = self._test(
            model=model,
            threshold=threshold,
            loss_func_class=loss_class,
            loss_func_domain=loss_domain,
            sim_dataloader=cast(CombinedDataLoader[MCBatchItem, MCBatchItemOut], self._sim_test_dl),
            exp_dataloader=cast(CombinedDataLoader[ExpBatchItem, ExpBatchItemOut], self._exp_test_dl),
        )

        self._log_results({f"test/{k}": v for k,v in class_results.test_metrics.to_dict().items()}, csv_name=f"test_metrics.csv")
        self._log_results({f"test/domain/{k}": v for k,v in domain_results.test_metrics.to_dict().items()}, csv_name=f"test_domain_metrics.csv")

        print("Class label test results:")
        print(class_results.test_metrics.to_dict())
        print("Domain label test results:")
        print(domain_results.test_metrics.to_dict())

        # To handle common interface, only class label results are returned
        return class_results

    def _train_one_epoch(
            self,
            model: torch.nn.Module,
            epoch: int,
            optimizer: Optimizer,
            loss_func_class: _Loss,
            loss_func_domain: _Loss,
            sim_dataloader: CombinedDataLoader[MCBatchItem, MCBatchItemOut],
            exp_dataloader: CombinedDataLoader[ExpBatchItem, ExpBatchItemOut]
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

        # TODO: Consider training common GroupID missing detector combination for
        # sim and exp in each iteration. Maybe it will improve results---distributions
        # between groups can differ.
        for i in tqdm(range(1, loader_len + 1), desc="Training DANN", total=loader_len):
            sim_inputs, sim_targets, sim_gids, _ = next(sim_iter)
            sim_gid: GroupID = cast(GroupID, sim_gids[0])
            exp_inputs, exp_gids, _ = next(exp_iter)
            exp_gid: GroupID = cast(GroupID, exp_gids[0])

            sim_inputs = sim_inputs.to(self._cfg.training.device)
            exp_inputs = exp_inputs.to(self._cfg.training.device)

            sim_binary_targets = (sim_targets == self._target_code).type(torch.float).to(self._cfg.training.device)

            with torch.autocast(device_type=self._cfg.training.device, dtype=torch.float16, enabled=self._cfg.mixed_precision):
                # Model returns Tensor with 0: positive class posterior, 1: target (exp) domain posterior
                sim_out: Tensor = model(sim_inputs)
                sim_class_out = sim_out[:,0].unsqueeze(dim=1)
                sim_domain_out = sim_out[:, 1].unsqueeze(dim=1)

                exp_out = model(exp_inputs)
                exp_domain_out = exp_out[:, 1].unsqueeze(dim=1)

                loss_source_class = loss_func_class(sim_class_out, sim_binary_targets)
                loss_source_domain = loss_func_domain(sim_domain_out, torch.zeros_like(sim_domain_out))
                loss_target_domain = loss_func_domain(exp_domain_out, torch.ones_like(exp_domain_out))

            loss = loss_source_class + loss_source_domain + loss_target_domain
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Handle metrics
            loss_run_sum += loss.item()
            final_loss += loss.item()
            count += 1

            if sim_gid not in group_losses.keys():
                group_losses[sim_gid] = np.array([])
            np.append(group_losses[sim_gid], loss.item())

            if i % self._cfg.training.steps_to_log == 0:
                self._log_results(
                    metrics={ "loss": loss_run_sum },
                    step=loader_len*epoch + i,
                    csv_name="training_loss.csv"
                )
                loss_run_sum = 0

        for gid, gid_losses in group_losses.items():
            if len(gid_losses) == 0:
                continue

            self._log_results(
                metrics = {
                    "gid": gid,
                    "mean_loss_epoch": gid_losses.mean(),
                    "gid_count": len(gid_losses),
                },
                csv_name="gid_losses.csv",
                offline=True,
            )

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
        for i in tqdm(range(loader_len), desc="Training DANN", total=loader_len):
            sim_inputs, sim_targets, _, _ = next(sim_iter)

            class_results["targets"].extend(sim_targets.cpu().detach().numpy())

            exp_inputs, _, _ = next(exp_iter)

            sim_inputs = sim_inputs.to(self._cfg.training.device)
            exp_inputs = exp_inputs.to(self._cfg.training.device)

            sim_binary_targets = (sim_targets == self._target_code).type(torch.float).to(self._cfg.training.device)

            with torch.autocast(device_type=self._cfg.training.device, dtype=torch.float16, enabled=self._cfg.mixed_precision):
                # Model returns Tensor with 0: positive class posterior, 1: target (exp) domain posterior
                sim_out = model(sim_inputs)
                sim_class_out = sim_out[:, 0].unsqueeze(dim=1)
                sim_domain_out = sim_out[:, 1].unsqueeze(dim=1)

                exp_out = model(exp_inputs)
                exp_domain_out = exp_out[:, 1].unsqueeze(dim=1)

                loss_source_class = loss_func_class(sim_class_out, sim_binary_targets)
                loss_source_domain = loss_func_domain(sim_domain_out, torch.zeros_like(sim_domain_out))
                loss_target_domain = loss_func_domain(exp_domain_out, torch.ones_like(exp_domain_out))

            loss = loss_source_class + loss_source_domain + loss_target_domain
            loss.backward()

            # Handle metrics
            val_loss += loss.item()
            count += 1

            predict_class = torch.sigmoid(sim_class_out)
            class_results["predictions"].extend(predict_class.cpu().detach().numpy())

            predict_domain = torch.sigmoid(torch.cat((sim_domain_out, exp_domain_out), dim=0))
            domain_targets = torch.cat((torch.zeros_like(sim_domain_out), torch.ones_like(exp_domain_out)), dim=0)
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
        squeezed_binary_targets_class = np.array(class_results["targets"], dtype=np.float32).squeeze() == self._target_code
        squeezed_predictions_class = np.array(class_results["predictions"]).squeeze()
        class_validation_metrics = ValidationMetrics(squeezed_binary_targets_class, squeezed_predictions_class, val_loss)

        # Domain label classification
        squeezed_binary_targets_domain = np.array(domain_results["targets"], dtype=np.float32).squeeze()
        squeezed_predictions_domain = np.array(domain_results["predictions"]).squeeze()
        domain_validation_metrics = ValidationMetrics(squeezed_binary_targets_domain, squeezed_predictions_domain, val_loss)

        return class_validation_metrics, domain_validation_metrics


    def _test(
            self,
            model: torch.nn.Module,
            threshold: float,
            loss_func_class: _Loss,
            loss_func_domain: _Loss,
            sim_dataloader: CombinedDataLoader[MCBatchItem, MCBatchItemOut],
            exp_dataloader: CombinedDataLoader[ExpBatchItem, ExpBatchItemOut],
    ) -> tuple[TestResults, TestResults]:
        DOMAIN_CLASSIFIER_THRESHOLD=0.5

        model.eval()

        class_results = {
            "inputs": [],
            "targets": [],
            "predictions": [],
            "gids": [],
            "unstandardized": {},
        }
        domain_results = {
            "inputs": [],
            "targets": [],
            "predictions": [],
            "gids": [],
            "unstandardized": {},
        }
        # input_data_tensors = []
        test_loss = 0.0
        count = 0

        # Undersample to minority from source/target dataloaders
        loader_len = min(len(sim_dataloader), len(exp_dataloader))

        sim_iter = iter(sim_dataloader)
        exp_iter = iter(exp_dataloader)

        # TODO: Consider training common GroupID missing detector combination for
        # sim and exp in each iteration. Maybe it will improve results---distributions
        # between groups can differ.
        for i in tqdm(range(loader_len), desc="Training DANN", total=loader_len):
            sim_inputs, sim_targets, sim_gids, sim_unstandardized = next(sim_iter)
            sim_gid: GroupID = cast(GroupID, sim_gids[0])

            class_results["targets"].extend(sim_targets.cpu().detach().numpy())
            class_results["inputs"].extend(sim_inputs.cpu().detach().numpy())
            class_results["gids"].extend(sim_gids.cpu().detach().numpy())
            for k, v in sim_unstandardized.items():
                if k not in class_results["unstandardized"]:
                    class_results["unstandardized"][k] = []
                class_results["unstandardized"][k].extend(v.cpu().detach().numpy())

            exp_inputs, exp_gids, exp_unstandardized = next(exp_iter)
            exp_gid: GroupID = cast(GroupID, exp_gids[0])

            domain_results["inputs"].extend(sim_inputs.cpu().detach().numpy())
            domain_results["inputs"].extend(exp_inputs.cpu().detach().numpy())

            domain_results["gids"].extend(sim_gids.cpu().detach().numpy())
            domain_results["gids"].extend(exp_gids.cpu().detach().numpy())

            for k, v in sim_unstandardized.items():
                if k not in domain_results["unstandardized"]:
                    domain_results["unstandardized"][k] = []
                domain_results["unstandardized"][k].extend(v.cpu().detach().numpy())
            for k, v in exp_unstandardized.items():
                if k not in domain_results["unstandardized"]:
                    domain_results["unstandardized"][k] = []
                domain_results["unstandardized"][k].extend(v.cpu().detach().numpy())

            sim_inputs = sim_inputs.to(self._cfg.training.device)
            exp_inputs = exp_inputs.to(self._cfg.training.device)

            sim_binary_targets = (sim_targets == self._target_code).type(torch.float).to(self._cfg.training.device)

            with torch.autocast(device_type=self._cfg.training.device, dtype=torch.float16, enabled=self._cfg.mixed_precision):
                # Model returns Tensor with 0: positive class posterior, 1: target (exp) domain posterior
                sim_out = model(sim_inputs)
                sim_class_out = sim_out[:, 0].unsqueeze(dim=1)
                sim_domain_out = sim_out[:, 1].unsqueeze(dim=1)

                exp_out = model(exp_inputs)
                exp_domain_out = exp_out[:, 1].unsqueeze(dim=1)

                loss_source_class = loss_func_class(sim_class_out, sim_binary_targets)
                loss_source_domain = loss_func_domain(sim_domain_out, torch.zeros_like(sim_domain_out))
                loss_target_domain = loss_func_domain(exp_domain_out, torch.ones_like(exp_domain_out))

            loss = loss_source_class + loss_source_domain + loss_target_domain
            loss.backward()

            # Handle metrics
            test_loss += loss.item()
            count += 1

            predict_class = torch.sigmoid(sim_class_out)
            class_results["predictions"].extend(predict_class.cpu().detach().numpy())

            predict_domain = torch.sigmoid(torch.cat((sim_domain_out, exp_domain_out), dim=0))
            domain_targets = torch.cat((torch.zeros_like(sim_domain_out), torch.ones_like(exp_domain_out)), dim=0)
            domain_results["targets"].extend(domain_targets.cpu().detach().numpy())
            domain_results["predictions"].extend(predict_domain.cpu().detach().numpy())

            del exp_out
            del sim_inputs
            del sim_targets
            del sim_gids
            del sim_unstandardized
            del exp_inputs
            del exp_gids
            del exp_unstandardized

        if count == 0:
            count = 1

        # Common test loss
        test_loss = test_loss / count

        # Class classification
        squeezed_targets_class = np.array(class_results["targets"], dtype=np.float32).squeeze()
        squeezed_predictions_class = np.array(class_results["predictions"]).squeeze()

        class_test_results = TestResults(
            inputs=np.array(class_results["inputs"]).squeeze(),
            targets=squeezed_targets_class,
            predictions=squeezed_predictions_class,
            unstandardized={k: np.array(v).squeeze() for k, v in class_results["unstandardized"].items()},
            test_metrics=TestMetrics(
                targets=squeezed_targets_class,
                predictions=squeezed_predictions_class,
                threshold=threshold,
                target_code=self._target_code,
                loss=test_loss,
            ),
        )

        # Domain classification
        squeezed_targets_domain = np.array(domain_results["targets"], dtype=np.float32).squeeze()
        squeezed_predictions_domain = np.array(domain_results["predictions"]).squeeze()
        domain_test_results = TestResults(
            inputs=np.array(domain_results["inputs"]).squeeze(),
            targets=squeezed_targets_domain,
            predictions=squeezed_predictions_domain,
            unstandardized={k: np.array(v).squeeze() for k, v in domain_results["unstandardized"].items()},
            test_metrics=TestMetrics(
                targets=squeezed_targets_domain,
                predictions=squeezed_predictions_domain,
                threshold=DOMAIN_CLASSIFIER_THRESHOLD,
                target_code=self._target_code,
                loss=test_loss,
            ),
        )

        return class_test_results, domain_test_results
