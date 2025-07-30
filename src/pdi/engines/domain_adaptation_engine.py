from typing import Optional, cast
from joblib.pool import np
from numpy.typing import NDArray
from sklearn.metrics import precision_score, recall_score
from torch.functional import Tensor
import torch
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from tqdm import tqdm
from pdi.config import Config
from pdi.data.data_preparation import CombinedDataLoader, ExpBatchItem, ExpBatchItemOut, MCBatchItem, GeneralDataPreparation, MCBatchItemOut
from pdi.data.types import GroupID
from pdi.engines.base_engine import BaseEngine, TestResults, TrainResults
from pdi.evaluate import maximize_f1
from pdi.losses import build_loss
from pdi.models import build_model
from pdi.optimizers import build_optimizer
from pdi.lr_schedulers import build_lr_scheduler

class DomainAdaptationEngine(BaseEngine):
    def __init__(self, cfg: Config, target_code: int) -> None:
        super().__init__(cfg, target_code)
        self._sim_data_prep = GeneralDataPreparation(cfg.data, cfg.sim_dataset_paths, cfg.seed)
        (self._sim_train_dl, self._sim_val_dl, self._sim_test_dl) = self._sim_data_prep.create_dataloaders(
            batch_size=self._cfg.training.batch_size,
            num_workers=self._cfg.training.num_workers,
            undersample=self._cfg.data.undersample_missing_detectors,
            seed=self._cfg.seed
        )
        if self._sim_data_prep._is_experimental:
            raise RuntimeError("DomainAdaptationEngine: Expected simulated data, but got experimental data in cfg.sim_dataset_paths!")

        self._exp_data_prep = GeneralDataPreparation(cfg.data, cfg.exp_dataset_paths, cfg.seed)
        (self._exp_train_dl, self._exp_val_dl, self._exp_test_dl) = self._exp_data_prep.create_dataloaders(
            batch_size=self._cfg.training.batch_size,
            num_workers=self._cfg.training.num_workers,
            undersample=self._cfg.data.undersample_missing_detectors,
            seed=self._cfg.seed
        )
        if not self._exp_data_prep._is_experimental:
            raise RuntimeError("DomainAdaptationEngine: Expected experiments data, but got simulated data in cfg.exp_dataset_paths!")


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

            # Validation
            val_metrics = self._evaluate(
                model=model,
                loss_func_class=loss_func_class,
                loss_func_domain=loss_func_domain,
                sim_dataloader=cast(CombinedDataLoader[MCBatchItem, MCBatchItemOut], self._sim_val_dl),
                exp_dataloader=cast(CombinedDataLoader[ExpBatchItem, ExpBatchItemOut], self._exp_val_dl),
            )

            # Threshold for posterior probability to identify as positive
            # it is optimized for f1 metric
            model.thres = torch.tensor(np.array(val_metrics["val/threshold"])).to(self._cfg.training.device)

            val_loss = val_metrics['val/loss']

            # New learning rate for the next
            scheduler.step()

            loss_arr.append(train_loss)
            val_loss_arr.append(val_metrics['val/loss'])

            self._early_stopping_step(
                model=model,
                threshold=val_metrics["val/threshold"],
                epoch=epoch,
                val_loss=val_loss,
                min_loss=min_loss,
                val_f1=val_metrics["val/f1"],
            )
            if  val_loss < min_loss:
                min_loss = val_loss

            # Log validation metrics
            self._log_results(
                metrics = {
                    "epoch": epoch,
                    "scheduled_lr": scheduler.get_last_lr()[0],
                    "val/f1_best": self._best_f1,
                    **val_metrics,
                },
                csv_name = f"validation_metrics.csv"
            )
            print(
                f"Epoch: {epoch}, F1: {val_metrics['val/f1']:.4f}, Loss: {train_loss:.4f}, Val_Loss:{val_loss:.4f}"
            )

            if self._should_early_stop():
                print(f"Finishing training early at epoch: {epoch}")
                break

        self._model = model

        return loss_arr, val_loss_arr

    def test(self, model_dirpath: Optional[str] = None) -> TestResults:
        loss_class: _Loss = build_loss(self._cfg.training)
        loss_domain: _Loss = build_loss(self._cfg.training)
        model: torch.nn.Module = build_model(self._cfg.model, group_ids=self._sim_data_prep.get_group_ids())
        model, threshold = self._load_model(model, model_dirpath)
        model.to(self._cfg.training.device)

        metrics, results = self._test(
            model=model,
            threshold=threshold,
            loss_func_class=loss_class,
            loss_func_domain=loss_domain,
            sim_dataloader=cast(CombinedDataLoader[MCBatchItem, MCBatchItemOut], self._sim_test_dl),
            exp_dataloader=cast(CombinedDataLoader[ExpBatchItem, ExpBatchItemOut], self._exp_test_dl),
        )

        self._log_results(metrics, csv_name=f"test_metrics.csv")
        self._save_test_results(results["class"], filename="test_prediction_results_class_labels")
        self._save_test_results(results["domain"], filename="test_prediction_results_class_domain")

        print(metrics)
        return metrics, results

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
            optimizer.zero_grad()

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

    # TODO: clean up this---for sure common behaviour can be extracted
    def _evaluate(
            self,
            model: torch.nn.Module,
            loss_func_class: _Loss,
            loss_func_domain: _Loss,
            sim_dataloader: CombinedDataLoader[MCBatchItem, MCBatchItemOut],
            exp_dataloader: CombinedDataLoader[ExpBatchItem, ExpBatchItemOut],
    ) -> dict[str, float]:
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

        # TODO: maybe it should be a DataFrame?
        # Final data, predicted posterior probabilities and their corresponding input tensors,
        # target codes and unstandardized columns
        final_class_results = {
            "targets": np.array(class_results["targets"], dtype=np.float32).squeeze(),
            "predictions": np.array(class_results["predictions"]).squeeze(),
        }

        val_loss = val_loss / count

        binary_targets = final_class_results["targets"] == self._target_code
        val_f1, val_prec, val_rec, var_thres = maximize_f1(binary_targets, final_class_results["predictions"])

        final_domain_results = {
            "targets": np.array(domain_results["targets"], dtype=np.float32).squeeze(),
            "predictions": np.array(domain_results["predictions"]).squeeze(),
        }
        # TODO: make constant threshold, add accuracy
        dom_val_f1, dom_val_prec, dom_val_rec, dom_var_thres = maximize_f1(final_domain_results["targets"], final_domain_results["predictions"])

        # TODO: clean up return value
        return {
            "val/f1": val_f1,
            "val/precision": val_prec,
            "val/recall": val_rec,
            "val/threshold": var_thres,
            "val/loss": val_loss,
            "domain/f1": dom_val_f1,
            "domain/precision": dom_val_prec,
            "domain/recall": dom_val_rec,
            "domain/optimal_threshold": dom_var_thres,
        }


    # TODO: clean up this---for sure common behaviour can be extracted
    def _test(
            self,
            model: torch.nn.Module,
            threshold: float,
            loss_func_class: _Loss,
            loss_func_domain: _Loss,
            sim_dataloader: CombinedDataLoader[MCBatchItem, MCBatchItemOut],
            exp_dataloader: CombinedDataLoader[ExpBatchItem, ExpBatchItemOut],
    ) -> tuple[dict, dict]:
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

            predict_class_binary = torch.sigmoid(sim_class_out) >= threshold
            class_results["predictions"].extend(predict_class_binary.cpu().detach().numpy())

            predict_domain_binary = torch.sigmoid(torch.cat((sim_domain_out, exp_domain_out), dim=0)) >= DOMAIN_CLASSIFIER_THRESHOLD
            domain_targets = torch.cat((torch.zeros_like(sim_domain_out), torch.ones_like(exp_domain_out)), dim=0)
            domain_results["targets"].extend(domain_targets.cpu().detach().numpy())
            domain_results["predictions"].extend(predict_domain_binary.cpu().detach().numpy())

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

        # TODO: maybe it should be a DataFrame?
        # Final data, predicted posterior probabilities and their corresponding input tensors,
        # target codes and unstandardized columns
        final_class_results = {
            "inputs": np.array(class_results["inputs"]).squeeze(),
            "targets": np.array(class_results["targets"], dtype=np.float32).squeeze(),
            "predictions": np.array(class_results["predictions"]).squeeze(),
            "gids": np.array(class_results["gids"]).squeeze(),
            **{k: np.array(v).squeeze() for k, v in class_results["unstandardized"].items()}
        }

        test_loss = test_loss / count

        binary_targets = final_class_results["targets"] == self._target_code
        binary_predictions = final_class_results["predictions"]
        test_precision: float = float(precision_score(binary_targets, binary_predictions))
        test_recall: float = float(recall_score(binary_targets, binary_predictions))
        test_f1 = float((test_precision * test_recall * 2) / (test_precision + test_recall + np.finfo(float).eps))

        final_domain_results = {
            "inputs": np.array(domain_results["inputs"]).squeeze(),
            "targets": np.array(domain_results["targets"], dtype=np.float32).squeeze(),
            "predictions": np.array(domain_results["predictions"]).squeeze(),
            "gids": np.array(domain_results["gids"]).squeeze(),
            **{k: np.array(v).squeeze() for k, v in domain_results["unstandardized"].items()}
        }

        domain_binary_targets = final_domain_results["targets"] == self._target_code
        domain_binary_predictions = final_domain_results["predictions"]
        domain_precision: float = float(precision_score(domain_binary_targets, domain_binary_predictions))
        domain_recall: float = float(recall_score(domain_binary_targets, domain_binary_predictions))
        domain_f1 = float((domain_precision * domain_recall * 2) / (domain_precision + domain_recall + np.finfo(float).eps))

        # TODO: clean up return value
        return {
            "test/f1": test_f1,
            "test/precision": test_precision,
            "test/recall": test_recall,
            "test/threshold": threshold,
            "test/loss": test_loss,
            "domain/f1": domain_f1,
            "domain/precision": domain_precision,
            "domain/recall": domain_recall,
        }, {
                "class": final_class_results,
                "domain": final_domain_results,
        }

