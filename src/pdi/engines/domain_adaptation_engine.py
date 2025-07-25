from typing import cast
from joblib.pool import np
from numpy.typing import NDArray
from torch.functional import Tensor
import torch
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from tqdm import tqdm
import wandb
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
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self._cfg = cfg
        self._models: dict[int, torch.nn.Module] = {}

        self._sim_data_prep = GeneralDataPreparation(cfg.data, cfg.sim_dataset_paths)
        (self._sim_train_dl, self._sim_val_dl, self._sim_test_dl) = self._sim_data_prep.create_dataloaders(
            batch_size=self._cfg.training.batch_size,
            num_workers=self._cfg.training.num_workers,
            undersample=self._cfg.data.undersample_missing_detectors,
            seed=self._cfg.seed
        )
        if self._sim_data_prep._is_experimental:
            raise RuntimeError("DomainAdaptationEngine: Expected simulated data, but got experimental data in cfg.sim_dataset_paths!")

        self._exp_data_prep = GeneralDataPreparation(cfg.data, cfg.exp_dataset_paths)
        (self._exp_train_dl, self._exp_val_dl, self._exp_test_dl) = self._exp_data_prep.create_dataloaders(
            batch_size=self._cfg.training.batch_size,
            num_workers=self._cfg.training.num_workers,
            undersample=self._cfg.data.undersample_missing_detectors,
            seed=self._cfg.seed
        )
        if not self._exp_data_prep._is_experimental:
            raise RuntimeError("DomainAdaptationEngine: Expected experiments data, but got simulated data in cfg.exp_dataset_paths!")


    def train(self, target_code: int) -> TrainResults:
        if wandb.run is None:
            wandb.init(config=self._cfg.to_dict())
        model = build_model(self._cfg.model, group_ids=self._sim_data_prep.get_group_ids())
        model.to(self._cfg.training.device)

        pos_weight = None
        # TODO: check if it works as expected, compare results with it and without
        if self._cfg.data.weight_particles_species:
            pos_weight = torch.tensor(self._sim_data_prep.pos_weight(target_code)).to(self._cfg.training.device)
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
                target_code=target_code,
                model=model,
                optimizer=optimizer,
                loss_func_class=loss_func_class,
                loss_func_domain=loss_func_domain,
                sim_dataloader=cast(CombinedDataLoader[MCBatchItem, MCBatchItemOut], self._sim_train_dl),
                exp_dataloader=cast(CombinedDataLoader[ExpBatchItem, ExpBatchItemOut], self._exp_train_dl),
            )

            # Validation
            val_metrics, _ = self._evaluate(
                target_code=target_code,
                model=model,
                loss_func_class=loss_func_class,
                loss_func_domain=loss_func_domain,
                sim_dataloader=cast(CombinedDataLoader[MCBatchItem, MCBatchItemOut], self._sim_val_dl),
                exp_dataloader=cast(CombinedDataLoader[ExpBatchItem, ExpBatchItemOut], self._exp_val_dl),
            )

            # Threshold for posterior probability to identify as positive
            # it is optimized for f1 metric
            model.thres = Tensor([val_metrics["class_label/optimal_threshold"]])

            val_loss = val_metrics['loss']

            # New learning rate for the next
            scheduler.step()

            # Log validation metrics
            self._log_results(
                metrics = {
                    "epoch": epoch,
                    "scheduled_lr": scheduler.get_last_lr()[0],
                    **val_metrics,
                },
                target_code = target_code,
                csv_name = f"validation_metrics.csv"
            )
            print(
                f"Epoch: {epoch}, F1: {val_metrics['class_label/f1']:.4f}, Loss: {train_loss:.4f}, Val_Loss:{val_loss:.4f}"
            )

            loss_arr.append(train_loss)
            val_loss_arr.append(val_metrics['loss'])

            self._early_stopping_step(val_loss, min_loss)
            if  val_loss < min_loss:
                min_loss = val_loss

            if self._should_early_stop():
                print(f"Finishing training early at epoch: {epoch}")
                break

        self._models[target_code] = model

        return loss_arr, val_loss_arr

    def test(self, target_code: int) -> TestResults:
        loss_class: _Loss = build_loss(self._cfg.training)
        loss_domain: _Loss = build_loss(self._cfg.training)

        metrics, results = self._evaluate(
            target_code=target_code,
            model=self._models[target_code],
            loss_func_class=loss_class,
            loss_func_domain=loss_domain,
            sim_dataloader=cast(CombinedDataLoader[MCBatchItem, MCBatchItemOut], self._sim_test_dl),
            exp_dataloader=cast(CombinedDataLoader[ExpBatchItem, ExpBatchItemOut], self._exp_test_dl),
        )

        self._log_results(metrics, target_code=target_code, csv_name=f"test_metrics.csv")
        self._save_test_results(results["class"], target_code=target_code, filename="test_prediction_results_class_labels")
        self._save_test_results(results["domain"], target_code=target_code, filename="test_prediction_results_class_domain")

        print(metrics)
        return metrics, results

    def _train_one_epoch(
            self,
            target_code: int,
            model: torch.nn.Module,
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
        for i in tqdm(range(loader_len), desc="Training DANN", total=loader_len):
            sim_inputs, sim_targets, sim_gids, sim_unstandardized = next(sim_iter)
            sim_gid: GroupID = cast(GroupID, sim_gids[0])
            exp_inputs, exp_gids, exp_unstandardized = next(exp_iter)
            exp_gid: GroupID = cast(GroupID, exp_gids[0])

            sim_inputs = sim_inputs.to(self._cfg.training.device)
            exp_inputs = exp_inputs.to(self._cfg.training.device)

            sim_binary_targets = (sim_targets == target_code).type(torch.float).to(self._cfg.training.device)
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
                    target_code=target_code,
                    step=i,
                    csv_name="training_loss.csv"
                )
                loss_run_sum = 0

        # No epoch number here, order of logged results must be sufficient
        for gid, gid_losses in group_losses.items():
            self._log_results(
                metrics = {
                    "gid": gid,
                    "mean_loss_epoch": gid_losses.mean(),
                    "gid_count": len(gid_losses),
                },
                target_code=target_code,
                csv_name="sim_gid_losses.csv",
                offline=True,
            )

        return final_loss / count

    # TODO: clean up this---for sure common behaviour can be extracted
    def _evaluate(
            self,
            target_code: int,
            model: torch.nn.Module,
            loss_func_class: _Loss,
            loss_func_domain: _Loss,
            sim_dataloader: CombinedDataLoader[MCBatchItem, MCBatchItemOut],
            exp_dataloader: CombinedDataLoader[ExpBatchItem, ExpBatchItemOut],
    ) -> tuple[dict, dict]:
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

            sim_binary_targets = (sim_targets == target_code).type(torch.float).to(self._cfg.training.device)

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

        val_loss = val_loss / count

        binary_targets = final_class_results["targets"] == target_code
        val_f1, val_prec, val_rec, var_thres = maximize_f1(binary_targets, final_class_results["predictions"])

        final_domain_results = {
            "inputs": np.array(domain_results["inputs"]).squeeze(),
            "targets": np.array(domain_results["targets"], dtype=np.float32).squeeze(),
            "predictions": np.array(domain_results["predictions"]).squeeze(),
            "gids": np.array(domain_results["gids"]).squeeze(),
            **{k: np.array(v).squeeze() for k, v in domain_results["unstandardized"].items()}
        }
        dom_val_f1, dom_val_prec, dom_val_rec, dom_var_thres = maximize_f1(final_domain_results["targets"], final_domain_results["predictions"])

        # TODO: clean up return value
        return {
            "class_label/f1": val_f1,
            "class_label/precision": val_prec,
            "class_label/recall": val_rec,
            "class_label/optimal_threshold": var_thres,
            "loss": val_loss,
            "domain/f1": dom_val_f1,
            "domain/precision": dom_val_prec,
            "domain/recall": dom_val_rec,
            "domain/optimal_threshold": dom_var_thres,
        }, {
                "class": final_class_results,
                "domain": final_domain_results,
        }

