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
from pdi.data.data_preparation import CombinedDataLoader, MCBatchItem, GeneralDataPreparation, MCBatchItemOut
from pdi.data.types import GroupID
from pdi.engines.base_engine import BaseEngine, TestResults, TrainResults
from pdi.evaluate import maximize_f1
from pdi.insertion_strategies import MISSING_DATA_STRATEGIES
from pdi.losses import build_loss
from pdi.models import NeuralNetEnsemble, build_model
from pdi.optimizers import build_optimizer
from pdi.lr_schedulers import build_lr_scheduler

class ClassicEngine(BaseEngine):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self._cfg = cfg
        self._models: dict[int, torch.nn.Module] = {}

        self._data_prep = GeneralDataPreparation(cfg.data, cfg.sim_dataset_paths)

        if self._cfg.model.architecture == "MLP":
            # Deal with missing data inplace on PreparedData object
            self._data_prep.transform_prepared_data(MISSING_DATA_STRATEGIES[cfg.model.mlp.missing_data_strategy])

        (self._train_dl, self._val_dl, self._test_dl) = self._data_prep.create_dataloaders(
            batch_size=self._cfg.training.batch_size,
            num_workers=self._cfg.training.num_workers,
            undersample=self._cfg.data.undersample_missing_detectors,
            seed=self._cfg.seed
        )

        if self._data_prep._is_experimental:
            raise RuntimeError("ClassicEngine got experimental data, it is not suited to handle it!")


    def train(self, target_code: int) -> TrainResults:
        if wandb.run is None:
            wandb.init(config=self._cfg.to_dict())
        model = build_model(self._cfg.model, group_ids=self._data_prep.get_group_ids())

        pos_weight = None
        # TODO: check if it works as expected, compare results with it and without
        if self._cfg.data.weight_particles_species:
            pos_weight = torch.tensor(self._data_prep.pos_weight(target_code))
        loss: _Loss = build_loss(self._cfg.training, pos_weight=pos_weight)

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
                loss_func=loss,
                dataloader=cast(CombinedDataLoader[MCBatchItem, MCBatchItemOut], self._train_dl),
            )

            # Validation
            val_metrics, _ = self._evaluate(
                target_code=target_code,
                model=model,
                loss_func=loss,
                dataloader=cast(CombinedDataLoader[MCBatchItem, MCBatchItemOut], self._val_dl),
            )

            # Threshold for posterior probability to identify as positive
            # it is optimized for f1 metric
            model.thres = torch.Tensor(np.array(val_metrics["optimal_threshold"]))

            val_loss = val_metrics["loss"]

            # New learning rate for the next
            scheduler.step()

            # Log validation metrics
            self._log_results(
                metrics = {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_f1": val_metrics["f1"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "val_threshold": val_metrics["optimal_threshold"],
                    "scheduled_lr": scheduler.get_last_lr()[0],
                },
                target_code = target_code,
                csv_name = f"validation_metrics.csv"
            )
            print(
                f"Epoch: {epoch}, F1: {val_metrics['f1']:.4f}, Loss: {train_loss:.4f}, Val_Loss:{val_loss:.4f}"
            )

            loss_arr.append(train_loss)
            val_loss_arr.append(val_loss)

            self._early_stopping_step(val_loss, min_loss)
            if  val_loss < min_loss:
                min_loss = val_loss

            if self._should_early_stop():
                print(f"Finishing training early at epoch: {epoch}")
                break

        self._models[target_code] = model

        return loss_arr, val_loss_arr

    def test(self, target_code: int) -> TestResults:
        loss_func: _Loss = build_loss(self._cfg.training)

        metrics, results = self._evaluate(
            target_code=target_code,
            model=self._models[target_code],
            loss_func=loss_func,
            dataloader=cast(CombinedDataLoader[MCBatchItem, MCBatchItemOut], self._test_dl),
        )

        self._log_results(metrics, target_code=target_code, csv_name=f"test_metrics.csv")
        self._save_test_results(results, target_code=target_code)

        print(metrics)
        return metrics, results

    def _train_one_epoch(self, target_code: int, model: torch.nn.Module, optimizer: Optimizer, loss_func: _Loss, dataloader: CombinedDataLoader[MCBatchItem, MCBatchItemOut]) -> float:
        model.train()
        group_losses: dict[GroupID, NDArray] = {}
        loss_run_sum = 0
        final_loss = 0.0
        count = 0

        for i, (input_data, targets, gids, _) in enumerate(tqdm(dataloader), start=1):
            # Constant value for a batch, but cannot obtain single value with pytorch interface
            gid: GroupID = cast(GroupID, int(gids[0]))

            input_data = input_data.to(self._cfg.training.device)
            binary_targets = (targets == target_code).type(torch.float).to(self._cfg.training.device)
            optimizer.zero_grad()

            out = model(input_data)
            loss = loss_func(out, binary_targets)
            loss.backward()
            optimizer.step()

            # Handle metrics
            loss_run_sum += loss.item()
            final_loss += loss.item()
            count += 1

            # TODO: decide if it is useful or not; should I log_every tens of steps too?
            self._log_results(
                metrics={ "loss": loss.item() },
                target_code=target_code,
                step=i,
                csv_name=f"gid_{gid}_training_loss.csv"
            )
            if gid not in group_losses.keys():
                group_losses[gid] = np.array([])
            np.append(group_losses[gid], loss.item())

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
            if len(gid_losses) == 0:
                continue

            self._log_results(
                metrics = {
                    "gid": gid,
                    "mean_loss_epoch": gid_losses.mean(),
                    "gid_count": len(gid_losses),
                },
                target_code=target_code,
                csv_name="gid_losses.csv",
                offline=True,
            )

        return final_loss / count

    def _evaluate(self, target_code: int, model: torch.nn.Module, loss_func: _Loss, dataloader: CombinedDataLoader[MCBatchItem, MCBatchItemOut]) -> TestResults:
        predictions = []
        targets = []
        unstandardized_data = {}
        input_data_tensors = []
        val_loss = 0.0
        count = 0

        model.eval()
        for input_data, target, gid, data_dict in tqdm(dataloader):
            targets.extend(target.numpy())
            input_data_tensors.extend(input_data.numpy())
            for k, v in data_dict.items():
                if k not in unstandardized_data:
                    unstandardized_data[k] = []
                unstandardized_data[k].extend(v.numpy())

            input_data = input_data.to(self._cfg.training.device)

            out = model(input_data)

            # loss
            binary_target = (target == target_code).type(torch.float).to(self._cfg.training.device)
            loss = loss_func(out, binary_target)
            val_loss += loss.item()
            count += 1

            # save data
            predict_target = torch.sigmoid(out)
            predictions.extend(predict_target.cpu().detach().numpy())


        if count == 0:
            count = 1

        # TODO: maybe it should be a DataFrame?
        # Final data, predicted posterior probabilities and their corresponding input tensors,
        # target codes and unstandardized columns
        results = {
            "inputs": np.array(input_data_tensors).squeeze(),
            "targets": np.array(targets, dtype=np.float32).squeeze(),
            "predictions": np.array(predictions).squeeze(),
            **{k: np.array(v).squeeze() for k, v in unstandardized_data.items()}
        }
        val_loss = val_loss / count

        binary_targets = results["targets"] == target_code
        val_f1, val_prec, val_rec, var_thres = maximize_f1(binary_targets, results["predictions"])

        return {
            "f1": val_f1,
            "precision": val_prec,
            "recall": val_rec,
            "optimal_threshold": var_thres,
            "loss": val_loss,
        }, results

