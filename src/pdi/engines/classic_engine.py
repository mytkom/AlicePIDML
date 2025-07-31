from typing import Optional, cast
from joblib.pool import np
from numpy.typing import NDArray
from sklearn.metrics import precision_score, recall_score
import torch
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from tqdm import tqdm
from pdi.config import Config
from pdi.data.data_preparation import CombinedDataLoader, MCBatchItem, DataPreparation, MCBatchItemOut
from pdi.data.types import GroupID
from pdi.engines.base_engine import BaseEngine, TestResults, TrainResults
from pdi.evaluate import maximize_f1
from pdi.insertion_strategies import MISSING_DATA_STRATEGIES
from pdi.losses import build_loss
from pdi.models import build_model
from pdi.optimizers import build_optimizer
from pdi.lr_schedulers import build_lr_scheduler

class ClassicEngine(BaseEngine):
    def __init__(self, cfg: Config, target_code: int) -> None:
        super().__init__(cfg, target_code)
        self._data_prep = DataPreparation(cfg.data, cfg.sim_dataset_paths, cfg.seed)

        if self._cfg.model.architecture == "mlp":
            # Deal with missing data inplace on PreparedData object
            self._data_prep.transform_prepared_data(MISSING_DATA_STRATEGIES[cfg.model.mlp.missing_data_strategy])

        (self._train_dl, self._val_dl, self._test_dl) = self._data_prep.create_dataloaders(
            batch_size=self._cfg.training.batch_size,
            num_workers=self._cfg.training.num_workers,
            undersample_missing_detectors=self._cfg.training.undersample_missing_detectors,
            undersample_pions=self._cfg.training.undersample_pions,
        )

        if self._data_prep._is_experimental:
            raise RuntimeError("ClassicEngine got experimental data, it is not suited to handle it!")


    def train(self) -> TrainResults:
        model = build_model(self._cfg.model, group_ids=self._data_prep.get_group_ids())
        if self._cfg.model.pretrained_model_dirpath:
            self._load_model(model, self._cfg.model.pretrained_model_dirpath)
        model.to(self._cfg.training.device)

        pos_weight = None
        # TODO: check if it works as expected, compare results with it and without
        if self._cfg.training.weight_particles_species:
            pos_weight = torch.tensor(self._data_prep.pos_weight(self._target_code)).to(self._cfg.training.device)
        loss: _Loss = build_loss(self._cfg.training, pos_weight=pos_weight)

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
                loss_func=loss,
                dataloader=cast(CombinedDataLoader[MCBatchItem, MCBatchItemOut], self._train_dl),
            )

            # Validation
            val_metrics = self._evaluate(
                model=model,
                loss_func=loss,
                dataloader=cast(CombinedDataLoader[MCBatchItem, MCBatchItemOut], self._val_dl),
            )

            # Threshold for posterior probability to identify as positive
            # it is optimized for f1 metric
            model.thres = torch.tensor(np.array(val_metrics["val/threshold"])).to(self._cfg.training.device)

            val_loss = val_metrics["val/loss"]

            # New learning rate for the next
            scheduler.step()

            loss_arr.append(train_loss)
            val_loss_arr.append(val_loss)

            self._early_stopping_step(
                model=model,
                threshold=val_metrics["val/threshold"],
                val_loss=val_loss,
                min_loss=min_loss,
                val_f1=val_metrics["val/f1"],
                epoch=epoch,
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

        return loss_arr, val_loss_arr

    def test(self, model_dirpath: Optional[str] = None) -> TestResults:
        loss_func: _Loss = build_loss(self._cfg.training)
        model: torch.nn.Module = build_model(self._cfg.model, group_ids=self._data_prep.get_group_ids())
        model, threshold = self._load_model(model, model_dirpath)
        model.to(self._cfg.training.device)

        metrics, results = self._test(
            model=model,
            threshold=threshold,
            loss_func=loss_func,
            dataloader=cast(CombinedDataLoader[MCBatchItem, MCBatchItemOut], self._test_dl),
        )

        self._log_results(metrics, csv_name=f"test_metrics.csv")
        self._save_test_results(results)

        print(metrics)
        return metrics, results

    def _train_one_epoch(self, model: torch.nn.Module, epoch: int, optimizer: Optimizer, loss_func: _Loss, dataloader: CombinedDataLoader[MCBatchItem, MCBatchItemOut]) -> float:
        model.train()
        group_losses: dict[GroupID, NDArray] = {}
        loss_run_sum = 0
        final_loss = 0.0
        count = 0

        loader_len = len(dataloader)

        for i, (input_data, targets, gids, _) in enumerate(tqdm(dataloader), start=1):
            # Constant value for a batch, but cannot obtain single value with pytorch interface
            gid: GroupID = cast(GroupID, int(gids[0]))

            input_data = input_data.to(self._cfg.training.device)
            binary_targets = (targets == self._target_code).type(torch.float).to(self._cfg.training.device)

            with torch.autocast(device_type=self._cfg.training.device, dtype=torch.float16):
                out = model(input_data)
                # output is float16 because linear layers ``autocast`` to float16.
                assert out.dtype is torch.float16
    
                loss = loss_func(out, binary_targets)
                # loss is float32 because ``mse_loss`` layers ``autocast`` to float32.
                assert loss.dtype is torch.float32

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Handle metrics
            loss_run_sum += loss.item()
            final_loss += loss.item()
            count += 1

            # TODO: decide if it is useful or not; should I log_every tens of steps too?
            self._log_results(
                metrics={ "loss": loss.item() },
                step=loader_len*epoch + i,
                csv_name=f"gid_{gid}_training_loss.csv",
                offline=True,
            )
            if gid not in group_losses.keys():
                group_losses[gid] = np.array([])
            np.append(group_losses[gid], loss.item())

            if i % self._cfg.training.steps_to_log == 0:
                self._log_results(
                    metrics={ "loss": loss_run_sum },
                    step=loader_len*epoch + i,
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
                csv_name="gid_losses.csv",
                offline=True,
            )

        return final_loss / count

    def _evaluate(self, model: torch.nn.Module, loss_func: _Loss, dataloader: CombinedDataLoader[MCBatchItem, MCBatchItemOut]) -> dict[str, float]:
        binary_targets = []
        predictions = []
        val_loss = 0.0
        count = 0

        model.eval()
        for _, (input_data, target, gid, _) in enumerate(tqdm(dataloader)):
            input_data = input_data.to(self._cfg.training.device)

            out = model(input_data)

            # loss
            binary_target = (target == self._target_code).type(torch.float).to(self._cfg.training.device)
            loss = loss_func(out, binary_target)
            val_loss += loss.item()
            count += 1

            # save data
            predict_target = torch.sigmoid(out)
            predictions.extend(predict_target.cpu().detach().numpy())
            binary_targets.extend(target.cpu().detach().numpy() == self._target_code)

            del input_data
            del binary_target
            del target
            del gid

        if count == 0:
            count = 1

        val_loss = val_loss / count

        binary_targets = np.array(binary_targets).squeeze()
        predictions = np.array(predictions).squeeze()
        val_f1, val_prec, val_rec, var_thres = maximize_f1(binary_targets, predictions)

        # TODO: make it a structure
        return {
            "val/f1": val_f1,
            "val/precision": val_prec,
            "val/recall": val_rec,
            "val/threshold": var_thres,
            "val/loss": val_loss,
        }

    def _test(self, model: torch.nn.Module, threshold: float, loss_func: _Loss, dataloader: CombinedDataLoader[MCBatchItem, MCBatchItemOut]) -> TestResults:
        predictions = []
        targets = []
        unstandardized_data = {}
        input_data_tensors = []
        val_loss = 0.0
        count = 0

        model.eval()
        for _, (input_data, target, gid, data_dict) in enumerate(tqdm(dataloader)):
            input_data = input_data.to(self._cfg.training.device)

            out = model(input_data)

            # loss
            binary_target = (target == self._target_code).type(torch.float).to(self._cfg.training.device)
            loss = loss_func(out, binary_target)
            val_loss += loss.item()
            count += 1

            # save data
            predict_target = torch.sigmoid(out)
            predictions.extend(predict_target.cpu().detach().numpy())
            targets.extend(target.cpu().detach().numpy())
            input_data_tensors.extend(input_data.cpu().detach().numpy())
            for k, v in data_dict.items():
                if k not in unstandardized_data:
                    unstandardized_data[k] = []
                unstandardized_data[k].extend(v.cpu().detach().numpy())

            del input_data
            del binary_target
            del target
            del gid
            del data_dict

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

        binary_targets = results["targets"] == self._target_code
        binary_predictions = torch.sigmoid(results["predictions"]) >= threshold
        test_precision: float = float(precision_score(binary_targets, binary_predictions))
        test_recall: float = float(recall_score(binary_targets, binary_predictions))
        test_f1 = float((test_precision * test_recall * 2) / (test_precision + test_recall + np.finfo(float).eps))

        return {
            "test/f1": test_f1,
            "test/precision": test_precision,
            "test/recall": test_recall,
        }, results

