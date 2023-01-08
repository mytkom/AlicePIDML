from typing import Callable, Dict

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.metrics import precision_recall_curve
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from pdi.data.constants import GROUP_ID_KEY


def validate_one_epoch(
    model: torch.nn.Module,
    target_code: int,
    validation_loader: DataLoader[tuple[Tensor, Tensor, Dict[str, Tensor]]],
    device: torch.device,
    loss_fun: Callable[[Tensor, Tensor], Tensor],
) -> tuple[float, ...]:
    predictions_list = []
    targets_list = []
    val_loss = 0.0

    model.eval()
    with torch.no_grad():
        for input_data, target, data_dict in tqdm(validation_loader):
            input_data = input_data.to(device)
            binary_target = (target == target_code).type(
                torch.float).to(device)

            group_id = data_dict.get(GROUP_ID_KEY)
            if group_id is None:
                out = model(input_data)
            else:
                out = model(input_data, group_id)

            val_loss += loss_fun(out, binary_target).item()
            predict_target = torch.sigmoid(out)

            targets_list.extend(binary_target.cpu().numpy())
            predictions_list.extend(predict_target.cpu().detach().numpy())

    predictions = np.array(predictions_list).squeeze()
    targets = np.array(targets_list, dtype=np.float32).squeeze()

    val_f1, val_prec, val_rec, var_thres = maximize_f1(targets, predictions)

    return val_loss, val_f1, val_prec, val_rec, var_thres


def maximize_f1(targets: NDArray[np.float32],
                predictions: NDArray[np.float32]) -> tuple[float, ...]:
    precision, recall, thresholds = precision_recall_curve(
        targets, predictions)
    f1 = 2 * precision * recall / (precision + recall + np.finfo(float).eps)
    argmax = np.argmax(f1)
    return f1[argmax], precision[argmax], recall[argmax], thresholds[argmax]


def get_predictions_and_data(
    model: torch.nn.Module, test_loader: DataLoader[tuple[Tensor, Tensor,
                                                          Dict[str, Tensor]]],
    device: torch.device
) -> tuple[NDArray[np.float32], NDArray[np.float32], Dict[
        str, NDArray[np.float32]]]:
    predictions = []
    targets = []
    model.eval()

    with torch.no_grad():
        additional_data: Dict[str, list[NDArray[np.float32]]] = {}
        for input_data, target, data_dict in tqdm(test_loader):

            input_data = input_data.to(device)
            group_id = data_dict.get(GROUP_ID_KEY)
            if group_id is None:
                out = model(input_data)
            else:
                out = model(input_data, group_id)
            predict_target = torch.sigmoid(out)

            targets.extend(target.numpy())
            predictions.extend(predict_target.cpu().detach().numpy())

            for k, v in data_dict.items():
                if k not in additional_data:
                    additional_data[k] = []
                additional_data[k].extend(v.numpy())

    predictions_arr = np.array(predictions).squeeze()
    targets_arr = np.array(targets, dtype=np.float32).squeeze()
    add_d = {k: np.array(v).squeeze() for k, v in additional_data.items()}
    return targets_arr, predictions_arr, add_d


def get_interval_predictions(
    binary_targets: NDArray[np.int32], predictions: NDArray[np.float32],
    p: NDArray[np.float32], thresholds: NDArray[np.float32]
) -> tuple[list[NDArray[np.int32]], list[NDArray[np.int32]], list[float]]:
    targets_intervals = []
    selected_intervals = []
    momenta = []

    for (p_min, p_max), threshold in thresholds:
        indices = (p < p_max) & (p >= p_min)
        targets_intervals.append(binary_targets[indices])
        selected_intervals.append(predictions[indices] > threshold)
        momenta.append((p_min + p_max) / 2)

    return targets_intervals, selected_intervals, momenta


def get_interval_purity_efficiency(
    binary_targets: NDArray[np.int32], predictions: NDArray[np.float32],
    momentum: NDArray[np.float32], thresholds: NDArray[np.float32]
) -> tuple[list[float], list[float], pd.DataFrame, list[float]]:
    purities_p_plot = []
    efficiencies_p_plot = []
    confidence_intervals = pd.DataFrame()

    targets_intervals, selected_intervals, avg_momenta = get_interval_predictions(
        binary_targets, predictions, momentum, thresholds)

    for targets, selected in zip(targets_intervals, selected_intervals):
        tp = int(np.sum(selected & targets))
        p = int(np.sum(targets))
        pp = int(np.sum(selected))
        purity, efficiency, p_ci, e_ci = calculate_precision_recall(tp, pp, p)
        confidence_intervals = pd.concat(
            [
                confidence_intervals,
                pd.DataFrame([{
                    "purity_lower": p_ci[0],
                    "purity_upper": p_ci[1],
                    "efficiency_lower": e_ci[0],
                    "efficiency_upper": e_ci[1],
                }])
            ],
            ignore_index=True,
        )
        purities_p_plot.append(purity)
        efficiencies_p_plot.append(efficiency)

    return purities_p_plot, efficiencies_p_plot, confidence_intervals, avg_momenta


def calculate_precision_recall(
        tp: int, pp: int, p: int
) -> tuple[float, float, tuple[float, float], tuple[float, float]]:
    eps = float(np.finfo(float).eps)
    precision = tp / (pp + eps)
    recall = tp / (p + eps)
    precision_ci = calculate_ci(tp, pp)
    recall_ci = calculate_ci(tp, p)

    return precision, recall, precision_ci, recall_ci


def calculate_ci(tp: int, n: int) -> tuple[float, float]:
    """Estimates confidence interval for Bernoulli p
    Args:
      tp: number of positive outcomes, TP in this case
      n: number of attemps, TP+FP for Precision, TP+FN for Recall
      alpha: confidence level
    Returns:
      tuple[float, float]: lower and upper bounds of the confidence interval
    """
    eps = float(np.finfo(float).eps)

    p_hat = float(tp) / (n + eps)
    if tp > 0:
        ci_diff = np.sqrt((tp**2) / (n**3 + eps) + 1 / (n + eps))
    else:
        ci_diff = 0
    return p_hat - ci_diff, p_hat + ci_diff
