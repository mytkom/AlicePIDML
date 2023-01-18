""" This module contains the functions used to evaluate trained models.

Functions
------
validate_one_epoch
    Obtains model predictions on the validation set and calculates loss and prediction metrics.
get_predictions_and_data
    Obtains model predictions on the test set and saves additional data used for analysis.
get_interval_purity_efficiency
    Calculates purity and efficiency of the classifier depending on the momentum of the particle.
calculate_precision_recall
    Calculates precision and recall based on the number of true positives, predicted positives and positives.

"""

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
    """validate_one_epoch calculates loss and prediction metrics of the model on the validation set.

    Args:
        model (torch.nn.Module): validated model
        target_code (int): target particle that the model is trained to predict
        validation_loader (DataLoader[tuple[Tensor, Tensor, Dict[str, Tensor]]]): validation set dataloader
        device (torch.device): torch device to use for processing the data. Model has to already be on this device
        loss_fun (Callable[[Tensor, Tensor], Tensor]): function used to calculate loss

    Returns:
        tuple[float, ...]: validation loss, f1 score, precision, recall and threshold for selecting the positive class
    """

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

    val_f1, val_prec, val_rec, var_thres = _maximize_f1(targets, predictions)

    return val_loss, val_f1, val_prec, val_rec, var_thres


def _maximize_f1(targets: NDArray[np.float32],
                 predictions: NDArray[np.float32]) -> tuple[float, ...]:
    precision, recall, thresholds = precision_recall_curve(
        targets, predictions)
    f1 = 2 * precision * recall / (precision + recall + np.finfo(float).eps)
    argmax = np.argmax(f1)
    return f1[argmax], precision[argmax], recall[argmax], thresholds[argmax]


def get_predictions_and_data(
    model: torch.nn.Module,
    test_loader: DataLoader[tuple[Tensor, Tensor, Dict[str, Tensor]]],
    device: torch.device,
) -> tuple[NDArray[np.float32], NDArray[np.float32], Dict[
        str, NDArray[np.float32]]]:
    """get_predictions_and_data obtains model predictions on the test set and saves additional data used for analysis.

    Args:
        model (torch.nn.Module): tested model
        test_loader (DataLoader[tuple[Tensor, Tensor, Dict[str, Tensor]]]): test set dataloader
        device (torch.device): model device

    Returns:
        tuple[NDArray[np.float32], NDArray[np.float32], Dict[ str, NDArray[np.float32]]]: target array, prediction array, and a dictionary of additional data
    """

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


def _get_interval_predictions(
    binary_targets: NDArray[np.int32],
    predictions: NDArray[np.float32],
    momentum: NDArray[np.float32],
    intervals: list[tuple[float, float]],
    threshold: float,
) -> tuple[list[NDArray[np.int32]], list[NDArray[np.int32]], list[float]]:
    targets_intervals = []
    selected_intervals = []
    momenta = []

    for (p_min, p_max) in intervals:
        indices = (momentum < p_max) & (momentum >= p_min)
        targets_intervals.append(binary_targets[indices])
        selected_intervals.append(predictions[indices] > threshold)
        momenta.append((p_min + p_max) / 2)

    return targets_intervals, selected_intervals, momenta


def get_interval_purity_efficiency(
    binary_targets: NDArray[np.int32],
    predictions: NDArray[np.float32],
    momentum: NDArray[np.float32],
    intervals: list[tuple[float, float]],
    threshold: float,
) -> tuple[list[float], list[float], pd.DataFrame, list[float]]:
    """get_interval_purity_efficiency calculates purity and efficiency metrics at different momentum intervals.

    Args:
        binary_targets (NDArray[np.int32]): array of positive and negative classes
        predictions (NDArray[np.float32]): array of predictions in range (0, 1)
        momentum (NDArray[np.float32]): array of momentum values
        intervals (list[tuple[float, float]]): momentum intevals
        threshold (float): threshold for selecting the positive class

    Returns:
        tuple[list[float], list[float], pd.DataFrame, list[float]]: 
            array of purities, array of efficiencies, metric confidence intervals, middle values of each interval
    """

    targets_intervals, selected_intervals, avg_momenta = _get_interval_predictions(
        binary_targets, predictions, momentum, intervals, threshold)

    purities_p_plot = []
    efficiencies_p_plot = []
    confidence_intervals = pd.DataFrame()

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
    """calculate_precision_recall

    Args:
        tp (int): number of true positives
        pp (int): number of predicted positives
        p (int): number of actual positives

    Returns:
        tuple[float, float, tuple[float, float], tuple[float, float]]: 
            precision, recall, precision confidence interval, recall confidence interval
    """
    eps = float(np.finfo(float).eps)
    precision = tp / (pp + eps)
    recall = tp / (p + eps)
    precision_ci = _calculate_ci(tp, pp)
    recall_ci = _calculate_ci(tp, p)

    return precision, recall, precision_ci, recall_ci


def _calculate_ci(tp: int, n: int) -> tuple[float, float]:
    #Estimate confidence interval for Bernoulli p
    eps = float(np.finfo(float).eps)

    p_hat = float(tp) / (n + eps)
    if tp > 0:
        ci_diff = np.sqrt((tp**2) / (n**3 + eps) + 1 / (n + eps))
    else:
        ci_diff = 0
    return p_hat - ci_diff, p_hat + ci_diff
