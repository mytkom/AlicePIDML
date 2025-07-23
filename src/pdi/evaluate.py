""" This module contains the functions used to evaluate trained models.

Functions
------
get_predictions_loss_and_data
validate_model
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

from pdi.data.constants import PART_DICT

P_CUT=0.5

def get_nsigma_predictions_data(
    dataloader: DataLoader[tuple[Tensor, Tensor, Dict[str, Tensor]]],
    target_code: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], float]:
    """get_nsigma_predictions_data returns all information in a dataloader in combined numpy arrays.
    The nsigma method as presented here is used always on a dataset including incomplete examples.

    Args:
        dataloader (DataLoader[tuple[Tensor, Tensor, Dict[str, Tensor]]])
        target_code (int, optional)

    Returns:
        tuple[NDArray[np.float32], NDArray[np.float32], Dict[Additional, NDArray[np.float32]], NDArray[np.float64]]:
            array of predictions, array of targets, dictionary of additional data for analysis, array of input data
    """
    predictions = []
    targets = []
    additional_data = {}
    input_data_tensors = []

    nsigma_tpc_col = "fTPCNSigma" + PART_DICT[abs(target_code)]
    nsigma_tof_col = "fTOFNSigma" + PART_DICT[abs(target_code)]

    for input_data, target, data_dict in tqdm(dataloader):
        tpc = abs(data_dict[nsigma_tpc_col])
        nsigmas = (
            np.sqrt(
                pow(data_dict[nsigma_tpc_col], 2) + pow(data_dict[nsigma_tof_col], 2)
            )
            .where(data_dict["fP"] > P_CUT, tpc)
            .where(~data_dict["fBeta"].isnan(), tpc)
        )
        # where puts value from second argument when condition is false

        predictions.extend(nsigmas.cpu().detach().numpy())
        targets.extend(target.numpy())
        input_data_tensors.extend(input_data.cpu().numpy())
        for k, v in data_dict.items():
            if k not in additional_data:
                additional_data[k] = []
            additional_data[k].extend(v.numpy())

    predictions_arr = np.array(predictions, dtype=np.float32).squeeze()
    targets_arr = np.array(targets, dtype=np.float32).squeeze()
    dict_arr = {k: np.array(v).squeeze() for k, v in additional_data.items()}

    return predictions_arr, targets_arr, dict_arr, np.array(input_data_tensors).squeeze()

def maximize_f1(
    binary_targets: NDArray[np.float32], predictions: NDArray[np.float32]
) -> tuple[float, ...]:
    precision, recall, thresholds = precision_recall_curve(binary_targets, predictions)
    f1 = 2 * precision * recall / (precision + recall + np.finfo(float).eps)
    argmax = np.argmax(f1)
    return f1[argmax], precision[argmax], recall[argmax], thresholds[argmax]


def _split_particles_intervals(
    binary_targets: NDArray[np.int32],
    selected: NDArray[np.int32],
    momentum: NDArray[np.float32],
    intervals: list[tuple[float, float]],
) -> tuple[list[NDArray[np.int32]], list[NDArray[np.int32]], list[float]]:
    targets_intervals = []
    selected_intervals = []
    momenta = []

    for p_min, p_max in intervals:
        indices = (momentum < p_max) & (momentum >= p_min)
        targets_intervals.append(binary_targets[indices])
        selected_intervals.append(selected[indices])
        momenta.append((p_min + p_max) / 2)

    return targets_intervals, selected_intervals, momenta


def get_interval_purity_efficiency(
    binary_targets: NDArray[np.int32],
    selected: NDArray[np.int32],
    momentum: NDArray[np.float32],
    intervals: list[tuple[float, float]],
) -> tuple[list[float], list[float], pd.DataFrame, list[float]]:
    """get_interval_purity_efficiency calculates purity and efficiency metrics at different momentum intervals.

    Args:
        binary_targets (NDArray[np.int32]): array of positive and negative targets
        selected (NDArray[np.float32]): array of predicted classes
        momentum (NDArray[np.float32]): array of momentum values
        intervals (list[tuple[float, float]]): momentum intevals

    Returns:
        tuple[list[float], list[float], pd.DataFrame, list[float]]:
            array of purities, array of efficiencies, confidence intervals, middle values of each interval
    """

    targets_intervals, selected_intervals, avg_momenta = _split_particles_intervals(
        binary_targets, selected, momentum, intervals
    )

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
                pd.DataFrame(
                    [
                        {
                            "purity_lower": p_ci[0],
                            "purity_upper": p_ci[1],
                            "efficiency_lower": e_ci[0],
                            "efficiency_upper": e_ci[1],
                        }
                    ]
                ),
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
    # Estimate confidence interval for Bernoulli p
    eps = float(np.finfo(float).eps)

    p_hat = float(tp) / (n + eps)
    if tp > 0:
        ci_diff = np.sqrt((tp**2) / (n**3 + eps) + 1 / (n + eps))
    else:
        ci_diff = 0
    return p_hat - ci_diff, p_hat + ci_diff
