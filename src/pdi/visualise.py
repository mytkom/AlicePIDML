"""This module contains the functions used to plot comparison graphs.

Functions
------
plot_precision_recall_comparison
    Plots precision-recall curves for different methods.
plot_purity_comparison
    Plots purity (precision) graph depending on the momentum of the particles.
plot_efficiency_comparison
    Plots efficiency (recall) graph depending on the momentum of the particles.
"""

import os
import matplotlib.pyplot as plt
from numpy import linspace
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing import Iterator

from pdi.constants import TARGET_CODE_TO_PART_NAME
from pdi.evaluate import get_interval_purity_efficiency
from pdi.results_and_metrics import TestResults


def plot_precision_recall_comparison(
    test_metrics: dict[str, TestResults], save_dir=None
) -> Figure:
    fig = plt.figure(figsize=(10, 6))
    target_codes = set()
    precision_recall_results: list[dict] = []
    for model_name, test_result in test_metrics.items():
        binary_targets = test_result.targets == test_result.test_metrics.target_code
        target_codes.add(test_result.test_metrics.target_code)
        precision, recall, thresholds = precision_recall_curve(
            binary_targets, test_result.predictions
        )

        # Add to results
        entry: dict = {}
        entry["model"] = model_name
        entry["precision"] = precision
        entry["recall"] = recall
        entry["thresholds"] = thresholds
        precision_recall_results.append(entry)

        # Plot
        plt.plot(recall, precision, label=model_name)

    if len(target_codes) > 1:
        raise AttributeError("Cannot compare results of different target_code!")
    target_code: int = target_codes.pop()

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall curve for {TARGET_CODE_TO_PART_NAME[target_code]}")
    plt.legend(loc="lower left")
    plt.grid()
    plt.close()

    # Save plot and results if save_dir was provided
    if save_dir:
        plt.savefig(os.path.join(save_dir, "precision-recall.png"))
        df = pd.DataFrame(precision_recall_results)
        df.to_csv(os.path.join(save_dir, "precision-recall.csv"), index=False)

    return fig


PT_LINSPACE = linspace(0, 5, 20)
PT_INTERVALS = list(zip(PT_LINSPACE[:-1], PT_LINSPACE[1:]))


def plot_metrics_vs_pt_comparison(
    test_metrics: dict[str, TestResults],
    pt: NDArray,
    pt_intervals=PT_INTERVALS,
    save_dir: str | None = None,
) -> Iterator[tuple[Figure, str]]:
    """
    Plots purity, efficiency, and F1 score vs pT for multiple models.

    Parameters
    ----------
    test_metrics : dict[str, TestResults]
        Dictionary containing test results for different models.
    pt : NDArray
        Array of transverse momentum values.
    pt_intervals : list[tuple], optional
        List of pT intervals, by default PT_INTERVALS.
    save_dir : str, optional
        Directory to save a CSV file, by default None.

    Returns
    -------
    Iterator[tuple[Figure, str]]
        An iterator yielding tuples, where each tuple contains:
        - A matplotlib Figure object for the plot.
        - A string representing the filename for the plot.

    Raises
    ------
    AttributeError
        If the test results contain multiple target codes, as comparison is only valid for a single target code.

    Notes
    -----
    - The function calculates purity, efficiency, and F1 score for each model over specified pT intervals.
    - It generates and optionally saves a CSV file containing the metrics.
    """
    target_codes = set()
    df = pd.DataFrame()

    # Prepare data for all models
    for model_name, test_result in test_metrics.items():
        target_codes.add(test_result.target_code)
        targets = test_result.targets == test_result.test_metrics.target_code
        selected = (
            test_result.predictions >= test_result.test_metrics.threshold
        ).astype("int")

        purities_pt_plot, efficiencies_pt_plot, confidence_intervals, momenta_avg = (
            get_interval_purity_efficiency(targets, selected, pt, pt_intervals)
        )

        # Store metrics in the DataFrame
        purities_pt_plot = np.array(purities_pt_plot)
        efficiencies_pt_plot = np.array(efficiencies_pt_plot)
        f1_scores = (2 * purities_pt_plot * efficiencies_pt_plot) / (
            purities_pt_plot + efficiencies_pt_plot + np.finfo(float).eps
        )

        df[f"Purity {model_name}"] = pd.Series(purities_pt_plot)
        df[f"Efficiency {model_name}"] = pd.Series(efficiencies_pt_plot)
        df[f"F1 {model_name}"] = pd.Series(f1_scores)
        df[f"Pt {model_name}"] = pd.Series(momenta_avg)

    if len(target_codes) > 1:
        raise AttributeError("Cannot compare results of different target_code!")
    target_code: int = target_codes.pop()

    def plot(metric_name, y_values, y_label, title_suffix, file_suffix):
        fig = plt.figure(figsize=(10, 6))
        for model_name in test_metrics.keys():
            plt.plot(
                df[f"Pt {model_name}"],
                df[f"{metric_name} {model_name}"],
                "o-",
                label=model_name,
            )
        plt.xlabel("p_t (GeV/c)")
        plt.ylabel(y_label)
        plt.ylim([0, 1.1])
        plt.title(
            f"{y_label} vs p_t ({TARGET_CODE_TO_PART_NAME[target_code]}) {title_suffix}"
        )
        plt.legend(loc="lower left")
        plt.grid()
        plt.close()

        return fig, f"{file_suffix}-vs-pt.png"

    yield plot("Purity", df.filter(like="Purity").values, "Purity", "", "purity")
    yield plot(
        "Efficiency",
        df.filter(like="Efficiency").values,
        "Efficiency",
        "",
        "efficiency",
    )
    yield plot("F1", df.filter(like="F1").values, "F1 Score", "", "f1")

    if save_dir:
        df.to_csv(os.path.join(save_dir, "metrics-vs-pt.csv"), index=False)


def plot_learning_curve(
    loss: list[float], val_loss: list[float], label: str = None, save_dir: str = None
):
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title(f"Learning curve {label}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Loss", "Val_Loss"])
    plt.savefig(f"{save_dir}/loss_{label}")
    plt.show()
    plt.close()
