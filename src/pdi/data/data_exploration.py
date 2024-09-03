import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from pdi.constants import PARTICLES_DICT
from pdi.data.constants import TARGET_COLUMN
from pdi.data.types import InputTarget, Split, Additional
from pdi.data.detector_helpers import detector_unmask
from pdi.data.utils import GroupedDataPreparation


def plot_particle_distribution(
    target_code: int,
    prep: GroupedDataPreparation,
    splits: list[Split],
    x_axis: str,
    name: str = None,
    save_dir: str = None,
):
    groups = prep.grouped_it
    bins = np.linspace(0, 5, 100)
    for split in splits:
        labels = []
        for gid, group in groups.items():
            targets = group[split][InputTarget.TARGET]
            targets = pd.DataFrame(targets, columns=[TARGET_COLUMN])
            group_add = prep.grouped_add[gid][split][Additional[x_axis]]
            x_values = group_add[targets[TARGET_COLUMN] == target_code]
            counts, bins = np.histogram(x_values, bins=bins)
            detectors = detector_unmask(gid)
            detectors = [d.name for d in detectors]
            detectors_label = ",".join(detectors)
            plt.plot(bins[:-1], counts)
            labels.append(f"Available detectors: {detectors_label}")
        plt.title(f"Distribution of {PARTICLES_DICT[target_code]}")
        plt.xlabel(x_axis)
        plt.ylabel("Count")
        plt.legend(labels, loc="upper right")
        plt.savefig(f"{save_dir}/{name}_{split}.png")
        plt.show()
        plt.close()


def plot_cor_matrix(df: pd.DataFrame, title: str, save_dir: str):
    _, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(
        corr,
        mask=np.zeros_like(corr, dtype=np.bool),
        cmap=sns.diverging_palette(220, 10, as_cmap=True),
        square=True,
        ax=ax,
        center=0,
    )

    plt.title(f"Correlation matrix {title}")

    plt.savefig(os.path.join(save_dir, title))


# Feature importance
def explain_model(
    model, df, batch_size: int = 16, batches: int = 50, hide_progress_bars: bool = False
):
    result = None
    rows_count = 0
    for j in range(batches):
        batch = df[j * batch_size : (j + 1) * batch_size]
        rows = len(batch.index)

        print(f"Batch {j + 1} of {batches}")
        if rows == 0:
            print("End of data")
            break
        rows_count += rows

        batch = batch.to_numpy(dtype=np.float32)

        not_nan_count = np.count_nonzero(~np.isnan(batch))
        feat_count = np.count_nonzero(~np.isnan(batch[0]))

        explainer = shap.KernelExplainer(model, batch)

        if not_nan_count % feat_count != 0:
            print(f"Feats: {feat_count}, Nans: {not_nan_count}")
            print("Bug report:")
            print(batch)

        shap_values = explainer(batch, silent=hide_progress_bars)
        if result is None:
            result = shap_values
        else:
            result.values = np.concatenate((result.values, shap_values.values), axis=0)
            result.data = np.concatenate((result.data, shap_values.data), axis=0)

    print("Explanation finished.")
    print("Number of entries explained: ", rows_count)
    return result, rows_count


def plot_and_save_beeswarm(result, save_dir: str, file_name: str, title: str):
    shap.plots.beeswarm(result[:, :, 0], max_display=19, show=False)
    plt.title(title)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, file_name), bbox_inches="tight")
    plt.show()
