import base64
import os
from io import BytesIO
from itertools import combinations
from typing import Iterator, Literal, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from IPython.display import HTML
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from pdi.constants import TARGET_CODE_TO_PART_NAME
from pdi.data.constants import TARGET_COLUMN
from pdi.data.detector_helpers import detector_unmask
from pdi.data.types import Additional, InputTarget, Split
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
        plt.title(f"Distribution of {TARGET_CODE_TO_PART_NAME[target_code]}")
        plt.xlabel(x_axis)
        plt.ylabel("Count")
        plt.legend(labels, loc="upper right")
        plt.savefig(f"{save_dir}/{name}_{split}.png")
        plt.show()
        plt.close()

def plot_cor_matrix(df: pd.DataFrame, title: str) -> Figure:
    """
    Plot the correlation matrix of a DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame for which the correlation matrix will be plotted.
        title (str): The title of the correlation matrix plot.

    Returns:
        matplotlib.figure.Figure: The Figure object containing the correlation matrix plot.
    """
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

    return ax.figure

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

def generate_figure_thumbnails_from_iterator(
    figure_iter: Iterator[Tuple[Figure, str]],
    save_path: str,
    thumbnail_width: int = 300
) -> HTML:
    """
    Accepts a generator yielding (fig, filename) pairs, saves them, and returns thumbnail HTML.

    Parameters:
    - figure_iter: generator yielding (Figure, filename) tuples.
    - thumbnail_width: width of the thumbnail in pixels.

    Returns:
    - IPython.display.HTML gallery.
    """
    thumbnails = []

    for fig, filename in figure_iter:
        filepath = os.path.abspath(os.path.join(save_path, filename))
        # Save full-size figure
        fig.savefig(filepath, dpi=300, bbox_inches='tight')

        # Save to buffer for thumbnail
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=50)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        fig.clf()
        plt.close(fig)

        # Create thumbnail HTML
        html = (
            f'<a href="{filepath}" target="_blank" style="margin: 5px; display:inline-block;">'
            f'<img src="data:image/png;base64,{b64}" width="{thumbnail_width}px"></a>'
        )
        thumbnails.append(html)

    return HTML("<div style='display:flex; flex-wrap: wrap; gap: 10px;'>" + "\n".join(thumbnails) + "</div>")

def plot_feature_distributions_by_condition(
    data: pd.DataFrame,
    features: list[str],
    group_labels: list[str],
    group_conditions: list[pd.Series],
    plot_type: Literal["kde", "hist", "boxplot", "violinplot", "ecdf"] = "kde",
    log_y: bool = False,
    title_template: str = "Distribution of {feature} by Group",
    hist_bins: int = 100,
) -> Iterator[Tuple[Figure, str]]:
    """
    Generator that yields matplotlib figures comparing distributions
    of features across custom groups defined by conditions.

    Parameters:
    - data: DataFrame containing the dataset.
    - features: List of column names to plot.
    - group_labels: Labels for each condition group.
    - group_conditions: List of boolean Series to filter data into groups.
    - plot_type: One of ['kde', 'hist', 'boxplot', 'violinplot', 'ecdf'].
    - log_y: Whether to use log scale on Y axis.
    - title_template: Template for plot titles (use {feature} as placeholder).
    """
    palette = sns.color_palette("husl", len(group_labels))

    for feature in features:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(title_template.format(feature=feature), fontsize=14)

        if log_y:
            ax.set_yscale('log')

        if plot_type == "kde":
            for (label, condition), color in zip(zip(group_labels, group_conditions), palette):
                sns.kdeplot(
                    data[condition][feature].dropna(),
                    label=label,
                    fill=True,
                    common_norm=True,
                    color=color,
                    ax=ax
                )
            ax.set_xlabel(feature)
            ax.set_ylabel("Density" if not log_y else "Log Density")

        elif plot_type == "hist":
            for (label, condition), color in zip(zip(group_labels, group_conditions), palette):
                ax.hist(
                    data[condition][feature].dropna(),
                    bins=hist_bins,
                    alpha=0.7,
                    label=label,
                    color=color
                )
            ax.set_xlabel(feature)
            ax.set_ylabel("Count" if not log_y else "Log Count")

        elif plot_type in ["boxplot", "violinplot"]:
            plot_data = []
            for label, condition in zip(group_labels, group_conditions):
                temp_df = pd.DataFrame({
                    feature: data[condition][feature].dropna(),
                    "group": label
                })
                plot_data.append(temp_df)
            plot_data = pd.concat(plot_data, ignore_index=True)

            if plot_type == "boxplot":
                sns.boxplot(x="group", y=feature, data=plot_data, hue="group", palette=palette, ax=ax)
            else:
                sns.violinplot(x="group", y=feature, data=plot_data, palette=palette, cut=0, ax=ax)

            ax.set_xlabel("Group")
            ax.set_ylabel(feature)

        elif plot_type == "ecdf":
            for (label, condition), color in zip(zip(group_labels, group_conditions), palette):
                sns.ecdfplot(
                    data[condition][feature].dropna(),
                    label=label,
                    color=color,
                    ax=ax
                )
            ax.set_xlabel(feature)
            ax.set_ylabel("ECDF" if not log_y else "Log ECDF")
            ax.grid(True, linestyle='--', alpha=0.7)

        if plot_type in ["kde", "hist", "ecdf"]:
            ax.legend(fontsize=10, framealpha=1)

        ax.tick_params(labelsize=10)
        fig.tight_layout()
        file_name = f"{feature}_{plot_type}{'_logY' if log_y else ''}.png"

        yield fig, file_name

def plot_feature_histogram2d_combinations(
    data: pd.DataFrame,
    features: list[str],
    title_template: str = "Histogram2D of {feature1} vs {feature2}",
    bins: int = 200,
    cmap: str = "viridis",
    density_threshold: float = 1e-3,
) -> Iterator[Tuple[Figure, str]]:
    """
    Generator that yields 2D histograms of combinations of features with dynamic axis ranges
    and logarithmic z-axis.

    Parameters:
    - data: DataFrame containing the dataset.
    - features: List of column names to plot.
    - title_template: Template for plot titles (use {feature1} and {feature2} as placeholders).
    - bins: Number of bins for the histogram.
    - cmap: Colormap for the histogram.
    - density_threshold: Minimum density value to include in the axis range.

    Returns:
    - Iterator of tuples containing the matplotlib Figure object and the file name.
    """
    for feature1, feature2 in combinations(features, 2):
        # Compute the 2D histogram
        hist, x_edges, y_edges = np.histogram2d(
            data[feature1],
            data[feature2],
            bins=bins,
            range=[[data[feature1].min(), data[feature1].max()], [data[feature2].min(), data[feature2].max()]],
        )

        # Filter out low-density regions to set axis ranges
        x_mask = hist.sum(axis=1) > density_threshold
        y_mask = hist.sum(axis=0) > density_threshold
        x_min, x_max = x_edges[np.where(x_mask)[0][[0, -1]]]
        y_min, y_max = y_edges[np.where(y_mask)[0][[0, -1]]]

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(title_template.format(feature1=feature1, feature2=feature2), fontsize=14)
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.grid(True, linestyle='--', alpha=0.7)

        # Plot the histogram with logarithmic z-axis
        im = ax.hist2d(
            data[feature1],
            data[feature2],
            bins=bins,
            cmap=cmap,
            norm=LogNorm(),
            range=[[x_min, x_max], [y_min, y_max]],
        )

        # Add colorbar and finalize layout
        fig.colorbar(im[3], ax=ax)
        fig.tight_layout()

        # Generate file name
        file_name = f"{feature1}_{feature2}_hist2d.png"
        yield fig, file_name

def plot_feature_combinations(
    data: pd.DataFrame,
    features: Union[list[str], list[Tuple[str, str]]],
    condition: np.ndarray = None,
    condition_legend: Tuple[str, str] = None,
    title_template: str = "Scatter Plot of {feature1} vs {feature2}",
    size: float = 0.8,
    alpha: float = 0.5,
) -> Iterator[Tuple[Figure, str]]:
    """
    Generator that yields scatter plots of combinations of features.

    Parameters:
    - data: DataFrame containing the dataset.
    - features: List of column names to plot or list of pairs of features.
    - condition: Boolean NDArray to color the scatter points.
    - condition_legend: Legend for the condition, a tuple of two elements.
    - title_template: Template for plot titles (use {feature1} and {feature2} as placeholders).
    - size: Size of the scatter points.
    - alpha: Transparency of the scatter points.
    """
    # Initialize condition and legend if not provided
    if condition is None:
        condition = np.zeros(len(data))
    if condition_legend is None:
        condition_legend = ("", "")

    # feature pairs
    feature_pairs = features if isinstance(features[0], tuple) else combinations(features, 2)

    def create_scatter_plot(feature1, feature2):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(title_template.format(feature1=feature1, feature2=feature2), fontsize=14)
        ax.scatter(data[feature1], data[feature2], c=condition, alpha=alpha, s=size)
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.tick_params(labelsize=10)
        if condition_legend != ("", ""):
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label=condition_legend[0], markerfacecolor='blue', markersize=8),
                Line2D([0], [0], marker='o', color='w', label=condition_legend[1], markerfacecolor='red', markersize=8)
            ]
            ax.legend(handles=legend_elements)
        fig.tight_layout()
        return fig

    # plots for each feature pair
    for feature1, feature2 in feature_pairs:
        fig = create_scatter_plot(feature1, feature2)
        file_name = f"{feature1}_vs_{feature2}.png"
        yield fig, file_name

def plot_group_ratio(
    group_labels: list[str],
    group_conditions: list[pd.Series],
    plot_type: Literal["pie", "bar", "barh"] = "pie",
    title: str = "Group Ratio",
) -> Figure:
    """
    Plot the ratio of group labels based on group conditions.

    Parameters:
    - group_labels: Labels for each condition group.
    - group_conditions: List of boolean Series to filter data into groups.
    - plot_type: One of ['pie', 'bar', 'barh'].
    - title: Title of the plot.

    Returns:
    - Figure: The matplotlib figure object.
    """
    group_counts = [sum(condition) for condition in group_conditions]
    total_count = sum(group_counts)
    group_ratios = [count / total_count for count in group_counts]
    sorted_data = sorted(zip(group_labels, group_ratios), key=lambda x: x[1], reverse=True)
    group_labels, group_ratios = zip(*sorted_data)

    fig = plt.figure(figsize=(8, 4))

    if plot_type == "pie":
        group_percentages = [f"{label}: {ratio * 100:.2f}%" for label, ratio in zip(group_labels, group_ratios)]
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])

        fig.suptitle(title, fontsize=14, y=1.05, ha='center')

        ax1 = fig.add_subplot(gs[0])
        ax1.axis("off")
        ax1.grid(False)
        wedges, _ = ax1.pie(group_ratios, startangle=180)

        ax2 = fig.add_subplot(gs[1])
        ax2.axis('off')
        ax2.legend(wedges, group_percentages, loc='center', bbox_to_anchor=(0.3,0.5), fontsize=12)

        gs.update(wspace=0.1, top=0.95, left=0.1, right=0.9, bottom=0.1)
    elif plot_type == "bar":
        ax = fig.add_subplot(111)
        ax.set_title(title, fontsize=12)
        ax.bar(group_labels, group_ratios)
        ax.set_xlabel("Group")
        ax.set_ylabel("Ratio")
        ax.tick_params(labelsize=10)
    elif plot_type == "barh":
        ax = fig.add_subplot(111)
        ax.set_title(title, fontsize=12)
        ax.barh(group_labels, group_ratios)
        ax.set_xlabel("Ratio")
        ax.set_ylabel("Group")
        ax.tick_params(labelsize=10)

    return fig
