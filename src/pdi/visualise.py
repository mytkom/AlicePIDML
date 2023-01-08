import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from sklearn.metrics import precision_recall_curve

from pdi.constants import PARTICLES_DICT
from pdi.evaluate import get_interval_purity_efficiency


def plot_precision_recall_comparison(target_name, data_dict, save_dir=None):
    plt.figure()
    for method_name, results in data_dict.items():
        targets = results["targets"]
        preds = results["predictions"]

        precision, recall, _ = precision_recall_curve(targets, preds)
        plt.plot(recall, precision, label=method_name)

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title(f"{target_name} classification")
    plt.legend(loc="lower left")
    if save_dir is not None:
        plt.savefig(f"{save_dir}/precision_recall.png")
    plt.show()
    plt.close()


def plot_purity_comparison(target_name,
                           data_dict,
                           intervals,
                           save_dir=None):  # TODO
    plt.figure()
    for method_name, results in data_dict.items():
        targets = results["targets"]
        preds = results["predictions"]
        fP = results["momentum"]
        threshold = results["threshold"]

        purities_p_plot, _, confidence_intervals, momenta_avg = get_interval_purity_efficiency(
            targets, preds, fP, intervals, threshold)

        p = plt.plot(momenta_avg, purities_p_plot, label=method_name)
        plt.fill_between(
            momenta_avg,
            confidence_intervals["purity_lower"],
            confidence_intervals["purity_upper"],
            color=p[0].get_color(),
            alpha=0.2,
        )

    plt.xlabel("p (GeV/c)")
    plt.ylabel("Purity")
    plt.ylim([0, 1.1])

    plt.title(f"{target_name} classification")
    plt.legend(loc="lower left")
    if save_dir is not None:
        plt.savefig(f"{save_dir}/p_purity_optimized_threshold.png")
    plt.show()
    plt.close()


def plot_efficiency_comparison(target_name,
                               data_dict,
                               intervals,
                               save_dir=None):
    plt.figure()
    for method_name, results in data_dict.items():
        targets = results["targets"]
        preds = results["predictions"]
        fP = results["momentum"]
        threshold = results["threshold"]

        _, efficiencies_p_plot, confidence_intervals, momenta_avg = get_interval_purity_efficiency(
            targets, preds, fP, intervals, threshold)

        p = plt.plot(momenta_avg, efficiencies_p_plot, label=method_name)
        plt.fill_between(
            momenta_avg,
            confidence_intervals["efficiency_lower"],
            confidence_intervals["efficiency_upper"],
            color=p[0].get_color(),
            alpha=0.2,
        )

    plt.xlabel("p (GeV/c)")
    plt.ylabel("Efficiency")
    plt.ylim([0, 1.1])
    # threshold: {selected_threshold}")
    plt.title(f"{target_name} classification")
    plt.legend(loc="lower left")
    if save_dir is not None:
        plt.savefig(f"{save_dir}/p_efficiency_optimized_threshold.png")
    plt.show()
    plt.close()


def plot_selected_particles(target_name,
                            selected,
                            p,
                            column_data,
                            column_name,
                            ybins_max,
                            save_dir=None):
    xbins = 10**np.linspace(-1, 1, 1000)
    ybins = np.linspace(0, ybins_max, 500)
    counts, _, _ = np.histogram2d(p[selected],
                                  column_data[selected],
                                  bins=(xbins, ybins))
    fig, ax = plt.subplots()
    ax.pcolormesh(xbins,
                  ybins,
                  counts.T,
                  cmap=plt.cm.jet,
                  norm=colors.LogNorm())
    ax.set_xscale("log")
    plt.title(f"Particle:{target_name} selected")
    plt.ylabel(column_name)
    if save_dir is not None:
        plt.savefig(f"{save_dir}/selected_particles_{column_name}.png")
    plt.show()
    plt.close()

    counts, _, _ = np.histogram2d(p[~selected],
                                  column_data[~selected],
                                  bins=(xbins, ybins))
    fig, ax = plt.subplots()
    ax.pcolormesh(xbins,
                  ybins,
                  counts.T,
                  cmap=plt.cm.jet,
                  norm=colors.LogNorm())
    ax.set_xscale("log")
    plt.title(f"Particle:{target_name} not selected")
    plt.ylabel(column_name)
    if save_dir is not None:
        plt.savefig(f"{save_dir}/not_selected_particles_{column_name}.png")
    plt.show()
    plt.close()


def plot_contamination(target_name,
                       targets,
                       selected,
                       p,
                       bins,
                       p_min,
                       p_max,
                       save_dir=None):
    selected_targets = targets[selected]
    selected_p = p[selected]
    plt.figure()
    for selected_target in np.unique(selected_targets):
        plt.hist(
            selected_p[selected_targets == selected_target],
            bins,
            histtype="step",
            label=selected_target,
            range=(p_min, p_max),
        )
    plt.xlabel("p (GeV/c)")
    plt.ylabel("Contamination")
    plt.yscale("log")

    plt.title(f"{target_name} contamination")
    plt.legend([PARTICLES_DICT[x] for x in np.unique(selected_targets)],
               loc="upper right")
    if save_dir is not None:
        plt.savefig(f"{save_dir}/p_contamination_optimized_threshold.png")
    plt.show()
    plt.close()
