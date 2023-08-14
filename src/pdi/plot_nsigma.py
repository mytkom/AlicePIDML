import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap

import datashader as ds
from datashader.mpl_ext import dsshow

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

INPUT_PATH = "../PID_in_O2/LHC18g4_train246_mc_multiple_detectors.csv"
PART_DICT = {
    11: "El",
    13: "Mu",
    211: "Pi",
    321: "Ka",
    2212: "Pr",
    -11: "El",
    -13: "Mu",
    -211: "Pi",
    -321: "Ka",
    -2212: "Pr",
}
NSIGMA_COLUMNS = [
     *["fTPCNSigma" + val for val in PART_DICT.values()],
     *["fTOFNSigma" + val for val in PART_DICT.values()]]

CSV_DELIMITER = ","

df = pd.read_csv(INPUT_PATH, sep=CSV_DELIMITER, index_col=0)

for part in PART_DICT:
    for det in ["fTPCNSigma", "fTOFNSigma"]:
        nsigma = det + PART_DICT[part]
        df_part = df.loc[df["fPdgCode"] == part]

        fig, ax = plt.subplots()
        dsartist = dsshow(
                df_part,
                ds.Point("fPt", nsigma),
                ds.count(),
                norm="log",
                aspect="auto",
                ax=ax,
            )
        plt.colorbar(dsartist)

        plt.xlim([0.0, 10])
        plt.ylim([-3.5, 3.5])
        plt.xlabel("pT")
        plt.ylabel(nsigma)
        plt.title(f"{nsigma} vs pT")

        plt.savefig(f"nsigma_{nsigma}_pt.png")
        plt.close()
