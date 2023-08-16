import pandas as pd
import matplotlib.pyplot as plt

import datashader as ds
from datashader.mpl_ext import dsshow

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
        count = df_part[nsigma].size
        mean_y = round(df_part[nsigma].mean(), 5)
        std_dev_y = round(df_part[nsigma].std(), 5)
        mean_x = round(df_part["fPt"].mean(), 5)
        std_dev_x = round(df_part["fPt"].std(), 5)

        fig, ax = plt.subplots()
        dsartist = dsshow(
            df_part,
            ds.Point("fPt", nsigma),
            ds.count(),
            norm="log",
            aspect="auto",
            ax=ax,
            )
        plt.colorbar(dsartist, label="entries")
        plt.figtext(0.5, 0.75, f"Entries    {count}\nMean x     {mean_x}\nStd Dev x   {std_dev_x}"\
                    f"Mean y    {mean_y}\nStd Dev y   {std_dev_y}",
                    bbox={"facecolor":"white", "pad":5})

        plt.xlim([0.0, 5.0])
        plt.ylim([-3.5, 3.5])
        plt.xlabel("pT")
        plt.ylabel(nsigma)
        plt.title(f"{nsigma} vs pT")

        plt.savefig(f"nsigma_{nsigma}_pt.png", bbox_inches="tight")
        plt.close()
