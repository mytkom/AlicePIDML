import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from pdi.data.constants import INPUT_PATH, CSV_DELIMITER, NSIGMA_COLUMNS

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

INPUT_PATH = "../PID_in_O2/LHC18g4_train246_mc_multiple_detectors.csv"
PART_DICT = {
    11: "El",
    13: "Mu",
    211: "Pi",
    321: "Ka",
    2212: "Pr",
}
NSIGMA_COLUMNS = [
     *["fTPCNSigma" + val for val in PART_DICT.values()],
     *["fTOFNSigma" + val for val in PART_DICT.values()]]

CSV_DELIMITER = ","

df = pd.read_csv(INPUT_PATH, sep=CSV_DELIMITER, index_col=0)

for column in NSIGMA_COLUMNS:
    plt.figure()
    plt.plot(df["fPt"], df[column], ",")
    plt.xlabel("pT")
    plt.ylabel(column)
    plt.title(f"{column} vs pT")
    plt.savefig(f"nsigma_{column}_pt.png")
    #plt.show()
    plt.close()
