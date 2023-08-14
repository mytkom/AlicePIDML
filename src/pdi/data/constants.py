"""This module contains project constants related to the used data.
"""

PART_DICT = {
    11: "El",
    13: "Mu",
    211: "Pi",
    321: "Ka",
    2212: "Pr",
}

COLUMNS_TO_SCALE = [
    "fTPCSignal",
    "fTRDSignal",
    "fTOFSignal",
    "fBeta",
    "fX",
    "fY",
    "fZ",
    "fAlpha",
    "fTPCNClsShared",
    "fDcaXY",
    "fDcaZ",
]

NSIGMA_COLUMNS = [
     *["fTPCNSigma" + val for val in PART_DICT.values()],
     *["fTOFNSigma" + val for val in PART_DICT.values()]]

CSV_DELIMITER = ","
DROP_COLUMNS_BIG = ["fPdgCode", "fIsPhysicalPrimary",
     *["fTPCExpSigma" + val for val in PART_DICT.values()],
     *["fTOFExpSigma" + val for val in PART_DICT.values()],
     *["fTPCExpSignalDiff" + val for val in PART_DICT.values()],
     *["fTOFExpSignalDiff" + val for val in PART_DICT.values()],
     "fTrackEtaEMCAL", "fTrackPhiEMCAL", "fCentRun2V0M",
     "fMultFV0A", "fMultFV0C", "fMultFV0M",
     "fMultFT0A", "fMultFT0C", "fMultFT0M",
     "fMultZNA", "fMultZNC", "fMultTracklets", "fMultTPC"]
DROP_COLUMNS_SMALL = ["fPdgCode", "fIsPhysicalPrimary"]
N_COLUMNS_BIG = 64
N_COLUMNS_SMALL = 21
N_COLUMNS_ML=19
N_COLUMNS_NSIGMAS=29
INPUT_PATH = "../PID_in_O2/LHC18g4_train246_mc_multiple_detectors.csv"
MISSING_VALUES = {
    "fBeta": -999,
    "fTRDPattern": 0,
}
N_COLUMNS = 19
PROCESSED_DIR = "data/processed"
SEED = 42
TARGET_COLUMN = "fPdgCode"
TEST_SIZE = 0.3
TRAIN_SIZE = 0.55
GROUP_ID_KEY = "group_id"
