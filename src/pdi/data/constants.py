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
DROP_COLUMNS = ["fPdgCode", "fIsPhysicalPrimary", *NSIGMA_COLUMNS,
     *["fTPCExpSigma" + val for val in PART_DICT.values()],
     *["fTOFExpSigma" + val for val in PART_DICT.values()],
     *["fTPCExpSignalDiff" + val for val in PART_DICT.values()],
     *["fTOFExpSignalDiff" + val for val in PART_DICT.values()],
     "fTrackEtaEMCAL", "fTrackPhiEMCAL", "fCentRun2V0M",
     "fMultFV0A", "fMultFV0C", "fMultFV0M",
     "fMultFT0A", "fMultFT0C", "fMultFT0M",
     "fMultZNA", "fMultZNC", "fMultTracklets", "fMultTPC"]
NSIGMA_DROP_COLUMNS = ["fPdgCode", "fIsPhysicalPrimary"]
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
