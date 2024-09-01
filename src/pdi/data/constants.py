"""This module contains project constants related to the used data.
"""
from pdi.data.config import RUN

PART_DICT = {
    11: "El",
    13: "Mu",
    211: "Pi",
    321: "Ka",
    2212: "Pr",
}

COLUMNS_TO_SCALE_RUN_2 = [
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

# no TRDSignal scaling in run3
COLUMNS_TO_SCALE_RUN_3 = [
    "fTPCSignal",
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

COLUMNS_TO_SCALE = COLUMNS_TO_SCALE_RUN_3 if RUN == 3 else COLUMNS_TO_SCALE_RUN_2

NSIGMA_COLUMNS = [
    *["fTPCNSigma" + val for val in PART_DICT.values()],
    *["fTOFNSigma" + val for val in PART_DICT.values()],
]

CSV_DELIMITER = ","
DROP_COLUMNS = ["fPdgCode", "fIsPhysicalPrimary"]
DROP_COLUMNS_BIG = [
    "fPdgCode",
    "fIsPhysicalPrimary",
    *["fTPCExpSigma" + val for val in PART_DICT.values()],
    *["fTOFExpSigma" + val for val in PART_DICT.values()],
    *["fTPCExpSignalDiff" + val for val in PART_DICT.values()],
    *["fTOFExpSignalDiff" + val for val in PART_DICT.values()],
    "fTrackEtaEMCAL",
    "fTrackPhiEMCAL",
    "fCentRun2V0M",
    "fMultFV0A",
    "fMultFV0C",
    "fMultFV0M",
    "fMultFT0A",
    "fMultFT0C",
    "fMultFT0M",
    "fMultZNA",
    "fMultZNC",
    "fMultTracklets",
    "fMultTPC",
]
DROP_COLUMNS_SMALL = ["fPdgCode", "fIsPhysicalPrimary"]
N_COLUMNS_BIG = 64
N_COLUMNS_SMALL = 21
N_COLUMNS_ML = 19
N_COLUMNS_NSIGMAS = 29
MISSING_VALUES = {
    "fBeta": -999,
    "fTOFSignal": -999,
    "fTRDPattern": 0,
    "fTRDSignal": 0
}
N_COLUMNS = 16
PROCESSED_DIR = "data/processed"
SEED = 42
TARGET_COLUMN = "fPdgCode"
TEST_SIZE = 0.3
TRAIN_SIZE = 0.55
GROUP_ID_KEY = "group_id"
P_CUT = 0.5
TPC_CUT = 10000

COLUMNS_DROPPED_FOR_TESTS = ["fPt", "fTRDSignal", "fTrackType"]
DO_NOT_SCALE = ["fPdgCode", "fIsPhysicalPrimary", "fSign", "fPt"]
