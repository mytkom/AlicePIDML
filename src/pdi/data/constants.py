"""This module contains project constants related to the used data.
"""
from pdi.data.types import GroupID


PART_DICT = {
    11: "El",
    13: "Mu",
    211: "Pi",
    321: "Ka",
    2212: "Pr",
}

COLUMNS_FOR_TRAINING = [
    "fTPCSignal", "fTRDPattern", "fTOFSignal",
    "fBeta", "fP", "fPx", "fPy", "fPz",
    "fSign", "fX", "fY", "fZ", "fAlpha",
    "fTPCNClsShared", "fDcaXY", "fDcaZ"
]
N_COLUMNS = len(COLUMNS_FOR_TRAINING)

MAX_GROUP_ID: GroupID = GroupID(int(len(COLUMNS_FOR_TRAINING) * "1", base=2))

NSIGMA_COLUMNS = [
    *["fTPCNSigma" + val for val in PART_DICT.values()],
    *["fTOFNSigma" + val for val in PART_DICT.values()],
]

PROCESSED_DIR = "data/processed"
TARGET_COLUMN = "fPdgCode"
GROUP_ID_KEY = "group_id"
