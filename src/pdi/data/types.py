""" This module contains the enums and other types used as keys for dictionaries in preparation classes.
"""

from enum import Enum
from typing import MutableMapping, NewType

from torch import Tensor

from pdi.data.constants import PART_DICT
from pdi.data.config import RUN

Split = Enum("Split", ["TRAIN", "VAL", "TEST"])
InputTarget = Enum("InputTarget", ["INPUT", "TARGET"])
Additional_Run2 = Enum(
    "Additional",
    [
        "fP",
        "fTPCSignal",
        "fBeta",
        "fPt",
        "fSign",
        *["fTPCNSigma" + val for val in PART_DICT.values()],
        *["fTOFNSigma" + val for val in PART_DICT.values()],
    ],
)

Additional_Run3 = Enum(
    "Additional",
    ["fP", "fTPCSignal", "fBeta", "fPt", "fSign",
    ],
)

Additional = Additional_Run3 if RUN == 3 else Additional_Run2

GroupID = NewType("GroupID", int)

DatasetItem = tuple[Tensor, Tensor, MutableMapping[str, Tensor]]


class Detector(Enum):
    TPC = 1
    TOF = 2
    TRD = 4


COLUMN_DETECTOR = {
    "fTPCSignal": Detector.TPC,
    "fTRDSignal": Detector.TRD,
    "fTRDPattern": Detector.TRD,
    "fTOFSignal": Detector.TOF,
    "fBeta": Detector.TOF,
}
