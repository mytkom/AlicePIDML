""" This module contains the enums and other types used as keys for dictionaries in preparation classes.
"""

from enum import Enum
from typing import MutableMapping, NewType

from torch import Tensor

from pdi.data.constants import NSIGMA_COLUMNS
from pdi.data.config import GET_NSIGMA

Split = Enum("Split", ["TRAIN", "VAL", "TEST"])
InputTarget = Enum("InputTarget", ["INPUT", "TARGET"])
Additional_list = [
    "fP",
    "fTPCSignal",
    "fBeta",
    "fPt",
    "fSign",
]

if GET_NSIGMA:
    Additional_list += NSIGMA_COLUMNS
Additional = Enum("Additional", Additional_list)

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
