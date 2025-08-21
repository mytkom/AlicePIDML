"""This module contains the enums and other types used as keys for dictionaries in preparation classes."""

from enum import Enum
from typing import NewType

Split = Enum("Split", ["TRAIN", "VAL", "TEST"])
InputTarget = Enum("InputTarget", ["INPUT", "TARGET", "UNSTANDARDIZED"])
GroupID = NewType("GroupID", int)


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
