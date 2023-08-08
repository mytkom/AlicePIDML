""" This module contains the enums and other types used as keys for dictionaries in preparation classes.
"""

from enum import Enum
from typing import MutableMapping, NewType

from torch import Tensor

from pdi.constants import PART_DICT, TARGET_CODES

Split = Enum("Split", ["TRAIN", "VAL", "TEST"])
InputTarget = Enum("InputTarget", ["INPUT", "TARGET"])
Additional = Enum(
    "Additional",
    ["fP", "fTPCSignal", "fBeta", "fPt",
     *["fTPCNSigma" + val for val in PART_DICT.values()],
     *["fTOFNSigma" + val for val in PART_DICT.values()]],
)

GroupID = NewType("GroupID", int)

DatasetItem = tuple[Tensor, Tensor, MutableMapping[str, Tensor]]
