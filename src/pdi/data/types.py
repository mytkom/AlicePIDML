from enum import Enum
from typing import NewType, TypeVar, Union

from numpy import ndarray
from pandas import DataFrame
from torch import Tensor

Split = Enum("Split", ["TRAIN", "VAL", "TEST"])
InputTarget = Enum("InputTarget", ["INPUT", "TARGET"])
Additional = Enum(
    "Additional",
    ["fP", "fTPCSignal", "fBeta"],
)

T = TypeVar("T")

DfOrArray = Union[DataFrame, ndarray]

SplitDict = dict[Split, T]
InputTargetDict = dict[InputTarget, T]
AddDict = dict[Additional, T]

GroupID = NewType("GroupID", int)
GIDDict = dict[GroupID, T]

DatasetItem = tuple[Tensor, Tensor, dict[str, Tensor]]
