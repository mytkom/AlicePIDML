from enum import Enum
from typing import Any, MutableMapping, NewType, TypeVar, Union

from numpy.typing import NDArray
from pandas import DataFrame
from torch import Tensor

Split = Enum("Split", ["TRAIN", "VAL", "TEST"])
InputTarget = Enum("InputTarget", ["INPUT", "TARGET"])
Additional = Enum(
    "Additional",
    ["fP", "fTPCSignal", "fBeta"],
)

T = TypeVar("T")

DfOrArray = Union[DataFrame, NDArray[Any]]

SplitMap = MutableMapping[Split, T]
GroupID = NewType("GroupID", int)
GIDMap = MutableMapping[GroupID, T]

InputTargetMap = MutableMapping[InputTarget, T]
AddMap = MutableMapping[Additional, T]

DatasetItem = tuple[Tensor, Tensor, MutableMapping[str, Tensor]]
