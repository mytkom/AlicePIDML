from typing import Literal
from pdi.data.types import Detector, COLUMN_DETECTOR, GroupID
from pdi.data.constants import COLUMNS_FOR_TRAINING
import pandas as pd
import numpy as np
from numpy.typing import NDArray

def filter_strings_by_binary(integer_filter: int, array: NDArray, persist_key: Literal['0'] | Literal['1'] = '1') -> NDArray:
    binary_str = bin(integer_filter)[2:].zfill(len(array))
    mask = np.array([bit == persist_key for bit in binary_str])

    return array[mask]

# For given row with NaNs in columns of missing values returns proper GroupID
# It should be input data row
def dataframe_row_to_group_id(row: pd.DataFrame) -> GroupID:
    notnarow = row.notna().astype("int").values
    bin_str = ''.join(map(str, notnarow))
    return GroupID(int(bin_str, base=2))

def binary_array_to_group_id(bin_arr: NDArray) -> GroupID:
    bin_str = ''.join(map(str, bin_arr.astype(int)))
    return GroupID(int(bin_str, base=2))

def group_id_to_binary_array(gid: GroupID):
    return np.array(list(bin(gid)[2:]), dtype=int)

def group_id_to_detectors_available(group_id: GroupID) -> list[Detector]:
    missing_columns = filter_strings_by_binary(int(group_id), np.array(COLUMNS_FOR_TRAINING), '0')
    detectors = [Detector.TPC, Detector.TOF, Detector.TRD]
    print(missing_columns)
    for col in missing_columns:
        if col in COLUMN_DETECTOR.keys() and COLUMN_DETECTOR[col] in detectors:
            detectors.remove(COLUMN_DETECTOR[col])
    return detectors

def group_id_to_missing_columns(group_id: GroupID) -> NDArray:
    return filter_strings_by_binary(int(group_id), np.array(COLUMNS_FOR_TRAINING), '0')

