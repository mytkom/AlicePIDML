from abc import abstractmethod, ABC
import dataclasses
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from typing import Any
from numpy.typing import NDArray

from pdi.config import DataConfig

class OutlierFilteringMethod(ABC):
    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> NDArray:
        pass

class IQRFilter(OutlierFilteringMethod):
    def __init__(self, iqr_multiplier: float = 1.5):
        self.iqr_multiplier = iqr_multiplier

    def __call__(self, df: pd.DataFrame) -> NDArray:
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((df < (Q1 - self.iqr_multiplier * IQR)) | (df > (Q3 + self.iqr_multiplier * IQR))).any(axis=1)
        return mask.to_numpy()

class IsolationForestFilter(OutlierFilteringMethod):
    def __init__(self, **kwargs: Any):
        self.model = IsolationForest(**kwargs)

    def __call__(self, df: pd.DataFrame) -> NDArray:
        self.model.fit(df)
        predictions = self.model.predict(df)
        return predictions == 1  # 1 indicates inliers, -1 indicates outliers

class OneClassSVMFilter(OutlierFilteringMethod):
    def __init__(self, **kwargs: Any):
        self.model = OneClassSVM(**kwargs)

    def __call__(self, df: pd.DataFrame) -> NDArray:
        self.model.fit(df)
        predictions = self.model.predict(df)
        return predictions == 1  # 1 indicates inliers, -1 indicates outliers

def build_outlier_filtering(cfg: DataConfig, seed: int) -> OutlierFilteringMethod:
    if cfg.outlier_filtering_method == "iqr":
        return IQRFilter(iqr_multiplier=cfg.outlier_filtering_methods.iqr.multiplier)
    elif cfg.outlier_filtering_method == "isolation_forest":
        return IsolationForestFilter(random_state=seed, **dataclasses.asdict(cfg.outlier_filtering_methods.isolation_forest))
    elif cfg.outlier_filtering_method == "ocsvm":
        return OneClassSVMFilter(**dataclasses.asdict(cfg.outlier_filtering_methods.ocsvm))
    else:
        raise KeyError(f"Unknown outlier filtering method: {cfg.outlier_filtering_method}")
