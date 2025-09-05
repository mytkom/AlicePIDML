import gzip
import pickle
from typing import Optional
from joblib.pool import np
from numpy.typing import NDArray
from sklearn.metrics import precision_score, recall_score, f1_score
from pdi.evaluate import maximize_f1


class ValidationMetrics:
    """
    Represents evaluation metrics for training/validation, calculated dynamically using optimal threshold for F1 score.
    """

    def __init__(self, targets: NDArray, predictions: NDArray, loss: float):
        self.f1, self.precision, self.recall, self.threshold = maximize_f1(
            targets, predictions
        )
        self.loss = loss

    def to_dict(self) -> dict:
        """
        Converts the validation metrics to a dictionary for logging or serialization.
        """
        return {
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "loss": self.loss,
            "threshold": self.threshold,
        }


class TestMetrics:
    """
    Represents evaluation metrics for test evaluation, calculated using the optimal threshold from validation.
    """

    def __init__(
        self,
        targets: NDArray,
        predictions: NDArray,
        target_code: int,
        threshold: float | None = None,
        loss: Optional[float] = None,
    ):
        binary_targets = targets == target_code
        if threshold is not None:
            self.binary_predictions = predictions >= threshold
        else:
            self.binary_predictions = predictions

        self.f1 = f1_score(binary_targets, self.binary_predictions, average="binary")
        self.precision = precision_score(
            binary_targets, self.binary_predictions, average="binary"
        )
        self.recall = recall_score(
            binary_targets, self.binary_predictions, average="binary"
        )
        self.loss = loss
        self.threshold = threshold
        self.target_code = target_code

    def to_dict(self) -> dict:
        """
        Converts the test metrics to a dictionary for logging or serialization.
        """
        return {
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "loss": self.loss,
            "threshold": self.threshold,
            "target_code": self.target_code,
        }


class TestResults:
    """
    Represents test results, including test metrics and data used for evaluation.
    """

    def __init__(
        self,
        targets: NDArray,
        predictions: NDArray,
        target_code: int,
        threshold: float | None = None,
        loss: float | None = None,
    ):
        self.predictions = predictions
        self.targets = targets
        self.target_code = target_code
        self.test_metrics = TestMetrics(
            targets, predictions, target_code, threshold, loss
        )

    @classmethod
    def from_file(cls, filepath: str) -> "TestResults":
        """
        Initializes the TestResults object by loading data from a pickle file.

        Args:
            filepath (str): Path to the pickle file containing serialized test results.

        Returns:
            TestResults: An instance of TestResults initialized with the loaded data.
        """
        with gzip.open(filepath, "rb") as file:
            data = pickle.load(file)

        return cls(
            targets=np.array(data["targets"]),
            predictions=np.array(data["predictions"]),
            threshold=data["test_metrics"]["threshold"],
            target_code=data["target_code"],
            loss=data["test_metrics"]["loss"],
        )

    def save(self, filepath: str):
        dict_repr = self.to_dict()

        with gzip.open(filepath, "w") as file:
            pickle.dump(dict_repr, file)

    def to_dict(self) -> dict:
        """
        Converts the test results to a dictionary for logging or serialization.
        """
        return {
            "targets": self.targets.tolist(),
            "predictions": self.predictions.tolist(),
            "target_code": self.target_code,
            "test_metrics": self.test_metrics.to_dict(),
        }
