import gzip
import pickle
from typing import List, Optional
from joblib.pool import np
from numpy.typing import NDArray
from sklearn.metrics import precision_score, recall_score, f1_score
from pdi.evaluate import maximize_f1

class ValidationMetrics:
    """
    Represents evaluation metrics for training/validation, calculated dynamically using optimal threshold for F1 score.
    """
    def __init__(self, targets: NDArray, predictions: NDArray, loss: float):
        self.f1, self.precision, self.recall, self.threshold = maximize_f1(targets, predictions)
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
    def __init__(self, targets: NDArray, predictions: NDArray, threshold: float, target_code: int, loss: Optional[float] = None):
        binary_targets = targets == target_code
        binary_predictions = predictions >= threshold
        self.f1 = f1_score(binary_targets, binary_predictions, average="binary")
        self.precision = precision_score(binary_targets, binary_predictions, average="binary")
        self.recall = recall_score(binary_targets, binary_predictions, average="binary")
        self.loss = loss
        self.threshold = threshold
        self.target_code = target_code
        self.targets = targets
        self.predictions = predictions

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

class TrainResults:
    """
    Represents training results, including validation metrics and loss data.
    """
    def __init__(self, train_losses: List[float], val_losses: List[float]):
        self.train_losses = train_losses
        self.val_losses = val_losses

    def to_dict(self) -> dict:
        """
        Converts the training results to a dictionary for logging or serialization.
        """
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }


class TestResults:
    """
    Represents test results, including test metrics and data used for evaluation.
    """
    def __init__(self, inputs: NDArray, targets: NDArray, predictions: NDArray, unstandardized: dict[str, NDArray], test_metrics: TestMetrics):
        self.inputs = inputs
        self.targets = targets
        self.predictions = predictions
        self.unstandardized = unstandardized
        self.test_metrics = test_metrics

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

        test_metrics = TestMetrics(
            targets=np.array(data["targets"]),
            predictions=np.array(data["predictions"]),
            threshold=data["test_metrics"]["threshold"],
            target_code=data["test_metrics"]["target_code"],
            loss=data["test_metrics"]["loss"],
        )

        return cls(
            inputs=np.array(data["inputs"]),
            targets=np.array(data["targets"]),
            predictions=np.array(data["predictions"]),
            unstandardized={key: np.array(value) for key, value in data["unstandardized"].items()},
            test_metrics=test_metrics,
        )

    def to_dict(self) -> dict:
        """
        Converts the test results to a dictionary for logging or serialization.
        """
        return {
            "inputs": self.inputs.tolist(),
            "targets": self.targets.tolist(),
            "predictions": self.predictions.tolist(),
            "unstandardized": {key: value.tolist() for key, value in self.unstandardized.items()},
            "test_metrics": self.test_metrics.to_dict(),
        }

