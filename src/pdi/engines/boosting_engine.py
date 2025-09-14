from typing import Optional
from pdi.config import Config
from pdi.data.data_preparation import (
    DataPreparation,
)
from pdi.engines.base_engine import BaseEngine
from pdi.results_and_metrics import (
    TestResults,
)


class BoostingEngine(BaseEngine):
    """
    Boosting engine is a class for gradient boosted decision trees fitting and testing.
    """

    def __init__(
        self, cfg: Config, target_code: int, base_dir: str | None = None
    ) -> None:
        super().__init__(cfg, target_code, base_dir)
        self._data_prep = DataPreparation(cfg.data, cfg.sim_dataset_paths, cfg.seed)

        self._train_dl, self._val_dl, self._test_dl = self.setup_dataloaders(self._cfg, self._data_prep) 
        if self._data_prep._is_experimental:
            raise RuntimeError(
                "ClassicEngine got experimental data, it is not suited to handle it!"
            )

        self._data_prep.save_dataset_metadata(self._base_dir)

    def get_data_prep(self) -> DataPreparation:
        return self._data_prep

    def train(self):
        # TODO: loop over some hyperparameters sets or use optuna to train test many hyperparameters sets
        #   but not too much few at most, you should have now intuition of the good hyperparameters set.
        #   For optuna implementation e.g.

        # TODO: fit tree with training split using the Config object
        train_dataframe = self._train_dl.unwrap()
        # Something like:
        #   model = GBDTClass.fit(train_dataframe, args)

        # TODO: validate on validation split
        val_dataframe = self._val_dl.unwrap()
        # Something like:
        # predictions = model.predict(val_dataframe)
        # calculate val_metrics on your own, ValidationMetrics class is too torchy. Maybe you can create another BinaryValidationMetrics or similar

        # Log validation metrics
        # self._log_results(
        #     metrics={
        #         **{f"val/{k}": v for k, v in val_metrics.to_dict().items()},
        #     },
        #     csv_name="validation_metrics.csv",
        # )
        # print(
        #     f"F1: {val_metrics.f1:.4f}, ..."
        # )
        pass

    def _test(self, model_dirpath: Optional[str] = None) -> TestResults:
        # if model_dirpath is set, then load model from dirpath (see ClassicEngine)

        # test_dataframe = self._test_dl.unwrap()
        # predictions = self._model.predict(test_dataframe)
        # test_results = TestResults(
        #     targets=test_dataframe[TARGET_COLUMN],
        #     predictions=predictions,
        #     target_code=self._target_code,
        # )
        #
        # return test_results
        pass
