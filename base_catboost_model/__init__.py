#
# Copyright (c) 2023-now by BUSUP TECHNOLOGIES, S.L.
# This file is part of project Brainer,
# and is released under the MIT License Agreement.
# See the LICENSE file for more information.
#
"""**Catboost base model**.

.. Note::

    In the code we follow the standard nomenclature for features and target:

    - X: For DataFrames with features.
    - y: For Series with target values.

"""
import time
from typing import Any, Dict, List, Tuple

import pandas as pd
import shap
from catboost import CatBoostClassifier, Pool
from pandera.typing import DataFrame
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

from brainer.core import logger
from brainer.core.exceptions import CatBoostError
from brainer.data.storage import default_storage

from .utils import EvaluationMetrics


class BaseCatBoostModel:
    """Class for CatBoostClassifier base model."""

    def __init__(
        self,
        target: str,
        features: List[str],
        cat_features: List[str],
        random_seed: int = 42,
        threshold: float = 0.5,
    ):
        """Initialize the model.

        Args:
            target: Target column.
            features: Features columns.
            cat_features: Categorical features columns.
            random_seed: Random seed for catboost.
            threshold: Threshold used in metrics calculations.
        """
        self.target = target
        self.prediction_column = f"probability_{self.target}"
        self.features = features
        self.cat_features = cat_features
        self.random_seed = random_seed
        self.threshold = threshold
        self.model = CatBoostClassifier()

    def __repr__(self) -> str:
        """Set what to print."""
        info = f"{self.__class__.__name__}:\n"
        info += f"  - target: {self.target}\n"
        info += f"  - prediction_column: {self.prediction_column}\n"
        info += f"  - cat_features: {self.cat_features}\n"
        info += f"  - features: {self.features}\n"
        info += f"  - random_seed: {self.random_seed}\n"
        info += f"  - threshold: {self.threshold}\n"
        return info

    def common_preprocessing(self, df: DataFrame) -> DataFrame:
        """Common preprocessing steps for training and prediction."""
        df.set_index("service_id", inplace=True)
        # drop rows that not permit missing values
        df.dropna(subset=self.cat_features, inplace=True)
        if self.target in df.columns:
            df.dropna(subset=[self.target], inplace=True)
        return df

    def _split_features_target(self, df: DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Split DataFrame into features (X) and target (y)."""
        X = df.drop(columns=[self.target])
        y = df[self.target]
        return X, y

    def _perform_grid_search_cv(
        self,
        *,
        param_grid: Dict[str, List],
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
    ) -> GridSearchCV:
        """Perform grid search for hyperparameter tuning.

        Args:
            param_grid: Grid parameters for cross validation.
            X_train: DataFrame with the features of the train set.
            y_train: DataFrame with the target of the train set.
        """
        logger.debug("Start grid search.")
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=5,
            verbose=10,
            n_jobs=4,
            scoring="neg_log_loss",
        )
        t0 = time.time()
        grid_search.fit(X_train, y_train)
        logger.debug(f"Grid search took {(time.time() - t0)/60:.2f} minutes.")
        return grid_search

    def train(
        self,
        *,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        param_grid: Dict[str, List[Any]] | None = None,
        path_to_model: str = "model_base_last.cbm",
        verbose: bool | int = 0,
        **kwargs,
    ) -> CatBoostClassifier:
        """Train the model.

        Args:
            X_train: DataFrame with the features of the train set.
            y_train: DataFrame with the target of the train set.
            param_grid: Grid for hyperparameter tuning cross validation.
            path_to_model: Path to save the model.
            verbose: If True, print information about the training process.
            kwargs: Keyword arguments for CatBoostClassifier.

        Returns:
            model: Trained model.
        """
        logger.debug("Model train started")
        # define model
        imbalance_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
        train_pool = Pool(
            data=X_train,
            label=y_train,
            cat_features=self.cat_features,
        )

        self.model = CatBoostClassifier(
            cat_features=self.cat_features,
            scale_pos_weight=imbalance_ratio,
            random_state=self.random_seed,
            verbose=verbose,
            **kwargs,
        )

        if param_grid is not None:
            grid_search = self._perform_grid_search_cv(
                param_grid=param_grid, X_train=X_train, y_train=y_train
            )
            # Get the best hyperparameters
            best_params = grid_search.best_params_
            logger.debug(f"Best Hyperparameters: {best_params}")
            logger.debug(f"Cross validation score: {grid_search.best_score_}")
            logger.debug(f"Training score: {grid_search.score(X_train, y_train)}")
            # Get the best model
            self.model = grid_search.best_estimator_
            logger.debug("Model cross-validation finished.")
        else:
            self.model.fit(train_pool)

        if path_to_model:
            default_storage.write(model=self.model, path=path_to_model)
            default_storage.write(
                model=self.model,
                path=path_to_model.replace("last", time.strftime("%Y%m%d_%H%M%S")),
            )
        logger.info("Model train finished.")
        return self.model

    def calculate_metrics(
        self, *, X_test: pd.DataFrame, y_true: pd.DataFrame
    ) -> EvaluationMetrics:
        """Calculate the metrics of the model.

        Args:
            X_test: DataFrame with the features of the test set.
            y_true: DataFrame with the target of the test set.

        Returns:
            EvaluationMetrics: Object with the metrics of the model.
        """
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred_proba = pd.Series(y_pred_proba, index=y_true.index)
        y_pred = (y_pred_proba > self.threshold).astype(int)

        evaluation_metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
            "pr_auc": average_precision_score(y_true, y_pred, pos_label=1),
            "brier_score_loss": brier_score_loss(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
        }
        return EvaluationMetrics(**evaluation_metrics)

    def predict(
        self,
        *,
        df: pd.DataFrame,
        path_to_model: str | None = "model_base_last.cbm",
    ) -> pd.DataFrame:
        """Predict is_play_pressed.

        Args:
            df: DataFrame with the features.
            path_to_model: Path to load the model.

        Returns:
            df_preds: DataFrame with the predictions.
        """
        logger.debug("Prediction started.")
        if path_to_model:
            self.model = default_storage.read(
                model=CatBoostClassifier(), path=path_to_model
            )

        if (self.model.classes_ == [0, 1]).all():
            preds = self.model.predict_proba(df)[:, 1]
        elif (self.model.classes_ == [1, 0]).all():
            preds = self.model.predict_proba(df)[:, 0]
        else:
            raise CatBoostError("The model has not been trained yet.")

        df_preds = pd.DataFrame(data=preds, columns=[self.prediction_column]).set_index(
            df.index
        )
        logger.debug("Prediction finished.")
        return df_preds

    def explain(self, df: pd.DataFrame) -> pd.DataFrame:
        """Explain the model's predictions with SHAP values.

        Args:
            df: DataFrame with the features.

        Returns:
            shap_values: DataFrame with the SHAP values.
        """
        logger.debug("SHAP explanation started.")
        explainer = shap.TreeExplainer(model=self.model)
        shap_explain = explainer(df)
        shap_values = pd.DataFrame(shap_explain.values).rename(
            columns=dict(enumerate(df.columns.map(lambda x: f"shap_{x}")))
        )
        shap_values.index = df.index
        logger.debug("SHAP explanation finished.")
        return shap_values

    def load_model(self, path_to_model: str) -> CatBoostClassifier:
        """Load model.

        Args:
            path_to_model: Path to model file that will be loaded.

        Returns:
            model: Loaded CatBoostClassifier model.
        """
        logger.debug("Load model started.")
        self.model = default_storage.read(
            model=CatBoostClassifier(), path=path_to_model
        )
        logger.debug("Load model finished.")
        return self.model
