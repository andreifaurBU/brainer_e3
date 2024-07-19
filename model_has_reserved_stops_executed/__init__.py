#
# Copyright (c) 2023-now by BUSUP TECHNOLOGIES, S.L.
# This file is part of project Brainer,
# and is released under the MIT License Agreement.
# See the LICENSE file for more information.
#
"""Catboost model predicting has_reserved_stops_executed."""
from typing import Tuple

import pandas as pd
from pandera.decorators import check_input
from pandera.typing import DataFrame
from sklearn.model_selection import train_test_split

from brainer.core.singleton import Singleton
from brainer.ml.prediction_models.base_catboost_model import BaseCatBoostModel
from brainer.schemas_df.features import (
    FeaturesServiceHasReservedStopsExecutedPredictDFSchema,
    FeaturesServiceHasReservedStopsExecutedTrainDFSchema,
)


class ModelHasReservedStopsExecuted(BaseCatBoostModel, metaclass=Singleton):
    """Class for CatBoost model has_reserved_stops_executed."""

    def __init__(self):
        """Initialize the model."""
        _cat_features = [
            "season_of_year",
            "month_of_year",
            "day_of_week",
            "hour_of_day",
            "driver_id",
            "operator_id",
            "route_id",
        ]
        _features = _cat_features + [
            "perc_pressed_play_driver",
            "num_past_services_driver",
            "perc_stop_on_time_driver",
            "perc_reserved_stop_executed_from_on_time_driver",
            "perc_reserved_stop_executed_from_on_time_route",
        ]
        super().__init__(
            target="has_reserved_stops_executed",
            cat_features=_cat_features,
            features=_features,
        )

    @staticmethod
    def _handle_missing_features(df: DataFrame) -> DataFrame:
        """Handle missing values."""
        # Fill missing 'perc_pressed_play_driver' with the mean
        perc_past_play = df["perc_pressed_play_driver"].mean()
        df["perc_pressed_play_driver"] = df["perc_pressed_play_driver"].fillna(
            perc_past_play
        )
        perc_past_stop_on_time = df["perc_stop_on_time_driver"].mean()
        df["perc_stop_on_time_driver"] = df["perc_stop_on_time_driver"].fillna(
            perc_past_stop_on_time
        )
        perc_past_reserved_stop_executed_from_on_time_driver = df[
            "perc_reserved_stop_executed_from_on_time_driver"
        ].mean()
        df["perc_reserved_stop_executed_from_on_time_driver"] = df[
            "perc_reserved_stop_executed_from_on_time_driver"
        ].fillna(perc_past_reserved_stop_executed_from_on_time_driver)

        perc_past_reserved_stop_executed_from_on_time_route = df[
            "perc_reserved_stop_executed_from_on_time_route"
        ].mean()
        df["perc_reserved_stop_executed_from_on_time_route"] = df[
            "perc_reserved_stop_executed_from_on_time_route"
        ].fillna(perc_past_reserved_stop_executed_from_on_time_route)
        # Fill missing 'num_past_services_driver' with 0
        df["num_past_services_driver"] = df["num_past_services_driver"].fillna(0)
        return df

    @check_input(FeaturesServiceHasReservedStopsExecutedTrainDFSchema.to_schema(), "df")
    def preprocess_train(
        self,
        *,
        df: DataFrame[FeaturesServiceHasReservedStopsExecutedTrainDFSchema],
        test_size: float = 0.2,
    ) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        """Preprocess the data.

        Args:
            df: DataFrame with the features.
            test_size: Size of the test set. If it is zero, then train with all data.

        Returns:
            X_train: DataFrame with the features of the train set.
            X_test: DataFrame with the features of the test set.
            y_train: DataFrame with the target of the train set.
            y_test: DataFrame with the target of the test set.
        """
        df = self.common_preprocessing(df)
        X, y = self._split_features_target(df)

        if test_size == 0:
            X_train = X
            y_train = y
            X_test = pd.DataFrame()
            y_test = pd.DataFrame()
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=self.random_seed
            )

        X_train = self._handle_missing_features(DataFrame(X_train))
        if not X_test.empty:
            X_test = self._handle_missing_features(DataFrame(X_test))

        return (
            DataFrame(X_train),
            DataFrame(X_test),
            DataFrame(y_train),
            DataFrame(y_test),
        )

    @check_input(FeaturesServiceHasReservedStopsExecutedPredictDFSchema.to_schema())
    def preprocess_predict(
        self,
        df: DataFrame[FeaturesServiceHasReservedStopsExecutedPredictDFSchema],
    ) -> pd.DataFrame:
        """Preprocess the data.

        Args:
            df: DataFrame with the features.

        Returns:
            X: DataFrame with the features.
        """
        X = self.common_preprocessing(df)
        X = self._handle_missing_features(X)
        return X
