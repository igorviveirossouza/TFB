# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import Optional

from ts_benchmark.models.model_base import ModelBase


class SNaive(ModelBase):
    """
    Seasonal Naive forecasting model.

    Forecast rule:
        y_hat(t+h) = y(t+h-season_length)
    """

    def __init__(self, season_length: int = 1):
        """
        :param season_length: The seasonal period (e.g., 12 for monthly yearly seasonality)
        """
        self._season_length = season_length
        self._train_series = None

    def forecast_fit(
        self,
        train_valid_data: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        **kwargs,
    ) -> "SNaive":
        """
        For SNaive, fitting only stores the training data.
        """
        self._train_series = train_valid_data.copy()
        return self

    def forecast(
        self,
        horizon: int,
        series: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Generate forecasts using seasonal naive logic.
        """
        if self._season_length <= 0:
            raise ValueError("season_length must be positive.")

        if len(series) < self._season_length:
            raise ValueError(
                "Series length must be >= season_length."
            )

        # Extract last full season
        last_season = series.iloc[-self._season_length:].values

        # Repeat season to cover horizon
        repetitions = int(np.ceil(horizon / self._season_length))
        forecast_values = np.tile(last_season, (repetitions, 1))

        # Trim to exact horizon
        forecast_values = forecast_values[:horizon]

        return forecast_values

    @property
    def model_name(self):
        return f"SNaive(s={self._season_length})"
