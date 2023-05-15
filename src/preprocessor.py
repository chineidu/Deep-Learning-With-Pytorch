"""This module is used for preprocessing data."""
from typing import Union

import numpy as np
import numpy.typing as npt
import pandas as pd


class Standardizer:
    """This class is used to standardize the data.
    i.e. the result has a mean of 0 and a standard deviation of 1."""

    def __init__(self) -> None:
        self.mean = None
        self.std = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, " f"std={self.std})"

    @staticmethod
    def _standardize(
        X: Union[pd.DataFrame, npt.NDArray[np.float_]],
        mean_: npt.NDArray[np.float_],
        std_: npt.NDArray[np.float_],
    ) -> float:
        """This is used to standardize the data."""
        return (X - mean_) / std_

    def fit(
        self, X: Union[pd.DataFrame, npt.NDArray[np.float_]], y=None
    ) -> Union[pd.DataFrame, npt.NDArray[np.float_]]:
        """This is used to learn the parameters,"""
        self.mean = np.zeros(shape=X.shape[1])
        self.std = np.zeros(shape=X.shape[1])

        for idx, var in enumerate(X.columns):
            self.mean[idx] = np.mean(X[var])  # type: ignore
            self.std[idx] = np.std(X[var])  # type: ignore

        return self

    def transform(
        self, X: Union[pd.DataFrame, npt.NDArray[np.float_]], y=None
    ) -> Union[pd.DataFrame, npt.NDArray[np.float_]]:
        """This applies the transformation."""
        X = self._standardize(X=X, mean_=self.mean, std_=self.std)
        return X

    def fit_transform(
        self, X: Union[pd.DataFrame, npt.NDArray[np.float_]], y=None
    ) -> Union[pd.DataFrame, npt.NDArray[np.float_]]:
        """This is used to learn the parameters and apply the transformation."""
        self.fit(X=X)
        X = self.transform(X=X)
        return X


class Normalizer:
    """This class is used to normalize the data.
    i.e. the result has a min and max value of 0 and 1 by default."""

    def __init__(self, min_value: float = 0, max_value: float = 1) -> None:
        self.min_value = min_value
        self.max_value = max_value
        self._min = 0
        self._max = 0

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(min_value={self.min_value}, " f"max_value={self.max_value})"
        )

    @staticmethod
    def _normalize(
        X: Union[pd.DataFrame, npt.NDArray[np.float_]],
        min_: npt.NDArray[np.float_],
        max_: npt.NDArray[np.float_],
    ) -> float:
        """This is used to normalize the data."""
        return (X - min_) / (max_ - min_)

    def _custom_normalize(
        self,
        X: Union[pd.DataFrame, npt.NDArray[np.float_]],
        min_: npt.NDArray[np.float_],
        max_: npt.NDArray[np.float_],
    ) -> float:
        """This is used to adjust the min and max values."""
        x_scaled = self._normalize(X=X, min_=min_, max_=max_)
        x_adjusted = self.min_value + x_scaled * (self.max_value - self.min_value)
        return x_adjusted

    def fit(
        self, X: Union[pd.DataFrame, npt.NDArray[np.float_]], y=None
    ) -> Union[pd.DataFrame, npt.NDArray[np.float_]]:
        """This is used to learn the parameters,"""
        self._min = np.zeros(shape=X.shape[1])
        self._max = np.zeros(shape=X.shape[1])

        for idx, var in enumerate(X.columns):
            self._min[idx] = np.min(X[var])  # type: ignore
            self._max[idx] = np.max(X[var])  # type: ignore

        return self

    def transform(
        self, X: Union[pd.DataFrame, npt.NDArray[np.float_]], y=None
    ) -> Union[pd.DataFrame, npt.NDArray[np.float_]]:
        """This applies the transformation."""
        # X = self._normalize(X=X, min_=self._min, max_=self._max)
        X = self._custom_normalize(X=X, min_=self._min, max_=self._max)
        return X

    def fit_transform(
        self, X: Union[pd.DataFrame, npt.NDArray[np.float_]], y=None
    ) -> Union[pd.DataFrame, npt.NDArray[np.float_]]:
        """This is used to learn the parameters and apply the transformation."""
        self.fit(X=X)
        X = self.transform(X=X)
        return X
