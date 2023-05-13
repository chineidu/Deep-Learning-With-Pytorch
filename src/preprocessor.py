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
