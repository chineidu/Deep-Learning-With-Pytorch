"""This module contains utility functions."""

import os
import sys
import logging
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def create_iris_data() -> tuple[torch.Tensor, torch.Tensor]:
    """This returns the independent and the target features."""
    # load data
    iris_data = sns.load_dataset("iris")

    # Preprocess the data
    condlist = [
        (iris_data["species"] == "setosa"),
        (iris_data["species"] == "versicolor"),
        iris_data["species"] == "virginica",
    ]
    choicelist = [0, 1, 2]
    iris_data["target"] = np.select(condlist=condlist, choicelist=choicelist)

    # Convert the data to Torch tensor
    X = torch.Tensor(iris_data.loc[:, iris_data.columns[:4]].values)
    y = torch.Tensor(iris_data["target"].values).long()

    print(f"Shape of X: {X.shape}, Shape of X: {y.shape}")
    return (X, y)


def create_qwerties_data() -> tuple[torch.Tensor, torch.Tensor]:
    """This is used to generate data. It returns a tuple containing X and y."""
    n_per_clust, blur = 300, 1

    A = [1, 1]
    B = [5, 1]
    C = [4, 3]

    # generate data
    a = [
        A[0] + np.random.randn(n_per_clust) * blur,
        A[1] + np.random.randn(n_per_clust) * blur,
    ]
    b = [
        B[0] + np.random.randn(n_per_clust) * blur,
        B[1] + np.random.randn(n_per_clust) * blur,
    ]
    c = [
        C[0] + np.random.randn(n_per_clust) * blur,
        C[1] + np.random.randn(n_per_clust) * blur,
    ]

    # true labels
    labels_np = np.hstack(
        (np.zeros((n_per_clust)), np.ones((n_per_clust)), 1 + np.ones((n_per_clust)))
    )

    # concatanate into a matrix
    data_np = np.hstack((a, b, c)).T

    # convert to a pytorch tensor
    X = torch.tensor(data_np).float()  # pylint: disable=no-member
    y = torch.tensor(labels_np).long()  # pylint: disable=no-member
    return (X, y)


# create a 1D smoothing filter
def smooth(X: npt.NDArray[np.float64], k: int = 5) -> npt.NDArray[np.float64]:
    """This is used to smoothen the plot."""
    return np.convolve(X, np.ones(k) / k, mode="same")


def set_up_logger(delim: str = "::") -> Any:
    """This is used to create a basic logger."""

    format_ = f"[%(levelname)s]{delim} %(asctime)s{delim} %(message)s"
    logging.basicConfig(level=logging.INFO, format=format_)
    logger = logging.getLogger(__name__)
    return logger


def go_up_from_current_directory(go_up: int = 1) -> None:
    """This is used to up a number of directories.

    Args:
    -----
    go_up: int, default=1
        This indicates the number of times to go back up from the current directory.

    Returns:
    --------
    None
    """

    CONST: str = "../"
    NUM: str = CONST * go_up

    # Goto the previous directory
    prev_directory = os.path.join(os.path.dirname(__name__), NUM)
    # Get the 'absolute path' of the previous directory
    abs_path_prev_directory = os.path.abspath(prev_directory)

    # Add the path to the System paths
    sys.path.insert(0, abs_path_prev_directory)
    print(abs_path_prev_directory)


Model: TypeAlias = nn.Module


def compute_accuracy(model: Model, dataloader: DataLoader) -> float:
    model = model.eval()

    correct: float = 0.0
    total_examples: int = 0

    for _, (features, labels) in enumerate(dataloader):
        with torch.inference_mode():  # Same as torch.no_grad
            logits = model(features)

        predictions: torch.Tensor = torch.argmax(logits, dim=1)

        compare: bool = (labels == predictions).float()
        correct += torch.sum(compare)
        total_examples += len(compare)  # type: ignore

    return correct / total_examples
