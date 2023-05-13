"""This module contains utility functions."""
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import torch


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


# create a 1D smoothing filter
def smooth(X: npt.NDArray[np.float64], k: int = 5) -> npt.NDArray[np.float64]:
    """This is used to smoothen the plot."""
    return np.convolve(X, np.ones(k) / k, mode="same")


def load_data(*, filename: str, sep: str = ",") -> pd.DataFrame:
    """This is used to load the data.

    NB: Supported formats are 'csv' and 'parquet'.

    Params;
        filename (str): The filepath.\n
        sep (str, default=","): The separator. e.g ',', '\t', etc \n

    Returns:
        data (pd.DataFrame): The loaded dataframe.
    """
    data = (
        pd.read_csv(filename, sep=sep)
        if filename.split(".")[-1] == "csv"
        else pd.read_parquet(filename)
    )
    print(f"Shape of data: {data.shape}\n")

    return data
