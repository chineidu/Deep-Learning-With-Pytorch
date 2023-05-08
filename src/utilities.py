"""This module contains utility functions."""
import numpy as np
import seaborn as sns
import torch


def create_iris_data() -> tuple[np.ndarray, np.ndarray]:
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
def smooth(X, k=5):
    """This is used to smoothen the plot."""
    return np.convolve(X, np.ones(k) / k, mode="same")
