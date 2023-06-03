"""This module is used for loading and manipulating data."""
from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.utilities import set_up_logger


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


def split_into_train_n_validation(
    *,
    data: pd.DataFrame,
    labels: pd.Series,
    convert_to_long: bool = False,
    test_size: float = 0.2,
    random_state: int = 123,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """This is used to split the the data into X_train, X_validation,
    y_train, and y_validation

    Params:
        data (pd.DataFrame): Data containing the independent features. \n
        labels (pd.Series): The target variable. \n
        convert_to_long (bool, default=False): If True, it converts the values of the
                                            target variable to PyTorch `long` datatype. \n
        test_size (float, default=0.2): The proportion of the validation size. \n
        random_state (int, default=123): Used to ensure reproducibility. \n

    Returns:
        X_train (torch.Tensor): The training data. \n
        X_validation (torch.Tensor): The validation data. \n
        y_train (torch.Tensor): The labels of the training data. \n
        y_validation (torch.Tensor): The labels of the validation data. \n
    """
    logger = set_up_logger()
    # Convert to Tensors
    X = torch.Tensor(data.to_numpy())
    y = torch.Tensor(labels.to_numpy())
    if convert_to_long:
        y = torch.Tensor(labels.to_numpy()).long()

    # Split data
    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"X_train: {X_train.shape}\nX_validation: {X_validation.shape}\n")
    return (X_train, X_validation, y_train, y_validation)


def _is_tensor(*, X: Any) -> torch.Tensor:
    """This is used to validate an input. It returns True
    if the input is a Tensor otherwise False."""
    if not isinstance(X, torch.Tensor):
        raise TypeError("X is not of type `Tensor`.")
    return X


def _create_torch_dataset(
    *,
    X_train: torch.Tensor,
    X_validation: torch.Tensor,
    y_train: torch.Tensor,
    y_validation: torch.Tensor,
) -> tuple[TensorDataset, TensorDataset]:
    """This returns the created TensorDataset objects.

    Params:
        X_train (torch.Tensor): The training data. \n
        X_validation (torch.Tensor): The validation data. \n
        y_train (torch.Tensor): The labels of the training data. \n
        y_validation (torch.Tensor): The labels of the validation data. \n

    Returns:
        train_DL (TensorDataset): The training TensorDataset object. \n
        validation_DL (TensorDataset): The validation TensorDataset object. \n
    """
    X_train, y_train = _is_tensor(X=X_train), _is_tensor(X=y_train)
    X_validation, y_validation = _is_tensor(X=X_validation), _is_tensor(X=y_validation)

    train_data = TensorDataset(X_train, y_train)
    validation_data = TensorDataset(X_validation, y_validation)
    return (train_data, validation_data)


def create_data_loader(
    *,
    X_train: torch.Tensor,
    X_validation: torch.Tensor,
    y_train: torch.Tensor,
    y_validation: torch.Tensor,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """This returns the created DataLoader objects.

    Params:
        X_train (torch.Tensor): The training data. \n
        X_validation (torch.Tensor): The validation data. \n
        y_train (torch.Tensor): The labels of the training data. \n
        y_validation (torch.Tensor): The labels of the validation data. \n
        batch_size (int): The number of samples to used to calculate
                        the loss and update the model weights per epoch.

    Returns:
        train_DL (DataLoader): The training dataloader object. \n
        validation_DL (DataLoader): The validation dataloader object. \n
    """
    train_data, validation_data = _create_torch_dataset(
        X_train=X_train, X_validation=X_validation, y_train=y_train, y_validation=y_validation
    )
    train_DL = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_DL = DataLoader(
        dataset=validation_data, batch_size=validation_data.tensors[0].shape[0]
    )
    return (train_DL, validation_DL)
