# Copied!
import os
from typing import Any, Optional, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn


def plot_training_loss(
    minibatch_loss_list: list[int],
    num_epochs: int,
    iter_per_epoch: int,
    results_dir=None,
    averaging_iterations: int = 100,
):
    """Plots the training loss over the course of training.

    Params:
    -------
        minibatch_loss_list (list[int]): A list of the loss values for each minibatch during training.
        num_epochs (int): The total number of epochs the model was trained for.
        iter_per_epoch (int): The number of iterations (minibatches) per epoch.
        results_dir (str, optional): The directory to save the plot to. If not provided, the plot will not be saved.
        averaging_iterations (int, optional): The number of iterations to average the loss over for the running average line. Defaults to 100.

    Returns:
    --------
        None
    """

    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(
        range(len(minibatch_loss_list)), (minibatch_loss_list), label="Minibatch Loss"
    )

    if len(minibatch_loss_list) > 1000:
        ax1.set_ylim([0, np.max(minibatch_loss_list[1000:]) * 1.5])
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Loss")

    ax1.plot(
        np.convolve(
            minibatch_loss_list,
            np.ones(
                averaging_iterations,
            )
            / averaging_iterations,
            mode="valid",
        ),
        label="Running Average",
    )
    ax1.legend()

    ###################
    # Set scond x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs + 1))

    newpos = [e * iter_per_epoch for e in newlabel]

    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])

    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.spines["bottom"].set_position(("outward", 45))
    ax2.set_xlabel("Epochs")
    ax2.set_xlim(ax1.get_xlim())
    ###################

    plt.tight_layout()

    if results_dir is not None:
        image_path = os.path.join(results_dir, "plot_training_loss.pdf")
        plt.savefig(image_path)


def plot_accuracy(
    train_acc_list: list[float],
    valid_acc_list: list[float],
    results_dir: str | None = None,
):
    num_epochs = len(train_acc_list)

    plt.plot(np.arange(1, num_epochs + 1), train_acc_list, label="Training")
    plt.plot(np.arange(1, num_epochs + 1), valid_acc_list, label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()

    if results_dir is not None:
        image_path = os.path.join(results_dir, "plot_acc_training_validation.pdf")
        plt.savefig(image_path)


Model: TypeAlias = nn.Module


def show_examples(
    model,
    data_loader: DataLoader,
    unnormalizer: Any | None = None,
    class_dict: dict[str, Any] | None = None,
):
    fail_features, fail_targets, fail_predicted = [], [], []
    for batch_idx, (features, targets) in enumerate(data_loader):
        with torch.no_grad():
            logits = model(features)
            predictions = torch.argmax(logits, dim=1)

            mask = targets != predictions

            fail_features.extend(features[mask])
            fail_targets.extend(targets[mask])
            fail_predicted.extend(predictions[mask])

        if len(fail_targets) > 15:
            break

    fail_features: torch.Tensor = torch.cat(fail_features)  # type: ignore
    fail_targets: torch.Tensor = torch.tensor(fail_targets)  # type: ignore
    fail_predicted: torch.Tensor = torch.tensor(fail_predicted)  # type: ignore

    fig, axes = plt.subplots(nrows=3, ncols=5, sharex=True, sharey=True)

    if unnormalizer is not None:
        for idx in range(fail_features.shape[0]):  # type: ignore
            features[idx] = unnormalizer(fail_features[idx])

    if fail_features.ndim == 4:  # type: ignore
        nhwc_img = np.transpose(fail_features, axes=(0, 2, 3, 1))
        nhw_img = np.squeeze(nhwc_img.numpy(), axis=3)

        for idx, ax in enumerate(axes.ravel()):
            ax.imshow(nhw_img[idx], cmap="binary")
            if class_dict is not None:
                ax.title.set_text(
                    f"P: {class_dict[fail_predicted[idx].item()]}"
                    f"\nT: {class_dict[fail_targets[idx].item()]}"
                )
            else:
                ax.title.set_text(f"P: {fail_predicted[idx]} | T: {fail_targets[idx]}")
            ax.axison = False

    else:
        for idx, ax in enumerate(axes.ravel()):
            ax.imshow(fail_features[idx], cmap="binary")
            if class_dict is not None:
                ax.title.set_text(
                    f"P: {class_dict[fail_predicted[idx].item()]}"
                    f"\nT: {class_dict[fail_targets[idx].item()]}"
                )
            else:
                ax.title.set_text(f"P: {fail_predicted[idx]} | T: {targets[idx]}")
            ax.axison = False
    plt.tight_layout()
    plt.show()


def show_failures(
    model: torch.nn.Module,
    data_loader: DataLoader,
    unnormalizer: Any | None = None,
    class_dict: Optional[dict[int, str]] = None,
    nrows: int = 3,
    ncols: int = 5,
    figsize: Optional[tuple[int, int]] = None,
) -> tuple[plt.Figure, np.ndarray]:
    failure_features: list[torch.Tensor] = []
    failure_pred_labels: list[int] = []
    failure_true_labels: list[int] = []

    for batch_idx, (features, targets) in enumerate(data_loader):
        with torch.inference_mode():
            features = features
            targets = targets
            logits = model(features)
            predictions = torch.argmax(logits, dim=1)

        for i in range(features.shape[0]):
            if targets[i] != predictions[i]:
                failure_features.append(features[i])
                failure_pred_labels.append(predictions[i])
                failure_true_labels.append(targets[i])

        if len(failure_true_labels) >= nrows * ncols:
            break

    features = torch.stack(failure_features, dim=0)
    targets = torch.tensor(failure_true_labels)
    predictions = torch.tensor(failure_pred_labels)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=figsize
    )

    if unnormalizer is not None:
        for idx in range(features.shape[0]):
            features[idx] = unnormalizer(features[idx])
    nhwc_img = np.transpose(features, axes=(0, 2, 3, 1))

    if nhwc_img.shape[-1] == 1:
        nhw_img = np.squeeze(nhwc_img.numpy(), axis=3)

        for idx, ax in enumerate(axes.ravel()):
            ax.imshow(nhw_img[idx], cmap="binary")
            if class_dict is not None:
                ax.title.set_text(
                    f"P: {class_dict[predictions[idx].item()]}"
                    f"\nT: {class_dict[targets[idx].item()]}"
                )
            else:
                ax.title.set_text(f"P: {predictions[idx]} | T: {targets[idx]}")
            ax.axison = False

    else:
        for idx, ax in enumerate(axes.ravel()):
            ax.imshow(nhwc_img[idx])
            if class_dict is not None:
                ax.title.set_text(
                    f"P: {class_dict[predictions[idx].item()]}"
                    f"\nT: {class_dict[targets[idx].item()]}"
                )
            else:
                ax.title.set_text(f"P: {predictions[idx]} | T: {targets[idx]}")
            ax.axison = False
    return fig, axes
