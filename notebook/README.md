# Summary

## Table of Content

- [Summary](#summary)
  - [Table of Content](#table-of-content)
  - [Model Building Styles In PyTorch](#model-building-styles-in-pytorch)
    - [1. Using Sequential API](#1-using-sequential-api)
    - [2. Using Classes](#2-using-classes)
  - [ANN (Using Sequential API)](#ann-using-sequential-api)
  - [Common Operations](#common-operations)
    - [DataLoaders](#dataloaders)
    - [Use Dynamic Activation Functions](#use-dynamic-activation-functions)
    - [Use Dynamic Optimizers](#use-dynamic-optimizers)
    - [Save A Trained Model](#save-a-trained-model)
    - [Load A Trained Model](#load-a-trained-model)
    - [Save The Best Model Weights](#save-the-best-model-weights)
    - [Model Evaluation](#model-evaluation)
    - [Print Result With Carriage Return](#print-result-with-carriage-return)
    - [Create Custom Transform](#create-custom-transform)
    - [Create Custom Dataset](#create-custom-dataset)
  - [Regularization](#regularization)
    - [1. Dropout](#1-dropout)
    - [2. Weight Regularization](#2-weight-regularization)
      - [Ridge/L2](#ridgel2)
      - [Lasso](#lasso)
    - [3. Mini-Batch](#3-mini-batch)
  - [FFN](#ffn)
  - [Weight Inits](#weight-inits)
  - [Autoencoders](#autoencoders)
  - [Convolution](#convolution)

## Model Building Styles In PyTorch

### 1. Using Sequential API

```python
import torch.nn as nn

def build_model() -> Any:
    """This is used to build the model architecture."""
    clf = nn.Sequential(
        nn.Linear(2, 1),  # input
        nn.ReLU(),  # Activation
        nn.Linear(1, 1),  # Ouput
        nn.Sigmoid(),  # Final activation
    )
    return clf
```

### 2. Using Classes

- It can be simple or complicated.
- It requires some time to setup.
- It's very flexible.

```python
import torch.nn.functional as F

def Model(nn.Module) -> Any:
    """This is used to build the model architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Linear(2, 1)
        self.output = nn.Linear(1, 1)

    def forward(self, X) -> Any:
        """This is used to perform forward propagation."""
        X = self.input(X)
        X = F.relu(X)
        X = self.output(X)
        X = torch.sigmoid(X)

        return X
```

## ANN (Using Sequential API)

```python
def build_model(*, n_units: int) -> Any:
    """This is used to build the model architecture."""
    clf = nn.Sequential(
        nn.Linear(4, n_units),  # input
        nn.ReLU(),  # Activation
        nn.Linear(n_units, n_units),  # Hidden layer 1
        nn.ReLU(),  # Activation
        nn.Linear(n_units, 3),  # Ouput
        nn.Softmax(dim=1),  # Final activation
    )
    return clf


def train_model(
    *, model: Any, learning_rate: float, epochs: int, verbose: bool = True
) -> Any:
    """This is used to train the model."""
    # Optimizer
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    losses = torch.zeros(size=(epochs,))

    # Train model
    for epoch_idx in range(epochs):
        # Reset the gradients from prev. step loss.backward()
        optimizer.zero_grad()

        # Fwd prop
        _y_pred = model(X)

        # Compute loss
        loss = criterion(_y_pred, y)
        losses[epoch_idx] = loss

        # Back prop
        loss.backward()
        optimizer.step()

    if verbose:
        print("Training done ...")

    return model
```

## Common Operations

### DataLoaders

```python
from torch.utils.data import DataLoader, TensorDataset


# Create datasets using the train and test data
train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

# Convert the train and test data to DataLoader objects
train_DL = DataLoader(dataset=train_data, batch_size=4)
# Batch size is not required for the test_data
test_DL = DataLoader(dataset=test_data, batch_size=1)
```

### Use Dynamic Activation Functions

```python
# Examples of actvation functions
activation_funcs = ["ReLU", "ReLU6", "LeakyReLU"]


class Net(nn.Module):
    """This is an ANN architecture with an argument for activation functions."""

    def __init__(self, activation_func: str) -> None:
        super().__init__()
        self.activation_func = activation_func
        self.input = nn.Linear(11, 32)
        self.hidden = nn.Linear(32, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """This is used to implement forward propagation."""
        act_func = getattr(torch, self.activation_func)
        X = act_func(self.input(X))
        X = act_func(self.hidden(X))
        X = torch.sigmoid(self.output(X))
        return X
```

### Use Dynamic Optimizers

```python
def train_model(
    *,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    epochs: int,
    learning_rate: float,
    optimizer_name: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """This is used to train the ANN model."""
    net, PCT = Net(), 100
    criterion = nn.CrossEntropyLoss()
    _optimizer_ = getattr(torch.optim, optimizer_name)
    optimizer = _optimizer_(params=net.parameters(), lr=learning_rate)
    train_accuracy, validation_accuracy = np.zeros(shape=(epochs,)), np.zeros(
        shape=(epochs,)
    )
    train_loss = np.zeros(shape=(epochs,))

    for epoch_idx in np.arange(epochs):
        net.train()
        batch_loss, batch_accuracy = [], []

        for X_, y_ in train_dataloader:
            # Reset gradients
            optimizer.zero_grad()

              # Function body ....

    return (train_accuracy, validation_accuracy, train_loss, net)

OPTIMIZERS = ["SGD", "RMSprop", "Adam"]
```

### Save A Trained Model

```python
import torch

# Train the model
trained_net = train_model()

# Save the model
model_path = "ffn_model_1.pt"
torch.save(trained_net.state_dict(), model_path)
```

### Load A Trained Model

```python
import torch

# Initialize model
model_1 = Net()

# Load the model weights
model_1.load_state_dict(torch.load(model_path))
```

### Save The Best Model Weights

- Check this [notebook](https://github.com/chineidu/Deep-Learning-With-Pytorch/blob/main/notebook/06_FNN/02_saving_n_loading_models.ipynb).

### Model Evaluation

- Check this [notebook](https://github.com/chineidu/Deep-Learning-With-Pytorch/blob/main/notebook/06_FNN/03_model_evaluation.ipynb).

### Print Result With Carriage Return

```Python
import sys
import time


for idx in np.arange(1, 101):
    time.sleep(0.3)
    msg = f"{idx} / 100"
    sys.stdout.write("\r" + msg)

# Output
# 1/100 .... 100/100
```

### Create Custom Transform

```python
import numpy.typing as npt


class NumpyToTensor:
    """This is used to convert NumpPy arrays to PyTorch tensors."""

    def __call__(self, X: npt.NDArray[np.float_]):
        """This is used for applying the transformation."""
        X = torch.tensor(data=X)
        return X

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

# ==== Usage ====
A = np.random.random(size=(5, 4))
A_torch = NumpyToTensor()(X=A)

```

### Create Custom Dataset

```python
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision.transforms as T


class CustomDataset(Dataset):
    """This is a custom dataset class where tensors is a tuple containing the
    independent and dependent features."""

    def __init__(self, tensors: tuple[torch.Tensor, torch.Tensor], transform=None):
        # Check that sizes of data and labels match
        assert all(
            tensors[0].size(0) == t.size(0) for t in tensors
        ), "Size mismatch between tensors"

        # Assign inputs
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        """Return transformed version of X if there are transforms."""
        if self.transform:
            X = self.transform(self.tensors[0][index])
        else:
            X = self.tensors[0][index]

        # Return the label
        y = self.tensors[1][index]

        return (X, y)

    def __len__(self):
        return self.tensors[0].size(0)

# ==== Usage ====
# Create a list of transforms to apply to the image
image_transform = T.Compose(
    [
        T.ToPILImage(),
        T.RandomVerticalFlip(p=0.5),
        T.ToTensor(),
    ]
)

dataset = CustomDataset(tensors=(X, y), tranform=image_transform)
```

## Regularization

### 1. Dropout

```python
class Net(nn.Module):
    """This is an ANN architecture."""

    def __init__(self, n_units: int, dropout_rate: float) -> None:
        super().__init__()
        # Layers
        self.input = nn.Linear(2, n_units)
        self.hidden = nn.Linear(n_units, n_units)
        self.output = nn.Linear(n_units, 1)

        # Parameters
        self.dr = dropout_rate  # probability of dropout

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """This is used to perform forward propagation."""
        X = F.relu(self.input(X))

        # Dropout after the input layer
        # self.training automatically turns on/off during training/eval mode
        X = F.dropout(X, p=self.dr, training=self.training)

        # Dropout after the hidden layer
        X = F.relu(self.hidden(X))
        X = F.dropout(X, p=self.dr, training=self.training)

        # No Dropout at the output layer
        X = torch.sigmoid(self.output(X))

        return X

def train_model(
    *,
    train_data_loader: DataLoader,
    validation_data_loader: DataLoader,
    dropout_rate: float,
    n_units: int = 64,
) -> tuple[list[Any], list[Any]]:
    """This is used to train the classifier."""
    net = Net(n_units=n_units, dropout_rate=dropout_rate)

    learning_rate, epochs = 0.01, 500
    THRESH, PCT = 0.5, 100
    optimizer = torch.optim.SGD(params=net.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()  # Used for binary outputs
    train_accuracy, validation_accuracy = [], []

    for epoch_idx in np.arange(epochs):
        batch_accuracy = []
        net.train()  # Activate regularization

        for X_, y_ in train_data_loader:
            # Reset gradients
            optimizer.zero_grad()

            # Compute forward prop and loss
            _y_proba = net(X_)
            loss = criterion(_y_proba, y_)

            # Compute backward prop
            loss.backward()
            optimizer.step()

            # Compute batch accuracy
            acc = torch.mean(((_y_proba > THRESH) == y_).float()) * PCT
            batch_accuracy.append(acc.detach())

        # Compute training accuracy
        train_accuracy.append(np.mean(batch_accuracy))

        # Compute validation accuracy
        net.eval()  # Deactivate regularization

        X_val, y_val = next(iter(validation_data_loader))
        y_proba_val = net(X_val)
        _val_acc = torch.mean(((y_proba_val > THRESH) == y_val).float()) * PCT
        validation_accuracy.append(_val_acc.detach())

    return (train_accuracy, validation_accuracy)
```

### 2. Weight Regularization

#### Ridge/L2

```python
class Net(nn.Module):
    """This is an ANN architecture. The output layer has 3 units."""

    def __init__(self, n_units: int) -> None:
        super().__init__()
        # Layers
        self.input = nn.Linear(4, n_units)
        self.hidden_1 = nn.Linear(n_units, n_units)
        self.hidden_2 = nn.Linear(n_units, n_units)
        self.hidden_3 = nn.Linear(n_units, n_units)
        self.output = nn.Linear(n_units, 3)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """This is used to perform forward propagation."""
        X = F.relu(self.input(X))
        X = F.relu(self.hidden_1(X))
        X = F.relu(self.hidden_2(X))
        X = F.relu(self.hidden_3(X))
        X = torch.softmax(self.output(X), dim=1)

        return X


def train_iris_model(
    *,
    train_data_loader: DataLoader,
    validation_data_loader: DataLoader,
    l2_lambda: float,
    n_units: int = 64,
) -> tuple[list[float], list[float], list[float]]:
    """This is used to train the classifier with L2 regularization.."""
    net = Net(n_units=n_units)

    learning_rate, epochs = 0.01, 800
    PCT = 100
    optimizer = torch.optim.SGD(
        params=net.parameters(),
        lr=learning_rate,
        weight_decay=l2_lambda,  # L2 regularization
    )
    criterion, losses = nn.CrossEntropyLoss(), []
    train_accuracy, validation_accuracy = [], []

    for _ in np.arange(epochs):
        batch_accuracy, batch_loss = [], []
        net.train()  # Activate regularization

        # Function body ....

    return (train_accuracy, validation_accuracy, losses)
```

#### Lasso

```python
def train_iris_model(
    *,
    train_data_loader: DataLoader,
    validation_data_loader: DataLoader,
    l1_lambda: float,
    n_units: int = 64,
) -> tuple[list[float], list[float], list[float]]:
    """This is used to train the classifier with L1 regularization."""
    net = Net(n_units=n_units)

    learning_rate, epochs = 0.005, 1_000
    PCT = 100
    optimizer = torch.optim.SGD(params=net.parameters(), lr=learning_rate)
    criterion, losses = nn.CrossEntropyLoss(), []
    train_accuracy, validation_accuracy = [], []

    # Calculate the number of weights: exclude the bias.
    num_weights = 0
    for param_, weight_ in net.named_parameters():
        if "bias" not in param_:
            num_weights += weight_.numel()

    # Iterate over the epochs
    for _ in np.arange(epochs):
        batch_accuracy, batch_loss = [], []
        net.train()  # Activate regularization

        # Iteration for the batch data
        for X_, y_ in train_data_loader:
            # Reset gradients
            optimizer.zero_grad()

            # Compute forward prop and loss
            _y_proba = net(X_)
            loss = criterion(_y_proba, y_)

            # Compute L1 Regularization
            # Initialize L1 term, compute the absoute sum of the weights
            l1_term = torch.tensor(data=0.0, requires_grad=True)

            for param_, weight_ in net.named_parameters():
                if "bias" not in param_:
                    l1_term = l1_term + torch.sum(torch.abs(weight_))
            # Add the mean value to the loss
            loss = loss + (l1_lambda * l1_term / num_weights)

            # Compute backward prop
            loss.backward()
            optimizer.step()

            # Compute batch accuracy
            _y_pred = torch.argmax(_y_proba, axis=1)
            acc = torch.mean((_y_pred == y_).float()) * PCT
            batch_accuracy.append(acc.detach())
            batch_loss.append(loss.detach())

        # Compute training loss and accuracy
        train_accuracy.append(np.mean(batch_accuracy))
        losses.append(np.mean(batch_loss))

        # Compute validation accuracy
        net.eval()  # Deactivate regularization

        X_val, y_val = next(iter(validation_data_loader))
        _y_pred_val = torch.argmax(net(X_val), axis=1)
        _val_acc = torch.mean((_y_pred_val == y_val).float()) * PCT
        validation_accuracy.append(_val_acc.detach())

    return (train_accuracy, validation_accuracy, losses)
```

### 3. Mini-Batch

```python
class Net(nn.Module):
    """This is an ANN architecture. The output layer has 3 units."""

    def __init__(self, n_units: int) -> None:
        super().__init__()
        # Layers
        self.input = nn.Linear(4, n_units)
        self.hidden = nn.Linear(n_units, n_units)
        self.output = nn.Linear(n_units, 3)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """This is used to perform forward propagation."""
        X = F.relu(self.input(X))
        X = F.relu(self.hidden(X))
        X = torch.softmax(self.output(X), dim=1)

        return X


def train_iris_model(
    *,
    train_data_loader: DataLoader,
    validation_data_loader: DataLoader,
    n_units: int = 64,
) -> tuple[list[float], list[float], list[float]]:
    """This is used to train the classifier with mini-batch."""
    net = Net(n_units=n_units)

    learning_rate, epochs = 0.01, 500
    PCT = 100
    optimizer = torch.optim.SGD(params=net.parameters(), lr=learning_rate)
    criterion, losses = nn.CrossEntropyLoss(), []
    train_accuracy, validation_accuracy = [], []

    for _ in np.arange(epochs):
        batch_accuracy, batch_loss = [], []
        net.train()  # Activate regularization

        # Iteration for the batch data
        for X_, y_ in train_data_loader:
            # Reset gradients
            optimizer.zero_grad()

            # Compute forward prop and loss
            _y_proba = net(X_)
            loss = criterion(_y_proba, y_)
            batch_loss.append(loss.detach())

            # Compute backward prop
            loss.backward()
            optimizer.step()

            # Compute batch accuracy
            _y_pred = torch.argmax(_y_proba, axis=1)
            acc = torch.mean((_y_pred == y_).float()) * PCT
            batch_accuracy.append(acc.detach())

        # Compute training loss and accuracy
        train_accuracy.append(np.mean(batch_accuracy))
        losses.append(np.mean(batch_loss))

        # Compute validation accuracy
        net.eval()  # Deactivate regularization

        X_val, y_val = next(iter(validation_data_loader))
        _y_pred_val = torch.argmax(net(X_val), axis=1)
        _val_acc = torch.mean((_y_pred_val == y_val).float()) * PCT
        validation_accuracy.append(_val_acc.detach())

    return (train_accuracy, validation_accuracy, losses)
```

## FFN

```python
# Build Model
class FFN(nn.Module):
    """This is used to build a Feed Forward Network architecture that
    is used for classification."""

    def __init__(self, input_size: int, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This performs the forward propagation."""
        # Flatten the input images
        x = x.view(-1, (28 * 28))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ==== Init model ====
model = FFN(
    input_size=input_size,
    hidden_size=hidden_size,
    num_classes=num_classes,
).to(device=device)

# ==== Define loss function and optimizer ====
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

# ==== Training loop ====
for epoch in tqdm(range(num_epochs)):
    model.train()
    running_loss = 0.0

    # ==== Batch training loop ====
    for images, labels in train_loader:
        # Push the data to GPU if available
        images, labels = images.to(device), labels.to(device)

        # ==== Forwardprop ====
        outputs = model(images)
        loss: nn.CrossEntropyLoss = criterion(outputs, labels)

        # ==== Backprop ====
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Update the loss
        running_loss += loss.item()

    # ==== Validation loop ====
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            # Push the data to GPU if available
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss = criterion(outputs, labels).item()
            # It returns the value and the index.
            # We're interested in the index
            _, predicted = torch.max(outputs, dim=1)
            val_total += labels.size(0)  # or labels.shape[0]
            val_correct += (predicted == labels).sum().item()
    val_accuracy = (val_correct / val_total) * 100

    print(
        f"Epoch {epoch + 1}/{num_epochs}, "
        f"Training Loss: {running_loss / len(train_loader)}, "
        f"Validation Loss: {val_loss / len(val_loader)}, "
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )

# Test the model on the test dataset
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        # Push the data to GPU if available
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        # It returns the value and the index.
        # We're interested in the index
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = (test_correct / test_total) * 100
print(f"Test Accuracy: { test_accuracy:.2f}%")
```

- Check this [notebook](https://github.com/chineidu/Deep-Learning-With-Pytorch/blob/main/notebook/06_FNN/04_FFNs.ipynb) for more.

## Weight Inits

## Autoencoders

```python
class CNNAutoencoder(nn.Module):
    """This is used to build a Convolutional Neutral Network Autoencoder
    architecture that is used for image compression.

    Params:
        input_size: This is the output of the final conv layer.
        num_classes (int): This is the number of class labels in the input data.
    """

    def __init__(self) -> None:
        super().__init__()
        # ====== Encoder ======
        # 16 to 64
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1
            ),
            nn.ReLU(),
        )

        # ====== Decoder ======
        # 64 to 16
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=3,
                kernel_size=5,
                stride=2,
                padding=1,
                output_padding=1,  # This ensures the original image == decoded image
            ),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This performs the forward propagation."""
        x = self.encoder(x)
        x = self.decoder(x)

        return x
```

## Convolution

```python
class CNN(nn.Module):
    """This is used to build a Convolutional Neutral Network architecture that
    is used for classification.

    NB:
        input_size: This is the output of the final conv layer.
    """

    def __init__(self, input_size: int, num_classes: int) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(
            in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_2 = nn.Conv2d(
            in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=1
        )
        self.fc1 = nn.Linear((20 * input_size * input_size), 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This performs the forward propagation."""

        # Conv -> pool
        x = self.pool(F.relu(self.conv_1(x)))
        x = self.pool(F.relu(self.conv_2(x)))

        # Flatten the input images to have the dim:
        # (num_samples, n_channels, n_input, n_input)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

def train(
    model: CNN,
    device: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: Any,
    optimizer: torch.optim,
    num_epochs: int,
) -> CNN:
    """This is used for training the model."""
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0

        # ==== Batch training loop ====
        for images, labels in train_loader:
            # Push the data to GPU if available
            images, labels = images.to(device), labels.to(device)

            # ==== Forwardprop ====
            outputs = model(images)
            loss: nn.CrossEntropyLoss = criterion(outputs, labels)

            # ==== Backprop ====
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Update the loss
            running_loss += loss.item()

        # ==== Validation loop ====
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                # Push the data to GPU if available
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss = criterion(outputs, labels).item()
                # It returns the value and the index.
                # We're interested in the index
                _, predicted = torch.max(outputs, dim=1)
                val_total += labels.size(0)  # or labels.shape[0]
                val_correct += (predicted == labels).sum().item()
        val_accuracy = (val_correct / val_total) * 100

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Training Loss: {running_loss / len(train_loader)}, "
            f"Validation Loss: {val_loss / len(val_loader)}, "
            f"Validation Accuracy: {val_accuracy:.2f}%"
        )
    return model


def test(model: CNN, device: Any, test_loader: DataLoader):
    """This is used to the model on the test dataset."""
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            # Push the data to GPU if available
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = (test_correct / test_total) * 100
    print(f"Test Accuracy: {test_accuracy:.2f}%")
```
