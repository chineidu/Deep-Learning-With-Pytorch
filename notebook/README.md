# Summary

## Table of Content

- [Summary](#summary)
  - [Table of Content](#table-of-content)
  - [Model Building Styles In PyTorch](#model-building-styles-in-pytorch)
    - [1. Using Sequential API](#1-using-sequential-api)
    - [2. Using Classes](#2-using-classes)
  - [ANN Using Sequential API](#ann-using-sequential-api)
  - [Common Operations](#common-operations)
    - [DataLoaders](#dataloaders)
  - [Regularization](#regularization)
    - [1. Dropout](#1-dropout)
    - [2. Weight Regularization](#2-weight-regularization)
      - [(Ridge/L2)](#ridgel2)
      - [Lasso](#lasso)
    - [3. Mini-Batch](#3-mini-batch)

## Model Building Styles In PyTorch

### 1. Using Sequential API

```python
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

## ANN Using Sequential API

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
# Create datasets using the train and test data
train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

# Convert the train and test data to DataLoader objects
train_DL = DataLoader(dataset=train_data, batch_size=4)
# Batch size is not required for the test_data
test_DL = DataLoader(dataset=test_data, batch_size=1)
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

#### (Ridge/L2)

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
