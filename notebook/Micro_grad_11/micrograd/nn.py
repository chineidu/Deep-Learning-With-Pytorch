"""A simple neural network module.

Inspired by: (https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&ab_channel=AndrejKarpathy
"""

from abc import ABC, abstractmethod

import numpy as np
from micrograd.basetype import Value


class Module(ABC):
    def zero_grad(self) -> None:
        """Set all the gradients to zero."""
        for p in self.parameters():
            p.grad = 0.0

    @abstractmethod
    def parameters(self) -> list[Value]:
        return []


class Neuron(Module):
    """A single neuron."""

    def __init__(self, n_inputs: int, nonlinear: bool) -> None:
        """
        Params:
        -------
            n_inputs (int): The number of inputs (weights) to each neuron in the layer.
            nonlinear (bool): Whether the layer should be a linear or a non-linear layer.
        """
        self.weights = [
            Value(data=np.random.uniform(-1, 1)) for _ in np.arange(n_inputs)
        ]
        self.bias = Value(data=np.random.uniform(-1, 1))
        self.nonlinear = nonlinear

    def __repr__(self) -> str:
        return f"{'ReLU' if self.nonlinear else 'Linear'}Neuron({len(self.weights)})"

    def __call__(self, x: list[Value]) -> Value:
        """This is used to perform the forward pass."""
        assert (
            len(x) == len(self.weights)
        ), f"Number of inputs must match number of weights. Expected {len(self.weights)} inputs, got {len(x)}"

        # w1x1 + w2x2 + ... + wnxn + b
        activation: Value = (
            np.sum((wi * xi) for wi, xi in zip(self.weights, x)) + self.bias
        )
        output: Value = activation.relu() if self.nonlinear else activation
        return output

    def parameters(self) -> list[Value]:
        """Returns the parameters of the layer."""
        return self.weights + [self.bias]


class Layer(Module):
    """A layer of neurons."""

    def __init__(self, n_inputs: int, n_outputs: int, **kwargs) -> None:
        """
        Params:
        -------
            n_inputs (int): The number of inputs (weights) to each neuron in the layer.
            n_outputs (int): The number of outputs (neurons) from the layer.
            nonlinear (bool): Whether the layer should be a linear or a non-linear layer.
        """
        self.neurons = [Neuron(n_inputs, **kwargs) for _ in np.arange(n_outputs)]

    def __repr__(self) -> str:
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

    def __call__(self, x: list[Value]) -> Value | list[Value]:
        """This is used to perform the forward pass of the layer."""
        output: list[Value] = [n(x) for n in self.neurons]
        return output[0] if len(output) == 1 else output

    def parameters(self) -> list[Value]:
        """Returns the parameters of the layer."""
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    """A multi-layer perceptron."""

    def __init__(self, n_inputs: int, out_layers: list[int]) -> None:
        """
        Params:
        -------
            n_inputs (int): The number of inputs (weights) to each neuron in the layer.
            out_layers (list[int]): The number of outputs (neurons) per layer. e.g. [3,3,1] would create a 3-layer MLP with 3 neurons
            in the first layer, 3 neurons in the second layer, and 1 neuron in the third layer.
        """
        assert len(out_layers) > 0, "Must have at least one layer"
        self.size = [n_inputs] + out_layers
        self.layers = [
            Layer(self.size[i], self.size[i + 1]) for i in range(len(out_layers))
        ]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

    def __call__(self, x: list[Value]) -> Value | list[Value]:
        """Forward pass through the network."""
        for layer in self.layers:
            output: Value | list[Value] = layer(x)  # type: ignore
        return output

    def parameters(self) -> list[Value]:
        """Returns the parameters of the network."""
        return [p for layer in self.layers for p in layer.parameters()]
