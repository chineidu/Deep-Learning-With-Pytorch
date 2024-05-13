"""This is a simple implementation of a backpropagation library. It performs the forward pass of a computational graph and
stores the operations that are performed on the data. It then performs the backward pass to calculate the gradient of the
computational graph.

Inspired by: (https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&ab_channel=AndrejKarpathy
"""

from typing import Any

import numpy as np


class Value:
    data_error: str = "only supporting int/float values"

    def __init__(
        self,
        data: int | float,
        _children: tuple[Any] = tuple(),  # type: ignore
        _op: str = "",
        label: str = "",
    ) -> None:
        assert isinstance(data, (int, float)), self.data_error
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0  # No gradient yet
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(data={self.data})"

    def __add__(self, other: "Value") -> "Value":
        """Addition of two values."""
        other = other if isinstance(other, Value) else Value(other)  # type: ignore
        out: Value = Value(self.data + other.data, (self, other), "+")  # type: ignore

        def _backward() -> None:
            """Closure for calculating gradient of an addition operation."""
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: "Value") -> "Value":
        """Addition of two values. i.e. `other + self`"""
        # return self.__add__(other)
        return self + other  # This is the same as above

    def __mul__(self, other: "Value") -> "Value":
        """Multiplication of two values."""
        other = other if isinstance(other, Value) else Value(other)  # type: ignore
        out: Value = Value(self.data * other.data, (self, other), "*")  # type: ignore

        def _backward() -> None:
            """Closure for calculating gradient of a multiplication operation.
            NB: The input values are flipped because of the multiplication.
            """
            self.grad += out.grad * other.data  # type: ignore
            other.grad += out.grad * self.data  # type: ignore

        out._backward = _backward
        return out

    def __rmul__(self, other: "Value") -> "Value":
        """Multiplication of two values. i.e. `other * self`"""
        return self * other

    def __neg__(self) -> int | float:
        """Negation of value."""
        return self * -1  # type: ignore

    def __sub__(self, other: "Value") -> "Value":
        """Subtraction of two values."""
        other = other if isinstance(other, Value) else Value(other)  # type: ignore
        out: Value = Value(self.data + (-other.data), (self, other), "-")  # type: ignore

        def _backward() -> None:
            """Closure for calculating gradient of a subtraction operation."""
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __rsub__(self, other: "Value") -> "Value":
        """Subtraction of two values. i.e. `other - self`"""
        return other + (-self)  # type: ignore

    def __pow__(self, other: "Value") -> "Value":
        """Exponentiation of value."""
        assert isinstance(other, (int, float)), self.data_error
        other = other if isinstance(other, Value) else Value(other)
        out: Value = Value(self.data**other.data, (self,), f"**{other.data}")

        def _backward() -> None:
            """Closure for calculating gradient of an exponentiation operation."""
            self.grad += out.grad * (other.data * self.data ** (other.data - 1))

        out._backward = _backward
        return out

    def __truediv__(self, other: "Value") -> int | float:
        """Division of value. i.e. `self / other`
        Note:
        -----
            2 / 3
            2 * (1 / 3)
            2 * (3^-1)
        """
        out = self * other**-1  # type: ignore
        return out

    def __rtruediv__(self, other: "Value") -> int | float:
        """`other / self`"""
        out = other * self**-1  # type: ignore
        return out

    def tanh(self) -> "Value":
        """Hyperbolic tangent of value."""
        x: int | float = self.data
        tanh = (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)
        out: Value = Value(tanh, _children=(self,), _op="tanh")

        def _backward() -> None:
            """Closure for calculating gradient of a hyperbolic tangent operation."""
            self.grad += (1 - tanh**2) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Value":
        """Exponent of value."""
        exp: int | float = np.exp(self.data)
        out: Value = Value(exp, _children=(self,), _op="exp")

        def _backward() -> None:
            """Closure for calculating gradient of an exponential operation."""
            self.grad += out.grad * exp  # type: ignore

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Propagate gradient backwards through the computation graph."""
        topology: list[Any] = []
        visited: set[Any] = set()

        def build_topology(node: Value) -> None:
            """Build a topology of the computation graph."""
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topology(child)
                topology.append(node)

        build_topology(node=self)
        # Apply backprop to all the nodes in the network, starting from the output node.
        # Base case
        self.grad = 1.0  # type: ignore
        for node in reversed(topology):
            node._backward()
