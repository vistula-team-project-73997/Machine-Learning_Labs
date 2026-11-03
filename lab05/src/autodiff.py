"""A scalar reverse-mode AD engine; no NumPy or PyTorch dependencies."""
import math


class Value:
    def __init__(self, data, parents=(), op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self.parents = tuple(parents)
        self.op = op
        self.label = label
        self._backward = lambda: None

    @staticmethod
    def wrap(x):
        return x if isinstance(x, Value) else Value(x)

    def __add__(self, other):
        other = self.wrap(other)
        out = Value(self.data + other.data, (self, other), "+")
        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = backward
        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = self.wrap(other)
        a, b = self.data, other.data
        out = Value(a * b, (self, other), "*")
        def backward():
            self.grad += b * out.grad
            other.grad += a * out.grad
        out._backward = backward
        return out

    __rmul__ = __mul__

    def __pow__(self, power):
        if not isinstance(power, (float, int)):
            raise TypeError("Only constant real exponents are supported")
        x = self.data
        result = x ** power
        if isinstance(result, complex):
            raise ValueError("Result is outside the real domain")
        slope = 0.0 if power == 0 else power * x ** (power - 1)
        out = Value(result, (self,), f"**{power}")
        def backward():
            self.grad += slope * out.grad
        out._backward = backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -self.wrap(other)

    def __rsub__(self, other):
        return self.wrap(other) + -self

    def __truediv__(self, other):
        return self * self.wrap(other) ** -1

    def __rtruediv__(self, other):
        return self.wrap(other) * self ** -1

    def relu(self):
        positive = self.data > 0
        out = Value(max(0.0, self.data), (self,), "relu")
        def backward():
            self.grad += positive * out.grad
        out._backward = backward
        return out

    def sigmoid(self):
        x = self.data
        s = 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))
        out = Value(s, (self,), "sigmoid")
        def backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = backward
        return out

    def log(self):
        x = self.data
        out = Value(math.log(x), (self,), "log")
        def backward():
            self.grad += out.grad / x
        out._backward = backward
        return out

    def topological(self):
        """Parents precede their users. Iterative DFS supports long sum chains."""
        ordered, seen = [], set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                ordered.append(node)
            elif node not in seen:
                seen.add(node)
                stack.append((node, True))
                stack.extend((p, False) for p in reversed(node.parents))
        return ordered

    def backward(self, clear=True, retain_graph=True):
        """Seed output with 1, then accumulate each edge in reverse order.

        Default clears the whole graph. clear=False preserves leaf gradients
        across calls, while always resetting intermediate gradients. This avoids
        multiplying old intermediate adjoints on a retained graph.
        retain_graph=False releases closures/edges after this backward pass.
        Do not backpropagate through a released graph again.
        """
        nodes = self.topological()
        for node in nodes:
            if clear or node.parents:
                node.grad = 0.0
        self.grad = 1.0
        for node in reversed(nodes):
            node._backward()
        if not retain_graph:
            for node in nodes:
                node._backward = lambda: None
                node.parents = ()

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g}, op={self.op!r})"
