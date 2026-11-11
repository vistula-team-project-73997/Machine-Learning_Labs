"""Exact Lab 3 CNN: 20,490 parameters; no dropout or batch normalization."""
from torch import nn


class CNN(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 10))
