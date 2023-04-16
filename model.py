import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(),
            nn.SELU(),
            nn.Linear(),
            nn.SELU(),
            nn.Linear(),
            nn.SELU()
        )

    def forward(self, x):
        return self.linear_relu_stack(x)



