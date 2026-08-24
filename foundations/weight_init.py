import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        std = math.sqrt(2 / (fan_in + fan_out))
        # Use torch.manual_seed(0) for reproducibility
        torch.manual_seed(0)
        weights = torch.randn(fan_out, fan_in) * std
        # Round to 4 decimal places and return as nested list
        return torch.round(weights, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        std = math.sqrt(2 / fan_in)
        # Use torch.manual_seed(0) for reproducibility        
        torch.manual_seed(0)
        weights = torch.randn(fan_out, fan_in) * std
        # Round to 4 decimal places and return as nested list        
        return torch.round(weights, decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        torch.manual_seed(0)
        
        dims = [input_dim] + [hidden_dim] * num_layers
        weights = []
        for i in range(num_layers):
            if init_type == 'xavier':
                std = math.sqrt(2 / (dims[i] + dims[i+1]))
            elif init_type == 'kaiming':
                std = math.sqrt(2 / dims[i])
            else:
                std = 1.0

            w = torch.randn(dims[i], dims[i+1]) * std
            weights.append(w)

        x = torch.randn(1, input_dim)
        stds = []

        for w in weights:
            x = x @ w.T
            x = torch.relu(x)
            s = round(x.std().item(),2)
            stds.append(s)
        # Return the std of activations after each layer, rounded to 2 decimals.
        return stds