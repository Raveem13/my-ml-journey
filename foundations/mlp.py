import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        h_layer = x   # Initial hidden layer
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        for index in range(len(weights)):
            z = np.matmul(h_layer, weights[index]) + biases[index]
        # Apply ReLU after each hidden layer, no activation on output layer
            if index < len(weights)-1:
                 h_layer = np.maximum(0, z) 
        # return np.round(your_answer, 5)
        return np.round(z, 5)