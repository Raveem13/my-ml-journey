import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Initialize w = zeros, b = 0
        w = np.zeros(X.shape[1])
        b = 0
        n = X.shape[0]

        for ep in range(epochs):
        # Model: y_hat = X @ w + b
            y_hat = X @ w + b
            # Loss: MSE = (1/n) * sum((y_hat - y)^2)
            error = y_hat - y
            loss = np.mean(error**2)

            dL_dw = (2/n) * X.T @ error
            dL_db = 2 * np.mean(error)

            w = w - lr * dL_dw
            b = b - lr * dL_db
        # return/n (np.round(w, 5), round(b, 5))
        return (np.round(w, 5), round(b, 5))