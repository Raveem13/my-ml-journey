import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)
        
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        z1 = np.matmul(x, W1.T) + b1
        a1 = np.maximum(0,z1)
        z2 = np.matmul(a1, W2.T) + b2

        # Loss: MSE = mean((predictions - y_true)^2)
        error = (z2 - y_true)
        mse = np.mean(np.square(error))

        loss = float(mse)

        dz2 = 2 * error / (len(y_true))
        db2 = dz2
        dW2 = dz2.reshape(-1,1) * a1.reshape(1,-1)

        da1 = dz2.reshape(1,-1) * W2
        da1 = da1.flatten()
        dz1 = da1 * (z1 > 0).astype(float)

        db1 = dz1
        dW1 = dz1.reshape(-1,1) * x.reshape(1,-1)
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        return {
            'loss': np.round(loss, 4),
            'dW1':  np.round(dW1, 4).tolist(),
            'db1':  np.round(db1, 4).tolist(),
            'dW2':  np.round(dW2, 4).tolist(),
            'db2':  np.round(db2, 4).tolist()
        }
        pass
