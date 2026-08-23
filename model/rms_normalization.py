import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        x = np.array(x)
        gamma = np.array(gamma)
        ep = 1e-5
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        mean_sqr = np.mean(x**2)
        rms = np.sqrt(mean_sqr + ep)
        # Normalize x, then scale by gamma
        x_hat = x/rms
        out = x_hat * gamma
        # Return result rounded to 4 decimal places as a list
        return np.round(out, 4)
