import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        p_arr = np.exp(z - max(z))
        res =  p_arr/ p_arr.sum()
        # return np.round(your_answer, 4)
        return np.round(res, 4)
        pass
