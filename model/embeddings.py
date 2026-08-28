import numpy as np
from numpy.typing import NDArray


class Solution:
    def lookup(self, embeddings: NDArray[np.float64], token_ids: NDArray[np.int64]) -> NDArray[np.float64]:
        # embeddings: (vocab_size, embed_dim) matrix
        # token_ids: 1D array of integer token IDs
        indices = token_ids.tolist()
        # Return the embedding vectors for the given token IDs
        emb_vec = embeddings[indices]
        # return np.round(your_answer, 5)
        return np.round(emb_vec, 5)
