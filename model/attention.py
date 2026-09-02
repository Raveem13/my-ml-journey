import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.key_linear = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query_linear = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.val_linear = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        keys = self.key_linear(embedded)
        querys = self.query_linear(embedded)
        values = self.val_linear(embedded)
        # print(keys)
        # print(querys)
        # print(values)
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        scores = querys @ torch.transpose(keys, 1, 2) / attention_dim ** 0.5
        context_len = scores.size(1)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,  
        low_tri = torch.tril(torch.ones(context_len, context_len))
        #    then masked_fill positions where mask == 0 with float('-inf')
        mask = (low_tri == 0)
        # print(mask)
        masked_scores = scores.masked_fill(mask, float('-inf'))
        # 4. Apply softmax(dim=2) to masked scores
        scores = torch.softmax(masked_scores, dim=2)
        # print(scores)
        # 5. Return (scores @ V) rounded to 4 decimal places
        out = scores @ values
        # print(out)
        return torch.round(out, decimals=4)
