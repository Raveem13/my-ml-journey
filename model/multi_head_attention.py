import torch
import torch.nn as nn
from torchtyping import TensorType

class MultiHeadedSelfAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        # Create num_heads SingleHeadAttention instances using nn.ModuleList
        # Each head size = attention_dim // num_heads
        # Use: self.SingleHeadAttention(embedding_dim, head_size)
        self.attn_heads = nn.ModuleList()
        for head in range(num_heads):
            single_head = self.SingleHeadAttention(embedding_dim, (attention_dim // num_heads))
            self.attn_heads.append(single_head)
        # After the heads, add an output projection: nn.Linear(attention_dim, attention_dim, bias=False)
        self.proj_out = nn.Linear(attention_dim, attention_dim, bias=False)


    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # Run each head on the input, concatenate outputs along dim=2
        head_outs = []
        for head in self.attn_heads:
            head_outs.append(head(embedded))
        concat_heads = torch.cat(head_outs, dim=2)
        # Pass concatenated result through the output projection (W_O)
        res = self.proj_out(concat_heads)
        # Return result rounded to 4 decimal places
        return torch.round(res, decimals=4)

    class SingleHeadAttention(nn.Module):
        def __init__(self, embedding_dim: int, attention_dim: int):
            super().__init__()
            torch.manual_seed(0)
            self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            k = self.key_gen(embedded)
            q = self.query_gen(embedded)
            v = self.value_gen(embedded)

            scores = q @ torch.transpose(k, 1, 2) # @ is the same as torch.matmul()
            context_length, attention_dim = k.shape[1], k.shape[2]
            scores = scores / (attention_dim ** 0.5)

            lower_triangular = torch.tril(torch.ones(context_length, context_length))
            mask = lower_triangular == 0
            scores = scores.masked_fill(mask, float('-inf'))
            scores = nn.functional.softmax(scores, dim = 2)

            return scores @ v
