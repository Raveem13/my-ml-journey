import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1    
        combined = positive + negative
        vocab    = sorted({word for sentiment in combined for word in sentiment.split()})
        # print(vocab)
        # 2. Encode each sentence by replacing words with their IDs
        word_ids = {word: (id+1) for id, word in enumerate(vocab)}
        # print(word_ids)
        # 3. Combine positive + negative into one list of tensors
        tokens = [torch.tensor([word_ids[word] for word in s.split()]) for s in combined]
        # print(tokens)
        # # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        padded = nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=0)
        return padded