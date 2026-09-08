from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        result = []
        for n in numbers:
            text = str(n)
            tokens = self.greedy_tokenize(text, vocab)
            result.append(tokens)
        # Return a list of token lists showing how each number gets split.
        print(result)
        return result

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        tokens = self.greedy_tokenize(text, vocab)
        return len(tokens)

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        tokens = self.greedy_tokenize(text, vocab)
        words = text.split()
        tkn_fertility = len(tokens) / len(words)
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        return round(tkn_fertility, 4)

    def greedy_tokenize(self, text: str, vocab: Dict[str, int]) -> List[str]:
        tokens = []
        i = 0
        while i < len(text):
            for l in range(len(text), 0, -1):
                substring = text[i:i+l]
                # print(substring)
                if substring in vocab:
                    best = substring
                    print(best)
                    break
            if best == None:
                i += 1
            else:
                tokens.append(best)
                i += len(best)
        return tokens