from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        tokens = list(corpus)
        merges = []
        # 2. For each merge step:
        for m in range(num_merges):
        #    a. Count frequency of all adjacent token pairs
            count = {}
            for i in range(len(tokens)-1):
                pair = (tokens[i], tokens[i+1])
                count[pair] = count.get(pair, 0) + 1
            # print(count)
        #    b. Find the most frequent pair (break ties lexicographically)
            max_count = max(count.values())
            candidates = sorted([p for p, c in count.items() if c == max_count])
            most_freq = candidates[0]
        #    c. Merge all non-overlapping occurrences left to right
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens)-1 and tokens[i] == most_freq[0] and tokens[i+1] == most_freq[1]:
                    new_tokens.append(most_freq[0] + most_freq[1])
                    i+=2
                else:
                    new_tokens.append(tokens[i])
                    i+=1
            tokens = new_tokens
        #    d. Record the merge as [token_a, token_b]
            merges.append([most_freq[0], most_freq[1]])
        # 3. Return the list of merges performed
        # print(merges)
        return merges