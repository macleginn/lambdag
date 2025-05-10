import trie


class NGramLM:
    def __init__(self, n):
        self.n = n
        self.vocabulary = set()
        self.trie = None
        # For predecessor counts
        self.reverse_trie = None
        self.unique_bigrams = set()

    def _add_sequence(self, sequence):
        """
        Adds a sequence of tokens to the trie.
        """
        if self.trie is None or self.reverse_trie is None:
            raise ValueError("Tries must be initialized before adding sequences.")

        if len(sequence) < self.n:
            raise ValueError(f"Sequence length must be at least {self.n}.")

        self.vocabulary.update(sequence)
        for i in range(len(sequence) - 1):
            self.unique_bigrams.add((sequence[i], sequence[i + 1]))

        for i in range(len(sequence) - self.n + 1):
            ngram = sequence[i : i + self.n]
            self.trie.add(ngram)

        sequence_rev = sequence[::-1]
        for i in range(len(sequence_rev) - self.n + 1):
            ngram = sequence_rev[i : i + self.n]
            self.reverse_trie.add(ngram)

    def fit(self, sequences):
        """
        Fits the n-gram model to the given sequences.
        """
        self.vocabulary = set()
        self.trie = trie.Trie()
        self.reverse_trie = trie.Trie()
        self.unique_bigrams = set()
        for sequence in sequences:
            self._add_sequence(sequence)

    def get_ngram_probability(self, ngram, method="kneser-ney", discount=0.75):
        """
        Get the probability of an n-gram using Kneser-Ney smoothing.
        """
        if method != "kneser-ney":
            raise NotImplementedError(f"Unknown method: {method}")

        if len(ngram) == 1:
            # Unigram case: P(w) = count(unique bigrams with w as second word) / count(all unique bigrams)
            # Get count of unique words that this word follows
            # The trie API needs a single-element list here
            _, n_predecessors = self.reverse_trie.getCounts(ngram)
            if n_predecessors[0] == 0:
                # The word not found in the reverse trie
                # Should not happen if the data set is preprocessed correctly
                return 1.0 / len(
                    self.vocabulary
                )  # Uniform distribution over vocabulary
            return n_predecessors[0] / len(self.unique_bigrams)

        n_gram_counts, n_successors = self.trie.getCounts(ngram)

        # Full n-gram count
        ngram_count = n_gram_counts[-1]
        # History count
        history_count = max(n_gram_counts[-2], 1)
        # Unique words after history
        unique_following_words = max(n_successors[-2], 1)

        discounted_count = max(ngram_count - discount, 0)
        first_term = discounted_count / history_count
        second_term = (
            (discount / history_count)
            * unique_following_words
            * (self.get_ngram_probability(ngram[1:], method=method, discount=discount))
        )
        return first_term + second_term
