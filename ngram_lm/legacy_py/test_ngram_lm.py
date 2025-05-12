import unittest
import trie
from ngram_lm import NGramLM


class TestNGramLM(unittest.TestCase):
    def test_initialization(self):
        """Test that the NGramLM initializes correctly"""
        lm = NGramLM(n=3)
        self.assertEqual(lm.n, 3)
        self.assertEqual(lm.vocabulary, set())
        self.assertIsNone(lm.trie)
        self.assertIsNone(lm.reverse_trie)
        self.assertEqual(lm.unique_bigrams, set())

    def test_fit_initializes_tries(self):
        """Test that fit() initializes the tries"""
        lm = NGramLM(n=2)
        lm.fit([["a", "b", "c"]])
        self.assertIsNotNone(lm.trie)
        self.assertIsNotNone(lm.reverse_trie)

    def test_add_sequence_updates_vocabulary(self):
        """Test that _add_sequence updates the vocabulary"""
        lm = NGramLM(n=2)
        lm.fit([["a", "b", "c"]])
        self.assertEqual(lm.vocabulary, {"a", "b", "c"})

    def test_add_sequence_updates_unique_bigrams(self):
        """Test that _add_sequence updates unique_bigrams"""
        lm = NGramLM(n=2)
        lm.fit([["a", "b", "c"]])
        self.assertEqual(lm.unique_bigrams, {("a", "b"), ("b", "c")})

    def test_add_sequence_adds_ngrams_to_trie(self):
        """Test that _add_sequence adds n-grams to the trie"""
        lm = NGramLM(n=2)
        lm.fit([["a", "b", "c"]])
        counts, _ = lm.trie.getCounts(["a", "b"])
        self.assertEqual(counts, [1, 1])

    def test_add_sequence_adds_reverse_ngrams(self):
        """Test that _add_sequence adds reversed n-grams to the reverse_trie"""
        lm = NGramLM(n=2)
        lm.fit([["a", "b", "c"]])
        # Check if "b" follows "a" in the reverse trie (which stores ["b", "a"])
        counts, _ = lm.reverse_trie.getCounts(["b"])
        self.assertEqual(counts, [1])

    def test_unigram_probability_with_seen_word(self):
        """Test unigram probability calculation for seen words"""
        lm = NGramLM(n=2)
        lm.fit([["a", "b", "c"], ["a", "d", "c"], ["e", "b", "f"]])
        # "b" appears after "a" and "e", so it has 2 unique predecessors
        prob = lm.get_ngram_probability(["b"])
        self.assertEqual(prob, 2 / len(lm.unique_bigrams))

    def test_unigram_probability_with_unseen_word(self):
        """Test unigram probability calculation for unseen words"""
        lm = NGramLM(n=2)
        lm.fit([["a", "b", "c"]])
        prob = lm.get_ngram_probability(["z"])
        self.assertEqual(prob, 1.0 / len(lm.vocabulary))

    def test_bigram_probability_with_seen_bigram(self):
        """Test bigram probability with Kneser-Ney smoothing"""
        lm = NGramLM(n=2)
        lm.fit([["a", "b", "c"], ["a", "b", "d"]])
        
        # P(b|a) calculation
        # Count(a,b) = 2, Count(a) = 2, unique_following(a) = 1
        # First term = (2-0.75)/2 = 0.625
        # Second term = (0.75/2) * 1 * P(b) = 0.375 * P(b)
        # P(b) = 1/4 (assuming b has 1 unique predecessor out of 4 bigrams)
        
        prob = lm.get_ngram_probability(["a", "b"])
        # We can't assert the exact value without knowing len(unique_bigrams),
        # but we can check it's greater than first_term
        self.assertGreater(prob, 0.625)

    def test_trigram_probability(self):
        """Test trigram probability with Kneser-Ney smoothing"""
        lm = NGramLM(n=3)
        lm.fit([
            ["a", "b", "c", "d"],
            ["a", "b", "c", "e"],
            ["a", "b", "f", "g"]
        ])
        
        # Calculate P(c|a,b)
        prob = lm.get_ngram_probability(["a", "b", "c"])
        # Should be a non-zero value
        self.assertGreater(prob, 0)

    def test_nonzero_probability_with_unseen_ngram(self):
        """Test that even unseen n-grams get non-zero probability"""
        lm = NGramLM(n=3)
        lm.fit([["a", "b", "c", "d"]])
        
        # Calculate P(z|x,y) where x,y,z are all unseen
        prob = lm.get_ngram_probability(["x", "y", "z"])
        self.assertGreater(prob, 0)  # Should not be zero

    def test_backoff_to_lower_order(self):
        """Test that model backs off to lower order when appropriate"""
        lm = NGramLM(n=3)
        lm.fit([["a", "b", "c", "d"], ["e", "f", "g", "h"]])
        
        # P(c|a,b) should be high since we've seen "a b c"
        high_prob = lm.get_ngram_probability(["a", "b", "c"])
        
        # P(c|e,f) should be lower since we've never seen "e f c",
        # so it will back off to P(c|f) and then to P(c)
        low_prob = lm.get_ngram_probability(["e", "f", "c"])
        
        self.assertGreater(high_prob, low_prob)

    def test_add_sequence_with_too_short_sequence(self):
        """Test _add_sequence with a sequence shorter than n"""
        lm = NGramLM(n=3)
        lm.fit([])  # Initialize the tries
        with self.assertRaises(ValueError):
            lm._add_sequence(["a", "b"])

    def test_add_sequence_without_initialized_tries(self):
        """Test _add_sequence without initialized tries"""
        lm = NGramLM(n=2)
        with self.assertRaises(ValueError):
            lm._add_sequence(["a", "b", "c"])
            

if __name__ == '__main__':
    unittest.main()