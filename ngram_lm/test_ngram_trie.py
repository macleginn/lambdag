import unittest
from ngram_trie import TrieWithCounts


class TestTrieWithCounts(unittest.TestCase):
    def setUp(self):
        """Set up a new TrieWithCounts instance for each test."""
        self.trie = TrieWithCounts()

    def test_insert_single_sequence(self):
        """Test inserting a single sequence into the trie."""
        sequence = ["a", "b", "c"]
        self.trie.insert(sequence)
        stats = self.trie.get_sequence_stats(sequence)
        self.assertEqual(
            stats, [1, 1, 1], "Counts for single sequence should all be 1."
        )

    def test_insert_multiple_sequences(self):
        """Test inserting multiple sequences into the trie."""
        sequence1 = ["a", "b", "c"]
        sequence2 = ["a", "b", "d"]
        self.trie.insert(sequence1)
        self.trie.insert(sequence2)
        stats1 = self.trie.get_sequence_stats(sequence1)
        stats2 = self.trie.get_sequence_stats(sequence2)
        self.assertEqual(
            stats1,
            [2, 2, 1],
            "Counts for shared prefixes should reflect multiple insertions.",
        )
        self.assertEqual(
            stats2,
            [2, 2, 1],
            "Counts for shared prefixes should reflect multiple insertions.",
        )

    def test_get_sequence_stats_partial_match(self):
        """Test retrieving stats for a sequence that partially matches the trie."""
        sequence = ["a", "b", "c"]
        self.trie.insert(sequence)
        partial_sequence = ["a", "b"]
        stats = self.trie.get_sequence_stats(partial_sequence)
        self.assertEqual(
            stats, [1, 1], "Counts for partial sequence should match inserted data."
        )

    def test_get_sequence_stats_no_match(self):
        """Test retrieving stats for a sequence that does not exist in the trie."""
        sequence = ["x", "y", "z"]
        stats = self.trie.get_sequence_stats(sequence)
        self.assertEqual(
            stats, [], "Counts for non-existent sequence should be an empty list."
        )

    def test_insert_duplicate_sequence(self):
        """Test inserting the same sequence multiple times."""
        sequence = ["a", "b", "c"]
        self.trie.insert(sequence)
        self.trie.insert(sequence)
        stats = self.trie.get_sequence_stats(sequence)
        self.assertEqual(
            stats,
            [2, 2, 2],
            "Counts should reflect multiple insertions of the same sequence.",
        )


if __name__ == "__main__":
    unittest.main()
