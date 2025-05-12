import unittest

# Attempt to import the compiled C++ module
try:
    import ngram_lm
except ImportError as e:
    print(f"Failed to import ngram_lm: {e}")
    print(
        "Make sure the compiled ngram_lm.so (or similar) is in your PYTHONPATH or current directory."
    )
    ngram_lm = None  # Set to None so tests can be skipped if import fails


@unittest.skipIf(ngram_lm is None, "ngram_lm C++ module not found or failed to import.")
class TestNGramLMBindings(unittest.TestCase):
    def assertApproxEqual(self, a, b, places=7, msg=None):
        self.assertAlmostEqual(a, b, places=places, msg=msg)

    def test_constructor(self):
        """Test NGramLM constructor."""
        try:
            lm = ngram_lm.NGramLM(n=2)
            self.assertIsInstance(lm, ngram_lm.NGramLM)
        except Exception as e:
            self.fail(f"NGramLM constructor failed: {e}")

        with self.assertRaises(
            TypeError, msg="Constructor should raise TypeError for missing 'n'"
        ):
            ngram_lm.NGramLM()  # type: ignore

        with self.assertRaises(
            ValueError, msg="Constructor should raise ValueError for n <= 0"
        ):
            ngram_lm.NGramLM(n=0)
        with self.assertRaises(
            ValueError, msg="Constructor should raise ValueError for n < 0"
        ):
            ngram_lm.NGramLM(n=-1)

    def test_fit_and_basic_properties(self):
        """Test fitting the model and basic reset behavior."""
        lm = ngram_lm.NGramLM(n=2)
        sequences1 = [["a", "b", "c"], ["a", "b", "d"]]
        try:
            lm.fit(sequences=sequences1)
        except Exception as e:
            self.fail(f"fit method failed: {e}")

        # Test calling fit again (should reset)
        sequences2 = [["x", "y"]]
        lm.fit(sequences=sequences2)
        # After fitting with [["x", "y"]], vocab is {"x", "y"}. "a" is unseen.
        # P(a) should be 1.0 / |{x,y}| = 1.0 / 2.0
        self.assertApproxEqual(lm.get_ngram_probability(ngram=["a"]), 1.0 / 2.0)

    def test_unigram_probabilities(self):
        """Test unigram probability calculations."""
        lm = ngram_lm.NGramLM(n=2)
        sequences = [
            ["a", "b", "c"],  # bigrams: (a,b), (b,c)
            ["x", "b", "y"],  # bigrams: (x,b), (b,y)
        ]
        lm.fit(sequences)
        # Vocab: {a,b,c,x,y} size 5
        # Unique bigrams: {(a,b), (b,c), (x,b), (b,y)} size 4

        # P("b"): "b" is preceded by "a" and "x". num_predecessors for "b" is 2.
        self.assertApproxEqual(lm.get_ngram_probability(ngram=["b"]), 2.0 / 4.0)

        # P("a"): "a" is not a successor in any bigram. num_predecessors for "a" is 0.
        self.assertApproxEqual(
            lm.get_ngram_probability(ngram=["a"]), 1.0 / 5.0
        )  # Fallback

        # P("c"): "c" is preceded by "b". num_predecessors for "c" is 1.
        self.assertApproxEqual(lm.get_ngram_probability(ngram=["c"]), 1.0 / 4.0)

        # P("z") (unseen word): Fallback to 1.0 / vocab_size
        self.assertApproxEqual(lm.get_ngram_probability(ngram=["z"]), 1.0 / 5.0)

        # Edge case: empty vocab (after fitting with empty sequences)
        lm_empty_vocab = ngram_lm.NGramLM(n=2)
        lm_empty_vocab.fit(sequences=[])
        self.assertApproxEqual(lm_empty_vocab.get_ngram_probability(ngram=["a"]), 0.0)

        # Model with n=1, unique_bigrams will be empty.
        lm_n1 = ngram_lm.NGramLM(n=1)
        lm_n1.fit(sequences=[["a"], ["b"]])  # Vocab {a,b}
        # Unigram prob for 'a' should be 1/vocab_size if no bigram context
        self.assertApproxEqual(lm_n1.get_ngram_probability(ngram=["a"]), 1.0 / 2.0)

    def test_higher_order_probabilities(self):
        """Test higher-order n-gram probability calculations."""
        lm = ngram_lm.NGramLM(n=2)
        discount = 0.75
        sequences = [["a", "b", "c"], ["a", "b", "d"], ["x", "b", "c"]]
        lm.fit(sequences)
        # Expected P(a,b) = 0.8125 (from C++ test logic)
        self.assertApproxEqual(
            lm.get_ngram_probability(ngram=["a", "b"], discount=discount), 0.8125
        )

        # Expected P(a,z) = 0.075 (backoff)
        self.assertApproxEqual(
            lm.get_ngram_probability(ngram=["a", "z"], discount=discount), 0.075
        )

        lm3 = ngram_lm.NGramLM(n=3)
        sequences3 = [["s1", "s2", "s3", "s4"], ["s1", "s2", "s3", "s5"]]
        lm3.fit(sequences3)
        # Expected P(s1,s2,s3) = 0.89453125
        self.assertApproxEqual(
            lm3.get_ngram_probability(ngram=["s1", "s2", "s3"], discount=discount),
            0.89453125,
        )

    def test_get_ngram_probability_errors(self):
        """Test error conditions for get_ngram_probability."""
        lm = ngram_lm.NGramLM(n=2)
        with self.assertRaises(RuntimeError, msg="Should throw if called before fit"):
            lm.get_ngram_probability(ngram=["a", "b"])

        lm.fit(sequences=[["x", "y"]])

        with self.assertRaises(
            ValueError, msg="Should throw for empty ngram"
        ):  # C++ throws std::invalid_argument
            lm.get_ngram_probability(ngram=[])

        with self.assertRaises(
            ValueError, msg="Should throw for unsupported method"
        ):  # C++ throws std::invalid_argument
            lm.get_ngram_probability(ngram=["x"], method="unsupported_method")

    def test_get_ngram_probabilities_concurrent(self):
        """Test concurrent batch probability calculation."""
        lm = ngram_lm.NGramLM(n=2)
        discount = 0.75
        sequences = [
            ["a", "b", "c"],
            ["a", "b", "d"],
            ["x", "b", "c"],
            ["fee", "fi", "fo"],
            ["fee", "fi", "fum"],
        ]
        lm.fit(sequences)

        batch_ngrams = [
            ["a", "b"],
            ["x", "b"],
            ["a", "z"],
            ["b"],
            ["fee", "fi"],
            ["non", "existent"],
        ]

        expected_probs = []
        for ngram_q in batch_ngrams:
            expected_probs.append(
                lm.get_ngram_probability(ngram=ngram_q, discount=discount)
            )

        concurrent_probs = lm.get_ngram_probabilities(
            ngrams_batch=batch_ngrams, discount=discount
        )

        self.assertEqual(
            len(concurrent_probs), len(batch_ngrams), "Concurrent results size mismatch"
        )
        for i in range(len(concurrent_probs)):
            self.assertApproxEqual(
                concurrent_probs[i],
                expected_probs[i],
                msg=f"Concurrent probability mismatch for ngram {batch_ngrams[i]} at index {i}",
            )

    def test_get_ngram_probabilities_empty_batch(self):
        """Test concurrent batch calculation with an empty batch."""
        lm = ngram_lm.NGramLM(n=2)
        lm.fit(sequences=[["a", "b"]])
        empty_batch = []
        empty_results = lm.get_ngram_probabilities(ngrams_batch=empty_batch)
        self.assertEqual(
            len(empty_results),
            0,
            "Concurrent processing of empty batch should return empty list",
        )

    def test_get_ngram_probabilities_errors(self):
        """Test error conditions for get_ngram_probabilities."""
        lm = ngram_lm.NGramLM(n=2)
        batch_ngrams = [["a", "b"]]
        with self.assertRaises(RuntimeError, msg="Should throw if called before fit"):
            lm.get_ngram_probabilities(ngrams_batch=batch_ngrams)

    def test_default_arguments(self):
        """Test that default arguments for method and discount work."""
        lm = ngram_lm.NGramLM(n=2)
        sequences = [["a", "b", "c"]]
        lm.fit(sequences)

        # Call with all defaults
        try:
            prob1 = lm.get_ngram_probability(ngram=["a", "b"])
            self.assertIsInstance(prob1, float)
        except Exception as e:
            self.fail(f"get_ngram_probability with defaults failed: {e}")

        # Call with explicit defaults
        prob2 = lm.get_ngram_probability(
            ngram=["a", "b"], method="kneser-ney", discount=0.75
        )
        self.assertApproxEqual(prob1, prob2)

        # For batch method
        try:
            probs_batch1 = lm.get_ngram_probabilities(ngrams_batch=[["a", "b"]])
            self.assertIsInstance(probs_batch1, list)
            self.assertIsInstance(probs_batch1[0], float)
        except Exception as e:
            self.fail(f"get_ngram_probabilities with defaults failed: {e}")

        probs_batch2 = lm.get_ngram_probabilities(
            ngrams_batch=[["a", "b"]], method="kneser-ney", discount=0.75
        )
        self.assertApproxEqual(probs_batch1[0], probs_batch2[0])


if __name__ == "__main__":
    if ngram_lm is None:
        print("Skipping tests as ngram_lm module could not be imported.")
    else:
        print(f"Running tests with ngram_lm module: {ngram_lm}")
        unittest.main()
