#include "gtest/gtest.h"
#include "ngram_lm.hpp" // This should include trie.hpp
#include <vector>
#include <string>
#include <stdexcept> // For testing exceptions
#include <numeric>   // For std::iota (C++11)
#include <algorithm> // For std::shuffle (C++11), std::min
#include <random>    // For std::mt19937 (C++11)
#include <iostream>

// Test fixture for NGramLM tests if common setup/teardown is needed.
// For now, we'll instantiate NGramLM directly in tests.
// class NGramLMTest : public ::testing::Test {
// protected:
//     NGramLM lm_{2}; // Example default LM
// };

TEST(NGramLMTest, ConstructorValid) {
    ASSERT_NO_THROW(NGramLM lm(3));
}

TEST(NGramLMTest, ConstructorInvalidN) {
    ASSERT_THROW(NGramLM lm_invalid(0), std::invalid_argument);
    ASSERT_THROW(NGramLM lm_invalid(-1), std::invalid_argument);
}

TEST(NGramLMTest, FitAndBasicProperties) {
    NGramLM lm(2);
    std::vector<std::vector<std::string>> sequences1 = {
        {"a", "b", "c"},
        {"a", "b", "d"}
    };
    ASSERT_NO_THROW(lm.fit(sequences1));

    // Test calling fit again (should reset)
    std::vector<std::vector<std::string>> sequences2 = {
        {"x", "y"}
    };
    ASSERT_NO_THROW(lm.fit(sequences2));
    // After fitting with {{"x", "y"}}, vocab is {"x", "y"}. "a" is unseen.
    // P(a) should be 1.0 / |{x,y}| = 1.0 / 2.0
    EXPECT_DOUBLE_EQ(lm.getNgramProbability({"a"}), 1.0 / 2.0);
}

TEST(NGramLMTest, UnigramProbabilities) {
    NGramLM lm(2); // n=2 for bigrams to populate unique_bigrams_
    std::vector<std::vector<std::string>> sequences = {
        {"a", "b", "c"}, // bigrams: (a,b), (b,c)
        {"x", "b", "y"}  // bigrams: (x,b), (b,y)
    };
    lm.fit(sequences);
    // Vocab: {a,b,c,x,y} size 5
    // Unique bigrams: {(a,b), (b,c), (x,b), (b,y)} size 4

    // P("b"): "b" is preceded by "a" and "x". num_predecessors for "b" is 2.
    EXPECT_DOUBLE_EQ(lm.getNgramProbability({"b"}), 2.0 / 4.0);

    // P("a"): "a" is not a successor in any bigram. num_predecessors for "a" is 0.
    EXPECT_DOUBLE_EQ(lm.getNgramProbability({"a"}), 1.0 / 5.0); // Fallback to 1.0 / vocab_size
    
    // P("c"): "c" is preceded by "b". num_predecessors for "c" is 1.
    EXPECT_DOUBLE_EQ(lm.getNgramProbability({"c"}), 1.0 / 4.0);

    // P("z") (unseen word): Fallback to 1.0 / vocab_size
    EXPECT_DOUBLE_EQ(lm.getNgramProbability({"z"}), 1.0 / 5.0);

    // Edge case: empty vocab (after fitting with empty sequences)
    NGramLM lm_empty_vocab(2);
    lm_empty_vocab.fit({});
    EXPECT_DOUBLE_EQ(lm_empty_vocab.getNgramProbability({"a"}), 0.0); // Vocab empty, returns 0.0

    NGramLM lm_no_bigrams(1); // n=1, so unique_bigrams_ will be empty
    lm_no_bigrams.fit({{"a"}, {"b"}});
    // Vocab: {a,b} size 2. unique_bigrams_ is empty.
    // num_predecessors for "a" (from reverse_trie with n=1) will be 0.
    EXPECT_DOUBLE_EQ(lm_no_bigrams.getNgramProbability({"a"}), 1.0 / 2.0); // Fallback
}

TEST(NGramLMTest, HigherOrderProbabilities) {
    NGramLM lm(2); // n=2
    double discount = 0.75;
    std::vector<std::vector<std::string>> sequences = {
        {"a", "b", "c"}, {"a", "b", "d"}, {"x", "b", "c"}
    };
    lm.fit(sequences);
    // Expected P(a,b) = 0.8125
    EXPECT_NEAR(lm.getNgramProbability({"a", "b"}, "kneser-ney", discount), 0.8125, 1e-9);

    // Expected P(a,z) = 0.075 (backoff)
    EXPECT_NEAR(lm.getNgramProbability({"a", "z"}, "kneser-ney", discount), 0.075, 1e-9);
    
    NGramLM lm3(3);
    std::vector<std::vector<std::string>> sequences3 = {
        {"s1", "s2", "s3", "s4"},
        {"s1", "s2", "s3", "s5"}
    };
    lm3.fit(sequences3);
    // Expected P(s1,s2,s3) = 0.89453125
    EXPECT_NEAR(lm3.getNgramProbability({"s1", "s2", "s3"}, "kneser-ney", discount), 0.89453125, 1e-9);
}

TEST(NGramLMTest, GetNgramProbabilityErrors) {
    NGramLM lm(2);
    ASSERT_THROW(lm.getNgramProbability({"a", "b"}), std::runtime_error) << "Should throw if called before fit";

    lm.fit({{"x", "y"}}); 

    ASSERT_THROW(lm.getNgramProbability({}), std::invalid_argument) << "Should throw for empty ngram";
    ASSERT_THROW(lm.getNgramProbability({"x"}, "unsupported_method"), std::invalid_argument) << "Should throw for unsupported method";
}

TEST(NGramLMTest, GetNgramProbabilitiesConcurrent) {
    NGramLM lm(2);
    double discount = 0.75;
    std::vector<std::vector<std::string>> sequences = {
        {"a", "b", "c"}, {"a", "b", "d"}, {"x", "b", "c"},
        {"fee", "fi", "fo"}, {"fee", "fi", "fum"}
    };
    lm.fit(sequences);

    std::vector<std::vector<std::string>> batch_ngrams = {
        {"a", "b"}, {"x", "b"}, {"a", "z"}, {"b"}, {"fee", "fi"}, {"non", "existent"}
    };

    std::vector<double> expected_probs;
    expected_probs.reserve(batch_ngrams.size());
    for(const auto& ngram_q : batch_ngrams) {
        expected_probs.push_back(lm.getNgramProbability(ngram_q, "kneser-ney", discount));
    }
    
    std::vector<double> concurrent_probs;
    ASSERT_NO_THROW(concurrent_probs = lm.getNgramProbabilities(batch_ngrams, "kneser-ney", discount));

    ASSERT_EQ(concurrent_probs.size(), batch_ngrams.size()) << "Concurrent results size mismatch";
    for (size_t i = 0; i < concurrent_probs.size(); ++i) {
        EXPECT_NEAR(concurrent_probs[i], expected_probs[i], 1e-9) 
            << "Concurrent probability mismatch for ngram at index " << i;
    }

    std::vector<std::vector<std::string>> empty_batch = {};
    std::vector<double> empty_results;
    ASSERT_NO_THROW(empty_results = lm.getNgramProbabilities(empty_batch));
    ASSERT_TRUE(empty_results.empty()) << "Concurrent processing of empty batch should return empty results";

    NGramLM lm_not_fitted(2);
    ASSERT_THROW(lm_not_fitted.getNgramProbabilities(batch_ngrams), std::runtime_error) 
        << "getNgramProbabilities should throw if called before fit";
}

// Main function for Google Test
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    std::cout << "Starting NGramLM Google Tests..." << std::endl;
    std::cout << "NOTE: These tests assume a correctly functioning Trie from trie.hpp" << std::endl;
    std::cout << "      and that Trie::getCounts returns vectors of the query ngram's length," << std::endl;
    std::cout << "      padding with 0s for non-existent paths/counts." << std::endl;
    std::cout << "------------------------------------" << std::endl;
    return RUN_ALL_TESTS();
}