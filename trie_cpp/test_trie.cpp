#include "trie.hpp"
#include <gtest/gtest.h>

class TrieTest : public ::testing::Test
{
protected:
    Trie trie;

    void SetUp() override
    {
        // Optional setup before each test
    }

    void TearDown() override
    {
        // Optional cleanup after each test
    }
};

TEST_F(TrieTest, AddSingleSequence)
{
    trie.add({"a", "b", "c"});
    std::vector<int> counts = trie.getCounts({"a", "b", "c"});
    EXPECT_EQ(counts, std::vector<int>({1, 1, 1}));
}

TEST_F(TrieTest, AddMultipleSequences)
{
    trie.add({"a", "b", "c"});
    trie.add({"a", "b", "d"});
    std::vector<int> counts1 = trie.getCounts({"a", "b", "c"});
    std::vector<int> counts2 = trie.getCounts({"a", "b", "d"});
    EXPECT_EQ(counts1, std::vector<int>({2, 2, 1}));
    EXPECT_EQ(counts2, std::vector<int>({2, 2, 1}));
}

TEST_F(TrieTest, GetCountsForPartialSequence)
{
    trie.add({"a", "b", "c"});
    std::vector<int> counts = trie.getCounts({"a", "b"});
    EXPECT_EQ(counts, std::vector<int>({1, 1}));
}

TEST_F(TrieTest, GetCountsForNonExistentSequence)
{
    trie.add({"a", "b", "c"});
    std::vector<int> counts = trie.getCounts({"x", "y", "z"});
    EXPECT_EQ(counts, std::vector<int>({0, 0, 0}));
}

TEST_F(TrieTest, AddDuplicateSequence)
{
    trie.add({"a", "b", "c"});
    trie.add({"a", "b", "c"});
    std::vector<int> counts = trie.getCounts({"a", "b", "c"});
    EXPECT_EQ(counts, std::vector<int>({2, 2, 2}));
}

TEST_F(TrieTest, GetCountsBatch)
{
    trie.add({"a", "b", "c"});
    trie.add({"a", "b", "d"});
    trie.add({"x", "y"});

    std::vector<std::vector<std::string>> queries = {
        {"a", "b", "c"},
        {"a", "b", "d"},
        {"x", "y"},
        {"z"}};

    std::vector<std::vector<int>> results = trie.getCountsBatch(queries);

    EXPECT_EQ(results[0], std::vector<int>({2, 2, 1}));
    EXPECT_EQ(results[1], std::vector<int>({2, 2, 1}));
    EXPECT_EQ(results[2], std::vector<int>({1, 1}));
    EXPECT_EQ(results[3], std::vector<int>({0}));
}