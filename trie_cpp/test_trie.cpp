// Compilation command: g++ -std=c++17 -pthread test_trie.cpp /usr/lib/libgtest.a /usr/lib/libgtest_main.a -o test_trie

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
    auto result = trie.getCounts({"a", "b", "c"});
    std::vector<int> counts = result.first;
    std::vector<int> unique_children = result.second;

    EXPECT_EQ(counts, std::vector<int>({1, 1, 1}));
    EXPECT_EQ(unique_children, std::vector<int>({1, 1, 0}));
}

TEST_F(TrieTest, AddMultipleSequences)
{
    trie.add({"a", "b", "c"});
    trie.add({"a", "b", "d"});
    auto result = trie.getCounts({"a", "b"});
    std::vector<int> counts = result.first;
    std::vector<int> unique_children = result.second;

    EXPECT_EQ(counts, std::vector<int>({2, 2}));
    EXPECT_EQ(unique_children, std::vector<int>({1, 2}));
}

TEST_F(TrieTest, GetCountsForPartialSequence)
{
    trie.add({"a", "b", "c"});
    auto result = trie.getCounts({"a", "b"});
    std::vector<int> counts = result.first;
    std::vector<int> unique_children = result.second;

    EXPECT_EQ(counts, std::vector<int>({1, 1}));
    EXPECT_EQ(unique_children, std::vector<int>({1, 1}));
}

TEST_F(TrieTest, GetCountsForNonExistentSequence)
{
    trie.add({"a", "b", "c"});
    auto result = trie.getCounts({"x", "y", "z"});
    std::vector<int> counts = result.first;
    std::vector<int> unique_children = result.second;

    EXPECT_EQ(counts, std::vector<int>({0, 0, 0}));
    EXPECT_EQ(unique_children, std::vector<int>({0, 0, 0}));
}

TEST_F(TrieTest, AddDuplicateSequence)
{
    trie.add({"a", "b", "c"});
    trie.add({"a", "b", "c"});
    auto result = trie.getCounts({"a", "b", "c"});
    std::vector<int> counts = result.first;
    std::vector<int> unique_children = result.second;

    EXPECT_EQ(counts, std::vector<int>({2, 2, 2}));
    EXPECT_EQ(unique_children, std::vector<int>({1, 1, 0}));
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

    auto results = trie.getCountsBatch(queries);

    EXPECT_EQ(results[0].first, std::vector<int>({2, 2, 1}));
    EXPECT_EQ(results[0].second, std::vector<int>({1, 2, 0}));

    EXPECT_EQ(results[1].first, std::vector<int>({2, 2, 1}));
    EXPECT_EQ(results[1].second, std::vector<int>({1, 2, 0}));

    EXPECT_EQ(results[2].first, std::vector<int>({1, 1}));
    EXPECT_EQ(results[2].second, std::vector<int>({1, 0}));

    EXPECT_EQ(results[3].first, std::vector<int>({0}));
    EXPECT_EQ(results[3].second, std::vector<int>({0}));
}