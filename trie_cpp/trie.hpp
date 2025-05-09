#ifndef TRIE_HPP
#define TRIE_HPP

#include <unordered_map>
#include <vector>
#include <string>
#include <mutex>
#include <thread>
#include <future>

class Trie
{
private:
    struct Node
    {
        std::string key;
        std::unordered_map<std::string, Node *> children;
        int count = 0;

        Node(const std::string &key = "") : key(key) {}
    };

    Node *root;
    std::mutex mutex;

    void addSequence(Node *node, const std::vector<std::string> &sequence, size_t index)
    {
        if (index == sequence.size())
        {
            return;
        }
        const std::string &word = sequence[index];
        if (node->children.find(word) == node->children.end())
        {
            node->children[word] = new Node(word);
        }
        node->children[word]->count++;
        addSequence(node->children[word], sequence, index + 1);
    }

    void getCounts(Node *node, const std::vector<std::string> &query, size_t index, std::vector<int> &counts) const
    {
        if (index == query.size())
        {
            return;
        }
        const std::string &word = query[index];
        if (node->children.find(word) == node->children.end())
        {
            // Pad the counts with zeros for the remaining words in the query
            counts.insert(counts.end(), query.size() - index, 0);
            return;
        }
        node = node->children[word];
        counts.push_back(node->count);
        getCounts(node, query, index + 1, counts);
    }

public:
    Trie() : root(new Node()) {}

    ~Trie()
    {
        std::function<void(Node *)> deleteNodes = [&](Node *node)
        {
            for (auto &child : node->children)
            {
                deleteNodes(child.second);
            }
            delete node;
        };
        deleteNodes(root);
    }

    void add(const std::vector<std::string> &sequence)
    {
        std::lock_guard<std::mutex> lock(mutex);
        addSequence(root, sequence, 0);
    }

    std::vector<int> getCounts(const std::vector<std::string> &query) const
    {
        std::vector<int> counts;
        getCounts(root, query, 0, counts);
        return counts;
    }

    std::vector<std::vector<int>> getCountsBatch(const std::vector<std::vector<std::string>> &queries) const
    {
        std::vector<std::future<std::vector<int>>> futures;
        for (const auto &query : queries)
        {
            futures.push_back(std::async(std::launch::async, [this, query]()
                                         { return this->getCounts(query); }));
        }

        std::vector<std::vector<int>> results;
        for (auto &future : futures)
        {
            results.push_back(future.get());
        }
        return results;
    }
};

#endif // TRIE_HPP