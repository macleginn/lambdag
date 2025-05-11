#ifndef TRIE_HPP
#define TRIE_HPP

#include <unordered_map>
#include <vector>
#include <string>
#include <mutex>
#include <thread>
#include <future>
#include <queue>
#include <algorithm>

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

    void getCounts(Node *node, const std::vector<std::string> &query, size_t index, std::vector<int> &counts, std::vector<int> &unique_children) const
    {
        if (index == query.size())
        {
            return;
        }
        const std::string &word = query[index];
        if (node->children.find(word) == node->children.end())
        {
            // Pad the counts and unique_children with zeros for the remaining words in the query
            counts.insert(counts.end(), query.size() - index, 0);
            unique_children.insert(unique_children.end(), query.size() - index, 0);
            return;
        }
        node = node->children[word];
        counts.push_back(node->count);
        unique_children.push_back(node->children.size());
        getCounts(node, query, index + 1, counts, unique_children);
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

    std::pair<std::vector<int>, std::vector<int>> getCounts(const std::vector<std::string> &query) const
    {
        std::vector<int> counts;
        std::vector<int> unique_children;
        getCounts(root, query, 0, counts, unique_children);
        return {counts, unique_children};
    }

    std::vector<std::pair<std::vector<int>, std::vector<int>>> getCountsBatch(const std::vector<std::vector<std::string>> &queries)
    {
        const size_t max_threads = std::thread::hardware_concurrency(); // Get the number of available hardware threads
        std::vector<std::future<std::vector<std::pair<size_t, std::pair<std::vector<int>, std::vector<int>>>>>> futures;
        std::queue<std::pair<size_t, std::vector<std::string>>> query_queue;

        // Push all queries into a queue with their original indices
        for (size_t i = 0; i < queries.size(); ++i)
        {
            query_queue.push({i, queries[i]});
        }

        // Worker function for processing queries
        auto worker = [&]()
        {
            std::vector<std::pair<size_t, std::pair<std::vector<int>, std::vector<int>>>> results;
            while (true)
            {
                std::pair<size_t, std::vector<std::string>> indexed_query;
                {
                    std::lock_guard<std::mutex> lock(mutex); // Protect access to the queue
                    if (query_queue.empty())
                    {
                        break;
                    }
                    indexed_query = query_queue.front();
                    query_queue.pop();
                }
                size_t index = indexed_query.first;
                const auto &query = indexed_query.second;
                results.push_back({index, this->getCounts(query)});
            }
            return results;
        };

        // Launch a limited number of threads
        for (size_t i = 0; i < max_threads; ++i)
        {
            futures.push_back(std::async(std::launch::async, worker));
        }

        // Collect results from all threads
        std::vector<std::pair<size_t, std::pair<std::vector<int>, std::vector<int>>>> indexed_results;
        for (auto &future : futures)
        {
            auto thread_results = future.get();
            indexed_results.insert(indexed_results.end(), thread_results.begin(), thread_results.end());
        }

        // Sort results by their original indices
        std::sort(indexed_results.begin(), indexed_results.end(),
                  [](const auto &a, const auto &b)
                  { return a.first < b.first; });

        // Extract the results in the correct order
        std::vector<std::pair<std::vector<int>, std::vector<int>>> results;
        for (const auto &indexed_result : indexed_results)
        {
            results.push_back(indexed_result.second);
        }

        return results;
    }
};

#endif // TRIE_HPP