#ifndef NGRAM_LM_HPP
#define NGRAM_LM_HPP

#include <vector>
#include <string>
#include <unordered_set>
#include <set>
#include <utility>   // For std::pair
#include <memory>    // For std::unique_ptr
#include <stdexcept> // For std::invalid_argument, std::runtime_error
#include <algorithm> // For std::max, std::reverse, std::sort

// For getNgramProbabilities
#include <thread>     // For std::thread, std::thread::hardware_concurrency
#include <queue>      // For std::queue
#include <functional> // For std::function, std::bind
#include <future>     // For std::future, std::async
#include <mutex>      // For std::mutex, std::lock_guard

#include "trie.hpp"

class NGramLM
{
private:
    int n_;
    std::unordered_set<std::string> vocabulary_;
    std::unique_ptr<Trie> trie_;
    std::unique_ptr<Trie> reverse_trie_;
    std::set<std::pair<std::string, std::string>> unique_bigrams_;

    void addSequence(const std::vector<std::string> &sequence)
    {
        if (!trie_ || !reverse_trie_)
        {
            throw std::runtime_error("Tries must be initialized before adding sequences. Call fit() first.");
        }

        for (const auto &token : sequence)
        {
            vocabulary_.insert(token);
        }

        if (sequence.size() >= 2)
        {
            for (size_t i = 0; i < sequence.size() - 1; ++i)
            {
                unique_bigrams_.insert({sequence[i], sequence[i + 1]});
            }
        }

        if (sequence.size() >= static_cast<size_t>(n_))
        {
            for (size_t i = 0; i <= sequence.size() - n_; ++i)
            {
                std::vector<std::string> current_ngram;
                current_ngram.reserve(n_);
                for (int j = 0; j < n_; ++j)
                {
                    current_ngram.push_back(sequence[i + j]);
                }
                trie_->add(current_ngram);
            }
        }

        if (sequence.size() >= static_cast<size_t>(n_))
        {
            std::vector<std::string> sequence_rev = sequence;
            std::reverse(sequence_rev.begin(), sequence_rev.end());
            for (size_t i = 0; i <= sequence_rev.size() - n_; ++i)
            {
                std::vector<std::string> current_ngram;
                current_ngram.reserve(n_);
                for (int j = 0; j < n_; ++j)
                {
                    current_ngram.push_back(sequence_rev[i + j]);
                }
                reverse_trie_->add(current_ngram);
            }
        }
    }

public:
    explicit NGramLM(int n) : n_(n), trie_(nullptr), reverse_trie_(nullptr)
    {
        if (n <= 0)
        {
            throw std::invalid_argument("n must be positive.");
        }
    }

    void fit(const std::vector<std::vector<std::string>> &sequences)
    {
        vocabulary_.clear();
        unique_bigrams_.clear();

        trie_ = std::make_unique<Trie>();
        reverse_trie_ = std::make_unique<Trie>();

        for (const auto &sequence : sequences)
        {
            addSequence(sequence);
        }
    }

    double getNgramProbability(
        const std::vector<std::string> &ngram,
        const std::string &method = "kneser-ney",
        double discount = 0.75) const
    {
        if (method != "kneser-ney")
        {
            throw std::invalid_argument("Unknown method: " + method + ". Only 'kneser-ney' is supported.");
        }

        if (ngram.empty())
        {
            throw std::invalid_argument("Input ngram cannot be empty.");
        }

        if (!trie_ || !reverse_trie_)
        {
            throw std::runtime_error("Model not fitted. Call fit() before getNgramProbability().");
        }

        if (ngram.size() == 1)
        {
            std::pair<std::vector<int>, std::vector<int>> reverse_counts_pair = reverse_trie_->getCounts(ngram);
            int num_predecessors = 0;
            // getCounts for a unigram query returns vectors of size 1 if the unigram exists as a prefix
            // Check if second vector is not empty and its first element is not zero
            if (!reverse_counts_pair.second.empty() && reverse_counts_pair.second[0] != 0)
            {
                num_predecessors = reverse_counts_pair.second[0];
            }

            if (vocabulary_.empty())
            {
                return 0.0;
            }

            if (num_predecessors == 0)
            {
                return 1.0 / static_cast<double>(std::max(vocabulary_.size(), static_cast<size_t>(1)));
            }

            if (unique_bigrams_.empty())
            {
                return 1.0 / static_cast<double>(std::max(vocabulary_.size(), static_cast<size_t>(1)));
            }
            return static_cast<double>(num_predecessors) / static_cast<double>(std::max(unique_bigrams_.size(), static_cast<size_t>(1)));
        }

        std::pair<std::vector<int>, std::vector<int>> counts_pair = trie_->getCounts(ngram);
        const std::vector<int> &ngram_all_counts = counts_pair.first;
        const std::vector<int> &ngram_all_successors = counts_pair.second;

        // Assuming trie_->getCounts ensures returned vectors are of size ngram.size(), padding with 0s if necessary.

        int current_ngram_count = (ngram_all_counts.empty() || ngram_all_counts.size() < ngram.size()) ? 0 : ngram_all_counts.back();
        int history_raw_count = (ngram_all_counts.size() < ngram.size() || ngram_all_counts.size() < 2) ? 0 : ngram_all_counts[ngram_all_counts.size() - 2];
        int history_count = std::max(history_raw_count, 1);

        int unique_following_raw_words = (ngram_all_successors.size() < ngram.size() || ngram_all_successors.size() < 2) ? 0 : ngram_all_successors[ngram_all_successors.size() - 2];
        int unique_following_words = std::max(unique_following_raw_words, 1);

        double discounted_count_val = std::max(static_cast<double>(current_ngram_count) - discount, 0.0);

        double first_term = discounted_count_val / static_cast<double>(history_count);

        std::vector<std::string> next_ngram(ngram.begin() + 1, ngram.end());
        double continuation_prob = getNgramProbability(next_ngram, method, discount);

        double second_term = (discount / static_cast<double>(history_count)) *
                             static_cast<double>(unique_following_words) *
                             continuation_prob;

        return first_term + second_term;
    }

    std::vector<double> getNgramProbabilities(
        const std::vector<std::vector<std::string>> &ngrams_batch,
        const std::string &method = "kneser-ney",
        double discount = 0.75) const
    {
        if (!trie_ || !reverse_trie_)
        {
            throw std::runtime_error("Model not fitted. Call fit() before getNgramProbabilities().");
        }

        size_t num_total_ngrams = ngrams_batch.size();
        if (num_total_ngrams == 0)
        {
            return {};
        }

        std::mutex queue_mutex;
        std::queue<std::pair<size_t, std::vector<std::string>>> task_queue;

        for (size_t i = 0; i < num_total_ngrams; ++i)
        {
            task_queue.push({i, ngrams_batch[i]});
        }

        size_t num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0)
            num_threads = 1;                                   // Fallback
        num_threads = std::min(num_threads, num_total_ngrams); // Don't create more threads than tasks

        std::vector<std::future<std::vector<std::pair<size_t, double>>>> futures;
        futures.reserve(num_threads);

        // Worker lambda
        auto worker_lambda = [&]() -> std::vector<std::pair<size_t, double>>
        {
            std::vector<std::pair<size_t, double>> local_results;
            while (true)
            {
                std::pair<size_t, std::vector<std::string>> current_task;
                bool task_retrieved = false;
                {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    if (!task_queue.empty())
                    {
                        current_task = task_queue.front();
                        task_queue.pop();
                        task_retrieved = true;
                    }
                } // Mutex released here

                if (!task_retrieved)
                {
                    break; // No more tasks
                }

                double prob = this->getNgramProbability(current_task.second, method, discount);
                local_results.push_back({current_task.first, prob});
            }
            return local_results;
        };

        for (size_t i = 0; i < num_threads; ++i)
        {
            futures.push_back(std::async(std::launch::async, worker_lambda));
        }

        std::vector<std::pair<size_t, double>> indexed_results;
        indexed_results.reserve(num_total_ngrams);

        for (auto &fut : futures)
        {
            try
            {
                std::vector<std::pair<size_t, double>> local_thread_results = fut.get();
                indexed_results.insert(indexed_results.end(), local_thread_results.begin(), local_thread_results.end());
            }
            catch (const std::exception &e)
            {
                throw std::runtime_error("Exception in worker thread: " + std::string(e.what()));
            }
        }

        // Sort results by original index to ensure correct order
        std::sort(indexed_results.begin(), indexed_results.end(),
                  [](const auto &a, const auto &b)
                  {
                      return a.first < b.first;
                  });

        std::vector<double> final_results;
        final_results.reserve(num_total_ngrams);
        for (const auto &res_pair : indexed_results)
        {
            final_results.push_back(res_pair.second);
        }

        return final_results;
    }
};

#endif // NGRAM_LM_HPP