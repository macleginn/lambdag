"""
Test the C++ trie implementation with Python bindings by populating the trie with a million random
ngrams and then asking for their stats in a single batched call.
"""

import trie
import string
import random
import numpy as np
from tqdm.auto import tqdm


t = trie.Trie()
print('Generating random ngrams...')
random_corpus = np.random.choice(
    list(string.ascii_lowercase),
    size=(10**7, 5),
    replace=True,
).tolist()
for ngram in tqdm(random_corpus, 'Adding ngrams'):
    t.add(ngram)
results = t.getCountsBatch(random_corpus)
print(len(results), "results")
count_matrix = np.array([r[0] for r in results])
child_count_matrix = np.array([r[1] for r in results])
print(f"Mean counts: {np.mean(count_matrix, axis=0).tolist()}")
print(f"Mean child counts: {np.mean(child_count_matrix, axis=0).tolist()}")
