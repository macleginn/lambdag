import random
from tqdm.auto import tqdm
from collections import defaultdict
import pandas as pd
from lambdag import LambdaG

print("Initializing LambdaG...")
lambda_g = LambdaG()
print("Loading data...")
records = []
with open(
    "/mnt/d/datasets/Gungor_2018_VictorianAuthorAttribution_data-train.csv",
    "r",
    encoding="utf-8",
    errors="replace",
) as f:
    for line in f:
        line = line.strip()
        if not line or "�" in line:
            continue
        records.append(line.split(","))
df = pd.DataFrame(records[1:], columns=records[0])
top_authors = df.author.value_counts()[:10].index.tolist()
author_to_fragments = {}
for author in top_authors:
    author_to_fragments[author] = df.loc[df.author == author].text.tolist()

# Predefine train and test splits for all authors
author_train_test_splits = {}
for author in author_to_fragments:
    fragments = author_to_fragments[author][:]
    random.shuffle(fragments)
    test_set_size = len(fragments) // 2
    author_train_test_splits[author] = {
        "train": fragments[test_set_size:],
        "test": fragments[:test_set_size],
    }

print("Computing LambdaG scores...")
results = defaultdict(dict)

# Train and test for each author
for author in tqdm(author_train_test_splits):
    # Get train and test fragments for the current author
    train_fragments = author_train_test_splits[author]["train"]
    test_fragments = author_train_test_splits[author]["test"]

    # Train the model on the training set of the current author
    lambda_g.train_known_author_model(train_fragments)

    # Compute LambdaG scores for the test fragments of the current author
    results[author][author] = lambda_g.compute_lambda_g(test_fragments)

    # Compute LambdaG scores for the test fragments of other authors
    for other_author in tqdm(author_train_test_splits, leave=False):
        if other_author == author:
            continue
        other_test_fragments = author_train_test_splits[other_author]["test"]
        results[author][other_author] = lambda_g.compute_lambda_g(other_test_fragments)

# Write results to file
print("Writing results to file...")
with open("lambda_g_results.txt", "w") as f:
    for author, other_authors in results.items():
        for other_author, score in other_authors.items():
            f.write(f"{author} vs {other_author}: {score}\n")
print("Done.")