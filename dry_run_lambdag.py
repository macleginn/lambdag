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
print("Computing LambdaG scores...")
results = defaultdict(dict)
for author in tqdm(author_to_fragments):
    target_author_fragments = author_to_fragments[author][:]
    random.shuffle(target_author_fragments)
    test_set_size = len(target_author_fragments) // 2
    train_fragments = target_author_fragments[test_set_size:]
    test_fragments = target_author_fragments[:test_set_size]
    lambda_g.train_known_author_model(train_fragments)
    results[author][author] = lambda_g.compute_lambda_g(test_fragments)
    for other_author in tqdm(author_to_fragments, leave=False):
        if other_author == author:
            continue
        print(f"Computing LambdaG for {author} vs {other_author}...")
        other_test_fragments = author_to_fragments[other_author]
        results[author][other_author] = lambda_g.compute_lambda_g(other_test_fragments)
print("Writing results to file...")
with open("lambda_g_results.txt", "w") as f:
    for author, other_authors in results.items():
        for other_author, score in other_authors.items():
            f.write(f"{author} vs {other_author}: {score}\n")
print("Done.")
