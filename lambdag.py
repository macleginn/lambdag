"""
A Python implementation of the LambdaG algorithm for authorship verification as described in:

@misc{nini2025lambdag,
      title={Grammar as a Behavioral Biometric: Using Cognitively Motivated Grammar Models for Authorship Verification},
      author={Andrea Nini and Oren Halvani and Lukas Graner and Valerio Gherardi and Shunichi Ishihara},
      year={2025},
      eprint={2403.08462},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2403.08462},
}
"""

import random
import os
import zipfile
import json
import multiprocessing as mp
from collections import Counter
import nltk
from nltk.lm import WittenBellInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm.vocabulary import Vocabulary
import numpy as np
from tqdm.auto import tqdm
from pos_noise import POSNoise

# This needs to be global to be used in the multiprocessing pool
pos_noise = POSNoise()


class LambdaG:
    def __init__(
        self,
        n=5,
        N=20,
        sample_size=10000,
        min_freq=5,
        reference_corpus=None,
        disable_tqdm=False,
    ):
        """
        Initialize the LambdaG model.

        :param n: The order of the n-gram model.
        :param N: The number of models to train on random subsets of the reference corpus.
        :param sample_size: The size of the training set (in sentences) for each model.
        :param min_freq: The minimum token frequency of vocabulary items.
        :param reference_corpus: The reference corpus for training the model.

        If the reference corpus is not provided, references models will be trained on the
        concatenation of NLTK's Brown, Gutenberg, Reuters, Webtext, and NPS Chat corpora,
        preprocessed with POS noise.
        """

        self.disable_tqdm = disable_tqdm
        self.n = n
        self.N = N
        self.sample_size = sample_size
        self.min_freq = min_freq
        self.reference_corpus = (
            reference_corpus if reference_corpus else self._load_reference_corporus()
        )
        self.vocabulary = self._load_vocabulary()

        # Train reference models on random subsets of the reference corpus
        # using multiprocessing.
        data = [
            (
                random.sample(self.reference_corpus, self.sample_size),
                self.n,
                self.vocabulary,
            )
            for _ in range(self.N)
        ]
        num_processes = mp.cpu_count()
        if num_processes > 1:
            num_processes -= 1
        # print('Training reference models...', end=' ')
        with mp.Pool(processes=num_processes) as pool:
            self.reference_models = list(
                tqdm(
                    pool.starmap(_train_ngram_model, data),
                    total=self.N,
                    desc="Training reference models",
                    disable=self.disable_tqdm,
                )
            )
        # print('Done.')
        self.known_author_model = None

    def train_known_author_model(self, sentences):
        """
        Train the known author model on the provided sentences.
        """
        sentences_with_pos_noise = self._apply_pos_noise(
            sentences, "Applying POS noise to the known-author corpus"
        )
        self.known_author_model = _train_ngram_model(
            sentences_with_pos_noise, n=self.n, vocabulary=self.vocabulary
        )

    def compute_lambda_g(self, sentences):
        """
        Compute the LambdaG score for the provided sentences.
        """
        if not self.reference_models:
            raise ValueError("Reference models have not been trained.")
        if self.known_author_model is None:
            raise ValueError("Known author model has not been trained.")
        result = 0.0
        sentences_with_pos_noise = self._apply_pos_noise(
            sentences, "Applying POS noise to the unknown-author corpus"
        )
        
        # TODO: parallelize this loop
        for sentence in tqdm(
            sentences_with_pos_noise,
            desc="Computing LambdaG scores",
            leave=False,
            disable=self.disable_tqdm,
        ):
            padded_sentence = ["<s>"] * (self.n - 1) + sentence + ["</s>"]
            for i, word in enumerate(padded_sentence):
                if i < self.n - 1:
                    continue
                context = tuple(padded_sentence[i - self.n + 1 : i])
                word = padded_sentence[i]
                for model in self.reference_models:
                    result += 1.0 / self.N * (self.known_author_model.score(word, context) - model.score(word, context))
        return result

    def _load_reference_corporus(self):
        """
        Try loading the reference corpus from a zip file. If it fails, download the corpora
        from NLTK and preprocess them with POS noise.
        """
        path_to_zip = "data/reference_corpus_w_pos_noise.zip"
        if os.path.exists(path_to_zip):
            with zipfile.ZipFile(path_to_zip, "r") as zip_ref:
                with zip_ref.open("reference_corpus_w_pos_noise.json") as json_file:
                    return json.load(json_file)
        print(
            "Preprocessed reference corpus not found. "
            "Will download the NLTK corpora and preprocess them with POSNoise."
        )
        nltk.download("brown")
        nltk.download("gutenberg")
        nltk.download("reuters")
        nltk.download("webtext")
        nltk.download("nps_chat")
        reference_sentences = (
            list(nltk.corpus.brown.sents())
            + list(nltk.corpus.gutenberg.sents())
            + list(nltk.corpus.reuters.sents())
            + list(nltk.corpus.webtext.sents())
            + [post.text for post in nltk.corpus.nps_chat.xml_posts()]
        )

        return self._apply_pos_noise(
            reference_sentences, "Applying POS noise to the reference corpus"
        )

    def _load_vocabulary(self):
        """
        Extract the vocabulary from the reference corpus.
        """
        if not self.reference_corpus:
            raise ValueError("Reference corpus is not loaded.")
        pos_counts = Counter()
        for sentence in self.reference_corpus:
            pos_counts.update(sentence)
        filtered_words = {
            word for word, count in pos_counts.items() if count >= self.min_freq
        }
        filtered_words.add("<s>")
        filtered_words.add("</s>")
        filtered_words.add("<UNK>")
        return Vocabulary(filtered_words)

    def _apply_pos_noise(self, sentences, description):
        """
        Apply POSNoise to a set of sentences using multiprocessing.
        """
        num_processes = mp.cpu_count()
        if num_processes > 1:
            num_processes -= 1
        with mp.Pool(processes=num_processes) as pool:
            pos_noised_sentences = list(
                tqdm(
                    pool.imap_unordered(pos_noise.apply_noise, sentences),
                    total=len(sentences),
                    desc=description,
                    disable=self.disable_tqdm,
                )
            )
        return pos_noised_sentences


def _train_ngram_model(sentences, n=3, vocabulary=None):
    """
    Train an n-gram model on the given corpus.
    """
    train, vocab_local = padded_everygram_pipeline(n, sentences)
    if vocabulary is None:
        vocabulary = vocab_local
    model = WittenBellInterpolated(order=n, vocabulary=vocabulary)
    model.fit(train)
    return model


def _get_log_prob(model, sentence):
    """
    Get the log probability of a sentence using the trained model.
    """
    n = model.order
    padded_sentence = ["<s>"] * (n - 1) + sentence + ["</s>"]
    log_prob = 0.0
    for i, word in enumerate(padded_sentence):
        if i < n - 1:
            continue
        context = tuple(padded_sentence[i - n + 1 : i])
        word = padded_sentence[i]
        log_prob += np.log(model.score(word, context) + 1e-10)
    return log_prob / len(sentence)
