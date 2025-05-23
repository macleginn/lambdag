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

import sys
from pathlib import Path

current_dir = Path(__file__).parent
import_dir = current_dir / 'ngram_lm'
sys.path.append(str(import_dir))

import random
import os
import zipfile
import json
import multiprocessing as mp
from collections import defaultdict, Counter
import numpy as np
import nltk
from tqdm.auto import tqdm
from .pos_noise.pos_noise import POSNoise
from .ngram_lm.ngram_lm import NGramLM

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
        disable_multiprocessing=False  # For online processing of short texts
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
        self.disable_multiprocessing = disable_multiprocessing
        self.n = n
        self.N = N
        if sample_size <= 0:
            raise ValueError("Sample size must be greater than 0.")
        self.sample_size = sample_size
        self.min_freq = min_freq
        self.reference_corpus = []
        self.vocabulary = set()
        if reference_corpus is not None:
            self._preprocess_reference_corpus(reference_corpus)
        else:
            self._preprocess_reference_corpus(self._load_reference_corporus())
        self.reference_models = []
        self.known_author_models = {}
        self.reference_models_cache = defaultdict(dict)
        # self.known_author_model_cache = {}
        self._train_reference_models()

    def _train_reference_models(self):
        """
        Train reference models on random subsets of the reference corpus in parallel.
        """
        self.reference_models = []
        self.reference_models_cache = defaultdict(dict)

        if self.N == 0:
            return
        if not self.reference_corpus:
            raise ValueError(
                "Reference corpus is empty. Cannot train reference models."
            )

        current_sample_size = min(self.sample_size, len(self.reference_corpus))
        for _ in tqdm(
            range(self.N),
            desc="Training reference models",
            disable=self.disable_tqdm,
            leave=False,
        ):
            # TODO: sample sentence chunks to have sentence boundaries
            # in the reference models
            training_set = random.sample(self.reference_corpus, current_sample_size)
            model = NGramLM(n=self.n)
            model.fit(training_set)
            self.reference_models.append(model)

    def _replace_unknown_words(self, sentences):
        if not self.vocabulary:
            raise ValueError("Vocabulary is empty. Please train the model first.")
        result = []
        for sentence in sentences:
            new_sentence = []
            for word in sentence:
                if word not in self.vocabulary:
                    new_sentence.append("<UNK>")
                else:
                    new_sentence.append(word)
            result.append(new_sentence)
        return result

    def train_known_author_model(self, sentences, author_id):
        """
        Train the known author model on the provided sentences.
        """
        # self.known_author_model_cache = {}
        sentences_with_pos_noise = self._apply_pos_noise(
            sentences, "Applying POS noise to the known-author corpus"
        )
        sentences_with_pos_noise = self._replace_unknown_words(sentences_with_pos_noise)
        self.known_author_models[author_id] = NGramLM(n=self.n)
        self.known_author_models[author_id].fit(sentences_with_pos_noise)

    def compute_lambda_g(
        self,
        sentences,
        author_id,
        averaging=True,
        clipping=True
    ):
        """
        Compute the LambdaG score for the provided sentences.
        Averaging removes the dependence of the score on input length.
        If clipping is set to true, individual scores will be clamped
        to the range [-1, 1].
        """
        if not self.reference_models:
            raise ValueError("Reference models have not been trained.")
        if author_id not in self.known_author_models:
            raise ValueError("Known author model has not been trained.")
        sentences_with_pos_noise = self._apply_pos_noise(
            sentences, "Applying POS noise to the unknown-author corpus"
        )
        result = 0.0
        ngrams = []
        for sentence in tqdm(
            sentences_with_pos_noise,
            desc="Computing LambdaG score",
            disable=self.disable_tqdm,
            leave=False,
        ):
            padded_sentence = ["<s>"] * (self.n - 1) + sentence + ["</s>"]
            for i in range(len(padded_sentence)):
                if i < self.n:
                    continue
                ngrams.append(padded_sentence[i - self.n : i])
        known_author_probabilities = np.log(
            self.known_author_models[author_id].get_ngram_probabilities(ngrams)
        )
        for model in self.reference_models:
            reference_probabilities = np.log(model.get_ngram_probabilities(ngrams))
            log_ratios = known_author_probabilities - reference_probabilities
            if clipping:
                log_ratios = np.clip(log_ratios, -1.0, 1.0)
            result += np.sum(log_ratios)
        if averaging:
            result /= len(ngrams)
        return 1.0 / self.N * result

    def _preprocess_reference_corpus(self, reference_corpus):
        """
        Preprocess the reference corpus by replacing rare words with <UNK>.
        This also fixes the vocabulary of the model.
        """
        word_counts = Counter()
        for sentence in reference_corpus:
            word_counts.update(sentence)
        filtered_corpus = []
        for sentence in reference_corpus:
            filtered_sentence = [
                word if word_counts[word] >= self.min_freq else "<UNK>"
                for word in sentence
            ]
            filtered_corpus.append(filtered_sentence)
            self.vocabulary.update(filtered_sentence)
        self.vocabulary.add("<s>")
        self.vocabulary.add("</s>")
        self.vocabulary.add("<UNK>")
        self.reference_corpus = filtered_corpus

    def _load_reference_corporus(self):
        """
        Try loading the reference corpus from a zip file. If it fails, 
        download the corpora from NLTK and preprocess them with POS noise.
        """
        path_to_zip = Path(__file__).parent / "data" / "reference_corpus_w_pos_noise.zip"
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

    def _apply_pos_noise(self, sentences, description):
        """
        Apply POSNoise to a set of sentences using multiprocessing.
        """
        num_processes = mp.cpu_count()
        if num_processes > 1:
            num_processes -= 1
        if num_processes == 1 or self.disable_multiprocessing:
            return list(
                tqdm(
                    map(
                        pos_noise.apply_noise,
                        sentences
                    ),
                    total=len(sentences),
                    disable=self.disable_tqdm
                )
            )
        else:
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
