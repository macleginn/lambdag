"""
A function for applying POS noise to the input sentence based on the paper

Oren Halvani and Lukas Graner. POSNoise: An Effective Countermeasure Against 
Topic Biases in Authorship Analysis. In Proceedings of the 16th International Conference 
on Availability, Reliability and Security, ARES ’21, New York, NY, USA, 2021. 
Association for Computing Machinery. https://arxiv.org/pdf/2403.08462
"""

import string
import spacy
from trie_py.trie import Trie

nlp = spacy.load("en_core_web_sm")

POS_TAGS = [
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X"
]

class POSNoise:
    def __init__(self):
        self.trie, self.max_seq_length = self.init_trie()

    @staticmethod
    def init_trie(path='data/POSNoise_PatternList_Ver_2_1.txt'):
        """
        Initializes the trie with the word sequences from the file
        split into tokens using the reference spacy model and records
        the maximum sequence length.
        """
        trie = Trie()
        max_seq_length = 0
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                tokens = [token.text for token in nlp(line)]
                trie.insert(tokens)
                max_seq_length = max(max_seq_length, len(tokens))
        return trie, max_seq_length
    
    @staticmethod
    def process_pretokenized(text_list):
        doc = spacy.tokens.Doc(nlp.vocab, text_list)
        for pipe_name in nlp.pipe_names:
            doc = nlp.get_pipe(pipe_name)(doc)
        return doc

    def apply_noise(self, text):
        """
        Applies POS noise to the input text by replacing words with their
        POS tags, except for punctuation signs and words from the allow-list
        of expressions recorded in the trie.

        The algorithm is as follows:
        1. Tokenize the input text using the reference spacy model.
        2. Record POS tags of the tokens.
        3. Create a binary mask for the tokens, where 1 indicates that the token
           is a punctuation sign or belongs to an expression from the allow-list.
        4. Replace the tokens marked with 0 in the binary mask with their POS tags.
        """
        if type(text) == str:
            doc = nlp(text.lower())
        elif type(text) == list:
            doc = self.process_pretokenized(list(map(lambda s: s.lower(), text)))
        else:
            raise ValueError("Input must be a string or a list of strings.")
        
        tokens = [token.text for token in doc]
        pos_tags = [token.pos_ for token in doc]
        mask = [0] * len(tokens)

        # Create a binary mask
        i = 0
        while i < len(tokens):
            # Keep punctuation signs
            # also check that the POS tag for a non-punctuation token is not PUNCT
            if tokens[i] in string.punctuation or tokens[i] in {"``", "''"}:
                mask[i] = 1
                i += 1
                continue

            # The token is _not_ a punctuation sign; if its POS tag is PUNCT,
            # it is something weird.
            if pos_tags[i] == "PUNCT":
                pos_tags[i] = "X"

            # Greedily check for the longest sequence in the trie
            # starting from the current token.
            longest_prefix = self.trie.test_prefix(
                tokens[i : i + self.max_seq_length])
            if longest_prefix:
                # This looks dangerous, since assigning to a slice in Python can silently
                # extend the underlying list, but it is actually safe because longest_prefix
                # cannot be longer than the remainder of the sequence.
                mask[i : i + len(longest_prefix)] = [1] * len(longest_prefix)
                i += len(longest_prefix)
            else:
                mask[i] = 0
                i += 1

        # Replace tokens with POS tags based on the mask
        noisy_tokens = [
            pos_tags[i] if mask[i] == 0 else tokens[i]
            for i in range(len(tokens))
        ]

        return noisy_tokens
