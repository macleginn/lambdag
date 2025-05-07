"""
An implementation of a trie that allows for step-by-step traversal.
It stores and retrieves iterables of hasahbles objects. If a bare string is passed,
it will be treated as a sequence of characters. In order to use strings as keys,
pass them inside a list or tuple.
"""

from collections.abc import Iterable


def is_iterable(obj):
    return isinstance(obj, Iterable)


class TrieNode:
    def __init__(self, node_key):
        self.key = node_key
        self.children = {}
        self.is_end_of_sequence = False

class Trie:
    def __init__(self):
        self.root = TrieNode('<root_node_key>')
    
    def insert(self, sequence):
        node = self.root
        for token in sequence:
            if token not in node.children:
                node.children[token] = TrieNode(token)
            node = node.children[token]
        node.is_end_of_sequence = True
    
    def get_node(self, prefix):
        """Returns the node at the end of the prefix, or None if not found."""
        if not is_iterable(prefix):  # A singleton key
            prefix = [prefix]
        node = self.root
        for token in prefix:
            if token not in node.children:
                return None
            node = node.children[token]
        return node
    
    def test_prefix(self, sequence):
        """
        Returns the longest prefix of the sequence the exists in the trie
        and is a valid sequence.
        """
        node = self.root
        longest_prefix = []
        tmp = []
        for token in sequence:
            if token not in node.children:
                break
            tmp.append(token)
            node = node.children[token]
            if node.is_end_of_sequence:
                for j in range(len(longest_prefix), len(tmp)):
                    longest_prefix.append(tmp[j])
        return longest_prefix
    
    