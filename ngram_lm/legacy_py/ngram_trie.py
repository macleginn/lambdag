"""An implementation of a trie that stores counts of inserted sequences."""

class TrieNodeWithCount:
    def __init__(self, node_key):
        self.key = node_key
        self.children = {}
        self.count = 0

class TrieWithCounts:
    def __init__(self):
        self.root = TrieNodeWithCount('<root_node_key>')
    
    def insert(self, sequence):
        node = self.root
        for token in sequence:
            if token not in node.children:
                node.children[token] = TrieNodeWithCount(token)
            node = node.children[token]
            node.count += 1

    def get_sequence_stats(self, sequence):
        """
        Returns the count of the sequence in the trie,
        as well as the counts of all its prefixes.
        """
        node = self.root
        counts = []
        for token in sequence:
            if token not in node.children:
                break
            node = node.children[token]
            counts.append(node.count)
        return counts
        