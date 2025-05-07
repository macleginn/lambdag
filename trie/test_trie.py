import unittest
from trie import Trie, TrieNode, is_iterable

class TestHelperFunctions(unittest.TestCase):
    def test_is_iterable(self):
        self.assertTrue(is_iterable([]))
        self.assertTrue(is_iterable("hello"))
        self.assertTrue(is_iterable(tuple()))
        self.assertTrue(is_iterable({}))
        self.assertTrue(is_iterable(set()))
        self.assertFalse(is_iterable(123))
        self.assertFalse(is_iterable(None))
        self.assertFalse(is_iterable(True))

class TestTrieNode(unittest.TestCase):
    def test_trie_node_initialization(self):
        node = TrieNode("test_key")
        self.assertEqual(node.key, "test_key")
        self.assertEqual(node.children, {})
        self.assertFalse(node.is_end_of_sequence)

class TestTrie(unittest.TestCase):
    def test_trie_initialization(self):
        trie = Trie()
        self.assertIsNotNone(trie.root)
        self.assertEqual(trie.root.key, '<root_node_key>')
        self.assertEqual(trie.root.children, {})
        self.assertFalse(trie.root.is_end_of_sequence)

    def test_insert_and_get_node(self):
        trie = Trie()
        sequence1 = ["apple", "pie"]
        trie.insert(sequence1)

        node = trie.get_node(sequence1)
        self.assertIsNotNone(node)
        self.assertTrue(node.is_end_of_sequence)
        self.assertEqual(node.key, "pie")

        node_prefix = trie.get_node(["apple"])
        self.assertIsNotNone(node_prefix)
        self.assertFalse(node_prefix.is_end_of_sequence) # "apple" itself wasn't inserted
        self.assertEqual(node_prefix.key, "apple")
        
        # Insert a single-word sequence
        sequence2 = ["apple"]
        trie.insert(sequence2)
        node_prefix_end = trie.get_node(["apple"])
        self.assertIsNotNone(node_prefix_end)
        self.assertTrue(node_prefix_end.is_end_of_sequence)
        self.assertEqual(node_prefix_end.key, "apple")

    def test_insert_multiple_sequences(self):
        trie = Trie()
        sequences = [
            ["a", "b", "c"],
            ["a", "b", "d"],
            ["a", "x", "y"],
            ["b", "c"]
        ]
        for seq in sequences:
            trie.insert(seq)

        self.assertIsNotNone(trie.get_node(["a", "b", "c"]))
        self.assertTrue(trie.get_node(["a", "b", "c"]).is_end_of_sequence)
        self.assertIsNotNone(trie.get_node(["a", "b", "d"]))
        self.assertTrue(trie.get_node(["a", "b", "d"]).is_end_of_sequence)
        self.assertIsNotNone(trie.get_node(["a", "x", "y"]))
        self.assertTrue(trie.get_node(["a", "x", "y"]).is_end_of_sequence)
        self.assertIsNotNone(trie.get_node(["b", "c"]))
        self.assertTrue(trie.get_node(["b", "c"]).is_end_of_sequence)

        self.assertIsNone(trie.get_node(["a", "b", "e"]))
        self.assertFalse(trie.get_node(["a", "b"]).is_end_of_sequence)

    def test_get_node_non_existent(self):
        trie = Trie()
        trie.insert(["hello", "world"])
        self.assertIsNone(trie.get_node(["goodbye", "world"]))
        self.assertIsNone(trie.get_node(["hello", "moon"]))
        self.assertIsNone(trie.get_node(["h"]))

    def test_get_node_with_singleton_key(self):
        trie = Trie()
        trie.insert([1])
        trie.insert([1, 2])

        node1 = trie.get_node(1) # Test non-iterable prefix
        self.assertIsNotNone(node1)
        self.assertTrue(node1.is_end_of_sequence)
        self.assertEqual(node1.key, 1)
        
        node2 = trie.get_node([1, 2])
        self.assertIsNotNone(node2)
        self.assertTrue(node2.is_end_of_sequence)
        self.assertEqual(node2.key, 2)

        self.assertIsNone(trie.get_node(3))

    def test_test_prefix_empty_trie(self):
        trie = Trie()
        self.assertEqual(trie.test_prefix(["a", "b"]), [])

    def test_test_prefix_empty_sequence(self):
        trie = Trie()
        trie.insert(["a", "b"])
        self.assertEqual(trie.test_prefix([]), [])

    def test_test_prefix_no_match(self):
        trie = Trie()
        trie.insert(["apple", "pie"])
        self.assertEqual(trie.test_prefix(["banana"]), [])
        self.assertEqual(trie.test_prefix(["app", "pie"]), []) # "app" not a child of root

    def test_test_prefix_exact_match(self):
        trie = Trie()
        sequence = ["match", "this"]
        trie.insert(sequence)
        self.assertEqual(trie.test_prefix(sequence), sequence)
        self.assertEqual(trie.test_prefix(["match", "this", "extra"]), sequence)

    def test_test_prefix_partial_match_is_valid_sequence(self):
        trie = Trie()
        trie.insert(["a", "b"])
        trie.insert(["a", "b", "c", "d"])
        
        # "a", "b" is a valid sequence
        self.assertEqual(trie.test_prefix(["a", "b", "x"]), ["a", "b"])
        # "a", "b", "c", "d" is the longest valid prefix
        self.assertEqual(trie.test_prefix(["a", "b", "c", "d", "e"]), ["a", "b", "c", "d"])

    def test_test_prefix_partial_match_not_valid_sequence_intermediate(self):
        trie = Trie()
        trie.insert(["long", "sequence", "word"]) # only this full sequence is "valid"
        # "long" or "long", "sequence" are prefixes but not marked as end_of_sequence
        self.assertEqual(trie.test_prefix(["long", "sequence", "other"]), [])
        self.assertEqual(trie.test_prefix(["long", "other"]), [])
        
        trie.insert(["long"]) # Now "long" is a valid sequence
        self.assertEqual(trie.test_prefix(["long", "sequence", "other"]), ["long"])

    def test_test_prefix_multiple_valid_prefixes_in_input(self):
        trie = Trie()
        trie.insert(["a"])
        trie.insert(["a", "b", "c"])
        trie.insert(["a", "b", "c", "d", "e"])

        # Testing "a", "b", "c", "d", "f"
        # "a" is a valid prefix.
        # "a", "b", "c" is a valid prefix.
        # It should return the longest one: ["a", "b", "c"]
        self.assertEqual(trie.test_prefix(["a", "b", "c", "d", "f"]), ["a", "b", "c"])

        # Testing "a", "b", "c", "d", "e", "f"
        # "a", "b", "c", "d", "e" is the longest valid prefix
        self.assertEqual(trie.test_prefix(["a", "b", "c", "d", "e", "f"]), ["a", "b", "c", "d", "e"])
        
        # Testing "a", "x"
        # "a" is the longest valid prefix
        self.assertEqual(trie.test_prefix(["a", "x"]), ["a"])

    def test_test_prefix_complex_case(self):
        trie = Trie()
        trie.insert(["http", "request"])
        trie.insert(["http", "response"])
        trie.insert(["http"])

        self.assertEqual(trie.test_prefix(["http", "request", "body"]), ["http", "request"])
        self.assertEqual(trie.test_prefix(["http", "get"]), ["http"])
        self.assertEqual(trie.test_prefix(["ftp", "request"]), [])

    def test_string_sequences_as_char_trie(self):
        trie = Trie()
        trie.insert("apple")
        trie.insert("apply")
        trie.insert("apricot")
        trie.insert("ape")
        trie.insert("banana")

        # Test get_node with string (implicit char sequence)
        node_apple = trie.get_node("apple")
        self.assertIsNotNone(node_apple)
        self.assertTrue(node_apple.is_end_of_sequence)
        self.assertEqual(node_apple.key, 'e') # last char of "apple"

        node_app = trie.get_node("app")
        self.assertIsNotNone(node_app)
        self.assertFalse(node_app.is_end_of_sequence) # "app" itself was not inserted
        self.assertEqual(node_app.key, 'p')

        node_ap = trie.get_node("ap")
        self.assertIsNotNone(node_ap)
        self.assertFalse(node_ap.is_end_of_sequence)
        self.assertEqual(node_ap.key, 'p')

        self.assertIsNone(trie.get_node("apples"))
        self.assertIsNone(trie.get_node("application"))

        # Test test_prefix with string (implicit char sequence)
        self.assertEqual(trie.test_prefix("applepie"), list("apple"))
        self.assertEqual(trie.test_prefix("application"), [])
        
        trie.insert("app")
        self.assertEqual(trie.test_prefix("application"), list("app"))

        self.assertEqual(trie.test_prefix("apricots"), list("apricot"))
        self.assertEqual(trie.test_prefix("aperture"), list("ape"))
        self.assertEqual(trie.test_prefix("banana"), list("banana"))
        self.assertEqual(trie.test_prefix("bandana"), []) # "ban" is not a valid sequence
        
        trie.insert("ban")
        self.assertEqual(trie.test_prefix("bandana"), list("ban"))

        self.assertEqual(trie.test_prefix("cat"), [])
        self.assertEqual(trie.test_prefix("a"), [])

        trie.insert("a")
        self.assertEqual(trie.test_prefix("alphabet"), list("a"))


if __name__ == '__main__':
    unittest.main()