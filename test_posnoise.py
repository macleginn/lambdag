import unittest
from unittest.mock import patch, MagicMock, mock_open
from pos_noise import POSNoise, Trie # Assuming Trie is in the same directory or accessible

# Mock spaCy's nlp and Doc objects
class MockSpacyToken:
    def __init__(self, text, pos_):
        self.text = text
        self.pos_ = pos_

class MockSpacyDoc:
    def __init__(self, tokens_data):
        self.tokens = [MockSpacyToken(text, pos) for text, pos in tokens_data]

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, item):
        return self.tokens[item]

class TestPOSNoise(unittest.TestCase):

    @patch('pos_noise.spacy.load')
    @patch('pos_noise.Trie')
    def test_init_trie_success(self, MockTrie, mock_spacy_load):
        # Mock the trie instance and its methods
        mock_trie_instance = MockTrie.return_value
        mock_trie_instance.insert = MagicMock()

        # Mock spaCy nlp object
        mock_nlp = MagicMock()
        mock_spacy_load.return_value = mock_nlp

        # Define how nlp processes lines into tokens
        def nlp_side_effect(text):
            if text == "hello world":
                return [MockSpacyToken("hello", "INTJ"), MockSpacyToken("world", "NOUN")]
            elif text == "good morning":
                return [MockSpacyToken("good", "ADJ"), MockSpacyToken("morning", "NOUN")]
            elif text == "test":
                return [MockSpacyToken("test", "NOUN")]
            return []
        mock_nlp.side_effect = nlp_side_effect

        # Mock file content
        file_content = """
# This is a comment
hello world
good morning
test
"""
        mock_file = mock_open(read_data=file_content)

        with patch('builtins.open', mock_file):
            trie, max_len = POSNoise.init_trie(path='dummy_path.txt')

        self.assertIsInstance(trie, MockTrie)
        mock_spacy_load.assert_called_once_with("en_core_web_sm") # From module level
        
        # Check calls to trie.insert
        calls = [
            unittest.mock.call(["hello", "world"]),
            unittest.mock.call(["good", "morning"]),
            unittest.mock.call(["test"])
        ]
        mock_trie_instance.insert.assert_has_calls(calls, any_order=False)
        self.assertEqual(mock_trie_instance.insert.call_count, 3)

        # Check max_seq_length
        self.assertEqual(max_len, 2) # "hello world" and "good morning" have 2 tokens

    @patch('pos_noise.spacy.load')
    @patch('pos_noise.Trie')
    def test_init_trie_empty_file(self, MockTrie, mock_spacy_load):
        mock_trie_instance = MockTrie.return_value
        mock_nlp = MagicMock()
        mock_spacy_load.return_value = mock_nlp
        
        mock_file = mock_open(read_data="")
        with patch('builtins.open', mock_file):
            trie, max_len = POSNoise.init_trie(path='empty.txt')

        self.assertIsInstance(trie, MockTrie)
        mock_trie_instance.insert.assert_not_called()
        self.assertEqual(max_len, 0)

    @patch('pos_noise.POSNoise.init_trie')
    @patch('pos_noise.spacy.load') # To mock the global nlp
    def test_pos_noise_constructor(self, mock_spacy_load_global, mock_init_trie):
        # Mock the return value of init_trie
        mock_trie_instance = MagicMock(spec=Trie)
        expected_max_len = 5
        mock_init_trie.return_value = (mock_trie_instance, expected_max_len)

        # We pass a dummy trie to constructor, it will be overwritten by the mocked init_trie
        pos_noiser = POSNoise(trie=None) 

        mock_init_trie.assert_called_once_with()
        self.assertIs(pos_noiser.trie, mock_trie_instance)
        self.assertEqual(pos_noiser.max_seq_length, expected_max_len)


    @patch('pos_noise.spacy.load') # Mock the global nlp used in apply_noise
    def test_apply_noise_simple_replacement(self, mock_spacy_load_global):
        # Setup POSNoise instance with a mocked trie
        mock_trie = MagicMock(spec=Trie)
        mock_trie.test_prefix.return_value = [] # No sequences in trie match

        pos_noiser = POSNoise(trie=None) # init_trie will be called
        pos_noiser.trie = mock_trie # Override with our controlled mock
        pos_noiser.max_seq_length = 3 # Arbitrary, won't matter if test_prefix is []

        # Mock the global nlp object for apply_noise
        mock_nlp_global = MagicMock()
        mock_spacy_load_global.return_value = mock_nlp_global
        
        # Simulate spaCy doc object for "this is a test."
        doc_data = [
            ("this", "PRON"), ("is", "AUX"), ("a", "DET"), 
            ("test", "NOUN"), (".", "PUNCT")
        ]
        mock_nlp_global.return_value = MockSpacyDoc(doc_data)

        result = pos_noiser.apply_noise("This is a test.")
        
        mock_nlp_global.assert_called_once_with("this is a test.")
        # test_prefix will be called for "this", "is", "a", "test"
        self.assertEqual(mock_trie.test_prefix.call_count, 4) 
        self.assertEqual(result, ["PRON", "AUX", "DET", "NOUN", "."])

    @patch('pos_noise.spacy.load')
    def test_apply_noise_with_trie_match(self, mock_spacy_load_global):
        mock_trie = MagicMock(spec=Trie)
        pos_noiser = POSNoise(trie=None)
        pos_noiser.trie = mock_trie
        pos_noiser.max_seq_length = 3 

        mock_nlp_global = MagicMock()
        mock_spacy_load_global.return_value = mock_nlp_global
        
        # Simulate spaCy doc for "hello world is fun."
        # "hello world" is in the trie
        doc_data = [
            ("hello", "INTJ"), ("world", "NOUN"), ("is", "AUX"),
            ("fun", "ADJ"), (".", "PUNCT")
        ]
        mock_nlp_global.return_value = MockSpacyDoc(doc_data)

        # Configure trie.test_prefix mock
        # When "hello", "world", "is" is checked, "hello", "world" matches
        # When "is", "fun", "." is checked, nothing matches
        # When "fun", ".", None is checked, nothing matches
        def test_prefix_side_effect(tokens_slice):
            if tokens_slice == ["hello", "world", "is"] or tokens_slice == ["hello", "world"]:
                return ["hello", "world"]
            return []
        mock_trie.test_prefix.side_effect = test_prefix_side_effect

        result = pos_noiser.apply_noise("Hello world is fun.")
        
        mock_nlp_global.assert_called_once_with("hello world is fun.")
        
        # Expected calls to test_prefix:
        # 1. tokens[0:3] -> ["hello", "world", "is"] -> returns ["hello", "world"] (len 2) -> i becomes 2
        # 2. tokens[2:5] -> ["is", "fun", "."] -> returns [] -> i becomes 3
        # 3. tokens[3:6] -> ["fun", "."] -> returns [] -> i becomes 4
        # Punctuation at index 4 is handled separately.
        self.assertEqual(mock_trie.test_prefix.call_count, 3)
        
        self.assertEqual(result, ["hello", "world", "AUX", "ADJ", "."])

    @patch('pos_noise.spacy.load')
    def test_apply_noise_only_punctuation(self, mock_spacy_load_global):
        mock_trie = MagicMock(spec=Trie)
        pos_noiser = POSNoise(trie=None)
        pos_noiser.trie = mock_trie
        pos_noiser.max_seq_length = 1

        mock_nlp_global = MagicMock()
        mock_spacy_load_global.return_value = mock_nlp_global
        doc_data = [("!", "PUNCT"), (".", "PUNCT"), ("?", "PUNCT")]
        mock_nlp_global.return_value = MockSpacyDoc(doc_data)

        result = pos_noiser.apply_noise("! . ?")
        self.assertEqual(result, ["!", ".", "?"])
        mock_trie.test_prefix.assert_not_called() # Punctuation is skipped before trie check

    @patch('pos_noise.spacy.load')
    def test_apply_noise_empty_input(self, mock_spacy_load_global):
        mock_trie = MagicMock(spec=Trie)
        pos_noiser = POSNoise(trie=None)
        pos_noiser.trie = mock_trie
        pos_noiser.max_seq_length = 1

        mock_nlp_global = MagicMock()
        mock_spacy_load_global.return_value = mock_nlp_global
        mock_nlp_global.return_value = MockSpacyDoc([]) # Empty doc

        result = pos_noiser.apply_noise("")
        self.assertEqual(result, [])
        mock_trie.test_prefix.assert_not_called()

    @patch('pos_noise.spacy.load')
    def test_apply_noise_greedy_longest_match(self, mock_spacy_load_global):
        mock_trie = MagicMock(spec=Trie)
        pos_noiser = POSNoise(trie=None)
        pos_noiser.trie = mock_trie
        pos_noiser.max_seq_length = 4 # Allow checking up to 4 tokens

        mock_nlp_global = MagicMock()
        mock_spacy_load_global.return_value = mock_nlp_global
        
        # "New York City" is a 3-token phrase, "New York" is a 2-token phrase.
        # The algorithm should pick "New York City".
        doc_data = [
            ("new", "PROPN"), ("york", "PROPN"), ("city", "PROPN"), ("is", "AUX"), ("big", "ADJ")
        ]
        mock_nlp_global.return_value = MockSpacyDoc(doc_data)

        def test_prefix_side_effect(tokens_slice):
            # Simulating that the trie has "new york city" and "new york"
            # test_prefix should return the longest valid one from the input slice
            if tokens_slice[:3] == ["new", "york", "city"]:
                return ["new", "york", "city"] # This is what trie.test_prefix would do
            if tokens_slice[:2] == ["new", "york"]:
                 return ["new", "york"] # This would be found if "new york city" wasn't
            return []
        mock_trie.test_prefix.side_effect = test_prefix_side_effect

        result = pos_noiser.apply_noise("New York City is big")
        
        # Expected calls:
        # 1. tokens[0:4] (["new", "york", "city", "is"]) -> returns ["new", "york", "city"] (len 3) -> i becomes 3
        # 2. tokens[3:7] (["is", "big"]) -> returns [] -> i becomes 4
        # 3. tokens[4:8] (["big"]) -> returns [] -> i becomes 5
        self.assertEqual(mock_trie.test_prefix.call_count, 3)
        self.assertEqual(result, ["new", "york", "city", "AUX", "ADJ"])

    @patch('pos_noise.spacy.load')
    def test_apply_noise_max_seq_length_slicing(self, mock_spacy_load_global):
        mock_trie = MagicMock(spec=Trie)
        pos_noiser = POSNoise(trie=None)
        pos_noiser.trie = mock_trie
        # Crucially, set max_seq_length to 2.
        # Even if a 3-token phrase exists, it won't be found if we only check 2 tokens at a time.
        pos_noiser.max_seq_length = 2

        mock_nlp_global = MagicMock()
        mock_spacy_load_global.return_value = mock_nlp_global
        
        doc_data = [
            ("a", "DET"), ("b", "NOUN"), ("c", "NOUN"), ("d", "NOUN")
        ]
        mock_nlp_global.return_value = MockSpacyDoc(doc_data)

        # Trie has "a b" and "a b c".
        # test_prefix should be called with slices of max_seq_length
        def test_prefix_side_effect(tokens_slice):
            self.assertLessEqual(len(tokens_slice), 2) # Check the slice length
            if tokens_slice == ["a", "b"]:
                return ["a", "b"]
            if tokens_slice == ["c", "d"]:
                 return ["c", "d"] # Assume "c d" is also in trie for this test
            return []
        mock_trie.test_prefix.side_effect = test_prefix_side_effect

        result = pos_noiser.apply_noise("a b c d")
        
        # Expected calls:
        # 1. tokens[0:2] (["a", "b"]) -> returns ["a", "b"] (len 2) -> i becomes 2
        # 2. tokens[2:4] (["c", "d"]) -> returns ["c", "d"] (len 2) -> i becomes 4
        self.assertEqual(mock_trie.test_prefix.call_count, 2)
        self.assertEqual(result, ["a", "b", "c", "d"])


if __name__ == '__main__':
    unittest.main()