"""Dependency-free unit tests of the native Whisper GGUF parity gate."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / "native"))
from validate_whisper_conversion import normalize, error_rate, compare


class ParityTests(unittest.TestCase):
    def test_turkish_words_and_punctuation(self):
        self.assertEqual(normalize("Bazı festivaller, küçük çocuklu AİLELER için!"),
                         "bazı festivaller küçük çocuklu aileler için")
        self.assertEqual(error_rate("Merhaba, dünya!", "merhaba dünya"), 0)

    def test_word_error_rate(self):
        self.assertEqual(error_rate("bir iki üç", "bir üç"), 1 / 3)
        self.assertEqual(error_rate("", ""), 0)
        self.assertEqual(error_rate("", "hayır"), 1)

    def test_reject_empty_and_significantly_different_inference(self):
        with self.assertRaises(ValueError):
            compare("merhaba", "", .35)
        self.assertFalse(compare("Merhaba dünya nasılsın?", "farklı sonuç", .35)["passed"])
        self.assertTrue(compare("Merhaba, dünya.", "merhaba dünya", .35)["passed"])

    def test_truth_is_optional_and_diagnostic(self):
        actual = compare("merhaba dünya", "merhaba dünya", .35, expected="merhaba dünya")
        self.assertEqual(actual["native_wer"], 0)
        self.assertTrue(actual["passed"])

    def test_invalid_parity_budget(self):
        with self.assertRaises(ValueError):
            compare("x", "y", -0.1)


if __name__ == "__main__":
    unittest.main()
