"""Dependency-free unit tests of the native Whisper GGUF parity gate."""
import argparse
import json
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent / "native"))
import validate_whisper_conversion as gate
from validate_whisper_conversion import normalize, error_rate, compare, corpus_accuracy, load_manifest


class ParityTests(unittest.TestCase):
    def test_turkish_words_and_punctuation(self):
        self.assertEqual(normalize("Bazı festivaller, küçük çocuklu AİLELER için!"),
                         "bazı festivaller küçük çocuklu aileler için")
        self.assertEqual(normalize("IĞDIR ve İSTANBUL"), "ığdır ve istanbul")
        self.assertEqual(error_rate("Merhaba, dünya!", "merhaba dünya"), 0)
        self.assertEqual(error_rate("Iğdır", "İğdir"), 1)

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


class CorpusTests(unittest.TestCase):
    def test_micro_wer_weights_clips_by_reference_length(self):
        rows = [
            compare("çok iyi", "çok iyi", .35, expected="çok iyi"),
            compare("bir iki üç dört", "bir iki üç", .35,
                    expected="bir iki üç dört"),
        ]
        aggregate = corpus_accuracy(rows)
        self.assertEqual(aggregate["clips"], 2)
        self.assertEqual(aggregate["words"], 6)
        self.assertEqual(aggregate["reference_wer"], 0)
        self.assertAlmostEqual(aggregate["native_wer"], 1 / 6)
        self.assertGreater(aggregate["native_cer"], 0)

    def test_manifest_accepts_relative_unicode_audio_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            wav = root / "Türkçe konuşma.wav"
            wav.touch()
            manifest = root / "clips.jsonl"
            manifest.write_text(
                json.dumps({"wav": wav.name, "text": "Iğdır ve İstanbul"},
                           ensure_ascii=False) + "\n", encoding="utf-8",
            )
            self.assertEqual(load_manifest(manifest), [(wav.resolve(), "Iğdır ve İstanbul")])
            manifest.write_text(manifest.read_text(encoding="utf-8") * 2,
                                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate WAV"):
                load_manifest(manifest)
            manifest.write_text('{"wav":"speech.wav","text":""}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "nonempty text"):
                load_manifest(manifest)

    def test_manifest_corpus_runs_reference_only_once_and_gates_accuracy(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("model.gguf", "native-cli", "model.safetensors"):
                (root / name).touch()
            for name in ("clip-1.wav", "clip-2.wav"):
                (root / name).touch()
            manifest = root / "eval.jsonl"
            manifest.write_text(
                "".join(json.dumps({"wav": f"clip-{i}.wav", "text": truth},
                                   ensure_ascii=False) + "\n"
                        for i, truth in ((1, "çok iyi"), (2, "bir iki üç dört"))),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                gguf=root / "model.gguf", checkpoint=root,
                native_cli=root / "native-cli", manifest=manifest, wav=None,
                model="turkspeak-itu", output=root / "metrics.json",
                max_parity_wer=.35, max_native_wer=.2,
            )
            with patch.object(gate, "read_audio", return_value=object()), \
                 patch.object(gate, "load_reference", return_value="cached") as load, \
                 patch.object(gate, "reference_text",
                              side_effect=["çok iyi", "bir iki üç dört"]) as reference, \
                 patch.object(gate, "native_text",
                              side_effect=["çok iyi", "bir iki üç"]), \
                 patch.object(gate, "sha256", return_value="fixture-sha256"):
                gate.run(args)
            load.assert_called_once_with(root.resolve())
            self.assertEqual(reference.call_count, 2)
            self.assertEqual(reference.call_args_list[0].args[0], "cached")
            report = json.loads(args.output.read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            self.assertAlmostEqual(report["aggregate"]["native_wer"], 1 / 6)
            self.assertEqual(len(report["clips"]), 2)

            args.max_native_wer = .1
            with patch.object(gate, "read_audio", return_value=object()), \
                 patch.object(gate, "load_reference", return_value="cached"), \
                 patch.object(gate, "reference_text",
                              side_effect=["çok iyi", "bir iki üç dört"]), \
                 patch.object(gate, "native_text",
                              side_effect=["çok iyi", "bir iki üç"]), \
                 patch.object(gate, "sha256", return_value="fixture-sha256"):
                with self.assertRaisesRegex(SystemExit, "GGUF reference parity failed"):
                    gate.run(args)
            self.assertFalse(json.loads(args.output.read_text(encoding="utf-8"))["passed"])


if __name__ == "__main__":
    unittest.main()
