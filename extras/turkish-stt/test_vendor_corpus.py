"""Model-free regression tests for the persistent vendor Turkish accuracy gate."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

import vendor_corpus as vendor


def make_wav(path: Path, rate=16000):
    with wave.open(str(path), "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(rate)
        recording.writeframes(b"\0\0" * rate)


class VendorCorpusTests(unittest.TestCase):
    def test_inference_requires_cached_adapter_and_never_auto_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            wav = base / "Türkçe konuşma.wav"
            make_wav(wav)
            manifest = base / "reviewed.jsonl"
            manifest.write_text(
                json.dumps({"wav": wav.name, "text": "Iğdır ve İstanbul"},
                           ensure_ascii=False) + "\n", encoding="utf-8"
            )
            args = argparse.Namespace(model="balbal-v2", manifest=manifest,
                                      output=base / "metrics.json",
                                      max_wer=1.0, max_cer=1.0)
            diagnostics = io.StringIO()
            with patch.object(vendor.stt, "load_runner") as load, \
                 contextlib.redirect_stderr(diagnostics):
                load.side_effect = lambda model, offline: (
                    print("Vendor initialization in progress")
                    or (lambda path: (print("Vendor decoding message") or "Iğdır ve İstanbul"))
                )
                report = vendor.evaluate(args)
            load.assert_called_once_with(vendor.stt.MODELS["balbal-v2"], offline=True)
            self.assertEqual(report["aggregate"]["wer"], 0)
            self.assertEqual(report["aggregate"]["cer"], 0)
            self.assertTrue(report["passed"])
            self.assertEqual(len(report["clips"][0]["wav_sha256"]), 64)
            self.assertFalse(report["model_revision_verified"])
            self.assertIn("Vendor initialization", diagnostics.getvalue())
            self.assertIn("Vendor decoding", diagnostics.getvalue())
            self.assertEqual(json.loads(args.output.read_text(encoding="utf-8"))["passed"], True)

            args.max_wer, args.max_cer = None, None
            with patch.object(vendor.stt, "load_runner",
                              return_value=lambda path: "Iğdır ve İstanbul"):
                ungraded = vendor.evaluate(args)
            self.assertIsNone(ungraded["passed"])
            self.assertFalse(ungraded["accuracy_gate_applied"])

            args.max_wer = 0.0
            with patch.object(vendor.stt, "load_runner",
                              return_value=lambda path: "Bambaşka sözler"):
                with self.assertRaisesRegex(SystemExit, "exceeded acceptance"):
                    vendor.evaluate(args)
            self.assertFalse(json.loads(args.output.read_text(encoding="utf-8"))["passed"])

    def test_all_five_vendor_models_and_reject_native_only_models(self):
        self.assertEqual(set(vendor.MODELS),
                         {"balbal-v2", "balbal-manas", "duyu-small",
                          "turkmedstt", "turkspeak-itu"})
        self.assertNotIn("seda-tiny", vendor.MODELS)

    def test_pcm_validation_rejects_bad_format(self):
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / "speech.wav"
            make_wav(wav)
            self.assertEqual(vendor.audio_samples(wav), 16000)
            make_wav(wav, 8000)
            with self.assertRaisesRegex(ValueError, "16 kHz"):
                vendor.audio_samples(wav)

    def test_corpus_micro_average(self):
        rows = [
            {"ground_truth": "bir iki", "text": "bir iki",
             "audio_seconds": 1, "decode_seconds": .2},
            {"ground_truth": "üç dört beş altı", "text": "üç dört beş",
             "audio_seconds": 3, "decode_seconds": .8},
        ]
        totals = vendor.summary(rows)
        self.assertAlmostEqual(totals["wer"], 1 / 6)
        self.assertAlmostEqual(totals["rtf"], .25)
        self.assertGreater(totals["cer"], 0)


if __name__ == "__main__":
    unittest.main()
