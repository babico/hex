"""Model-free contract tests of the offline Turkish native corpus evaluator."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent / "native"))
import benchmark_corpus as corpus


def write_wav(path, rate=16000):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b"\0\0" * rate)


class CorpusTests(unittest.TestCase):
    def test_micro_wer_and_realtime_factor_are_weighted_correctly(self):
        aggregate = corpus.summarize([
            {"ground_truth": "çok iyi", "text": "çok iyi",
             "audio_seconds": 1, "decode_seconds": .2},
            {"ground_truth": "bir iki üç dört", "text": "bir iki üç",
             "audio_seconds": 3, "decode_seconds": 1.8},
        ])
        self.assertEqual(aggregate["clips"], 2)
        self.assertEqual(aggregate["reference_words"], 6)
        self.assertAlmostEqual(aggregate["wer"], 1 / 6)
        self.assertAlmostEqual(aggregate["rtf"], .5)

    def test_model_specific_weights_are_recorded_without_a_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp)
            nested = weights / "sherpa-onnx-tdt"
            nested.mkdir()
            for name in ("encoder.int8.onnx", "decoder.int8.onnx",
                         "joiner.int8.onnx", "tokens.txt"):
                (nested / name).write_bytes(name.encode("utf-8"))
            manifest = corpus.weight_manifest("ufakzeka-kulak", weights)
            self.assertEqual(len(manifest), 4)
            self.assertTrue(all(row["name"].startswith("sherpa-onnx-tdt/")
                                for row in manifest))
            (nested / "tokens.txt").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "tokens.txt"):
                corpus.weight_manifest("ufakzeka-kulak", weights)

    def test_audio_requires_pcm16_mono_16khz(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "speech.wav"
            write_wav(path)
            self.assertEqual(corpus.validated_audio(path), 16000)
            write_wav(path, rate=8000)
            with self.assertRaisesRegex(ValueError, "16 kHz"):
                corpus.validated_audio(path)

    def test_evaluation_uses_single_worker_and_enforces_corpus_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / "weights"
            weights.mkdir()
            for name in ("encoder.int8.onnx", "decoder.onnx",
                         "joiner.int8.onnx", "tokens.txt"):
                (weights / name).write_bytes(f"reviewed-{name}".encode("utf-8"))
            executable = root / "worker"
            executable.write_bytes(b"fake native worker")
            records = [
                ("bir.wav", "çok iyi"),
                ("iki.wav", "bir iki üç dört"),
            ]
            manifest = root / "corpus.jsonl"
            manifest.write_text("".join(
                json.dumps({"wav": name, "text": truth}, ensure_ascii=False) + "\n"
                for name, truth in records
            ), encoding="utf-8")
            for name, _ in records:
                write_wav(root / name)

            class FakeWorker:
                instances = 0
                def __init__(self, binary, model, directory):
                    self.binary, self.model, self.directory = binary, model, directory
                    self.calls = 0
                    FakeWorker.instances += 1

                def __enter__(self):
                    return self

                def __exit__(self, *_):
                    return None

                def transcribe(self, wav, count):
                    self.calls += 1
                    self_count = count
                    assert self_count == 16000
                    return "çok iyi" if wav.name == "bir.wav" else "bir iki üç"

            args = argparse.Namespace(
                worker=executable, weights=weights, model="seda-tiny",
                manifest=manifest, output=root / "metrics.json",
                max_wer=.2, max_cer=None,
            )
            with patch.object(corpus, "ResidentWorker", FakeWorker):
                result = corpus.evaluate(args)
            self.assertEqual(FakeWorker.instances, 1)
            self.assertTrue(result["passed"])
            self.assertEqual(len(result["weight_files"]), 4)
            self.assertEqual(len(result["weight_files"][0]["sha256"]), 64)
            self.assertAlmostEqual(result["aggregate"]["wer"], 1 / 6)
            self.assertEqual(len(result["clips"]), 2)

            args.max_wer = .1
            with patch.object(corpus, "ResidentWorker", FakeWorker):
                with self.assertRaisesRegex(SystemExit, "below required threshold"):
                    corpus.evaluate(args)
            self.assertFalse(json.loads(args.output.read_text(encoding="utf-8"))["passed"])
            self.assertEqual(FakeWorker.instances, 2)


if __name__ == "__main__":
    unittest.main()
