"""Offline tests for staging HF model files without modifying originals."""
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / "native"))
from prepare_whisper_checkpoint import stage, validate_safetensors

class WhisperCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src = self.root / "input"
        self.src.mkdir()
        (self.src / "config.json").write_text('{"model_type":"whisper"}')
        (self.src / "generation_config.json").write_text('{}')
        (self.src / "preprocessor_config.json").write_text('{}')
        (self.src / "vocab.json").write_text('{}')
        (self.src / "merges.txt").write_text('#version: 0.2\n')
        metadata = {
            "model.encoder.conv1.weight": {"dtype":"F16", "shape":[1], "data_offsets":[0,2]},
            "model.decoder.embed_tokens.weight": {"dtype":"F16", "shape":[1], "data_offsets":[2,4]},
        }
        header = json.dumps(metadata).encode()
        (self.src / "model.safetensors").write_bytes(struct.pack('<Q',len(header))+header+b'\0'*4)

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_fast_tokenizer_is_generated_without_modifying_source(self):
        def fake_builder(folder):
            (folder / "tokenizer.json").write_text('{"model":{"type":"BPE"}}')
        dst = self.root / "staged"
        self.assertTrue(stage(self.src, dst, fake_builder))
        self.assertFalse((self.src / "tokenizer.json").exists())
        self.assertEqual((self.src / "model.safetensors").read_bytes(), (dst / "model.safetensors").read_bytes())
        self.assertTrue(os.path.samefile(self.src / "model.safetensors", dst / "model.safetensors"))

    def test_existing_fast_tokenizer_copied(self):
        (self.src / "tokenizer.json").write_text('{"model":{"type":"BPE"}}')
        self.assertFalse(stage(self.src, self.root / "stage"))

    def test_unmerged_or_pointer_checkpoint_rejected(self):
        (self.src / "model.safetensors").write_text('version https://git-lfs.github.com/spec/v1')
        with self.assertRaisesRegex(ValueError, "weights|pointer"):
            validate_safetensors(self.src / "model.safetensors")

    def test_source_directory_never_overwritten(self):
        with self.assertRaises(ValueError):
            stage(self.src, self.src / "staged")

    def test_wrong_architecture_rejected(self):
        (self.src / "config.json").write_text('{"model_type":"qwen2"}')
        with self.assertRaisesRegex(ValueError, "Whisper"):
            stage(self.src, self.root / "staged")

if __name__ == '__main__':
    unittest.main()
