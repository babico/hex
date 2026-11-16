import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("stt", Path(__file__).with_name("stt.py"))
stt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stt)

class RegistryTests(unittest.TestCase):
    def test_registry_has_nine_unique_models(self):
        self.assertEqual(len(stt.MODELS), 9)
        for model in stt.MODELS.values():
            self.assertIn("tr", model["languages"])
            self.assertIn(model["adapter"], {"balbal", "mihu", "duyu", "kulak", "seda", "whisper"})
            self.assertTrue(model["dependencies"])

    def test_list(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(stt.main(["list"]), 0)
        self.assertEqual(len(json.loads(out.getvalue())), 9)

    def test_dispatch_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / "audio.wav"
            wav.touch()
            for name in stt.MODELS:
                out = io.StringIO()
                with patch.object(stt, "recognize", return_value="Merhaba") as adapter, contextlib.redirect_stdout(out):
                    self.assertEqual(stt.main(["transcribe", name, str(wav), "--offline"]), 0)
                    adapter.assert_called_once_with(stt.MODELS[name], wav.resolve(), True)
                self.assertEqual(json.loads(out.getvalue()), {"model": name, "text": "Merhaba"})

if __name__ == "__main__":
    unittest.main()
