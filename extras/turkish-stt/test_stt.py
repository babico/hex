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

class PersistentTests(unittest.TestCase):
    def test_balbal_loads_once_and_dispatches_correct_variant(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        import sys
        fake = MagicMock()
        fake.load.return_value.transcribe.return_value = {"text": "merhaba"}
        with patch.dict(sys.modules, {"balbal": fake}):
            for name, upstream in [("balbal-v2", "balbal-v2"), ("balbal-manas", "balbal-v2-manas")]:
                fake.load.reset_mock()
                runner = stt.load_runner(stt.MODELS[name], offline=True)
                self.assertEqual(runner(Path("speech.wav")), "merhaba")
                self.assertEqual(runner(Path("speech.wav")), "merhaba")
                fake.load.assert_called_once_with(upstream)

    def test_serve_one_load_for_multiple_clips_and_errors(self):
        inp = io.StringIO("/tmp/a.wav\n/tmp/b.wav\n")
        out = io.StringIO()
        calls = []
        def transcribe(path):
            calls.append(str(path))
            if len(calls) == 2:
                raise ValueError("decode error")
            return "İyi günler"
        with patch.object(stt, "load_runner", return_value=transcribe) as loader:
            self.assertEqual(stt.serve(stt.MODELS["balbal-v2"], True, inp, out), 0)
            loader.assert_called_once()
        messages = [json.loads(s) for s in out.getvalue().splitlines()]
        self.assertEqual(messages, [{"ready": True}, {"text": "İyi günler"}, {"error": "decode error"}])
        self.assertEqual(calls, ["/tmp/a.wav", "/tmp/b.wav"])

    def test_download_dispatches_explicit_model_load(self):
        out = io.StringIO()
        with patch.object(stt, "load_runner") as loader, contextlib.redirect_stdout(out):
            self.assertEqual(stt.main(["download", "balbal-v2"]), 0)
            loader.assert_called_once_with(stt.MODELS["balbal-v2"], offline=False)
        self.assertEqual(json.loads(out.getvalue()), {"model": "balbal-v2", "cached": True})
