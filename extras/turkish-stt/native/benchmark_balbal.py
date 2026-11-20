"""Real Turkish WAV test of the pinned official Balbal Python runtime.
License: Balbal Community License 1.1. CI evaluates models locally;
no weights or vendor executable are bundled or redistributed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
import wave

from benchmark import F, REF, fetch, wer

NAMES = {
    "balbal-v2": "balbal-v2",
    "balbal-manas": "balbal-v2-manas",
}

def benchmark(model_id: str, destination: Path):
    import balbal
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        with fetch(F) as response:
            audio = response.read(2_000_000)
        digest = hashlib.sha1(f"blob {len(audio)}\0".encode() + audio).hexdigest()
        if digest != "47ae56f2500f69d6d0e56d6e1b0f87c6967246fe":
            raise RuntimeError("Audio integrity mismatch")
        wav = root / "turkish.wav"
        wav.write_bytes(audio)
        with wave.open(str(wav), "rb") as reader:
            if (reader.getnchannels(), reader.getframerate(), reader.getsampwidth()) != (1, 16000, 2):
                raise RuntimeError("Unexpected WAV format")
            duration = reader.getnframes() / reader.getframerate()
        began = time.perf_counter()
        recognizer = balbal.load(NAMES[model_id])
        cold = time.perf_counter() - began
        results = []
        for _ in range(2):
            began = time.perf_counter()
            value = recognizer.transcribe(str(wav))
            elapsed = time.perf_counter() - began
            if not isinstance(value, dict) or not isinstance(value.get("text"), str):
                raise TypeError("Balbal returned unexpected result")
            words = value["text"].strip()
            results.append({"text": words, "wer": wer(REF, words),
                            "seconds": elapsed, "rtf": elapsed / duration})
        # Exercise the same multi-clip WER/CER gate used for native models,
        # reusing the recognizer already loaded for the two smoke decodes.
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from vendor_corpus import evaluate
        manifest = root / "reviewed.jsonl"
        manifest.write_text(json.dumps({"wav": wav.name, "text": REF},
                                       ensure_ascii=False) + "\n", encoding="utf-8")
        evaluate(argparse.Namespace(
            model=model_id, manifest=manifest,
            output=destination.parent / "corpus.json",
            max_wer=.8, max_cer=None,
        ), runner=lambda path: recognizer.transcribe(str(path))["text"])
        report = {"model": model_id, "engine": "balbal==0.6.3",
                  "attribution": "Powered by balbal - https://github.com/erayyap/balbal",
                  "reference": REF, "clip": F, "audio_seconds": duration,
                  "cold_load_seconds": cold, "decodes": results}
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if any(not row["text"] or row["wer"] > .8 for row in results):
            raise RuntimeError("Balbal failed the Turkish real-speech smoke test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=NAMES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    opts = parser.parse_args()
    benchmark(opts.model, opts.output)
