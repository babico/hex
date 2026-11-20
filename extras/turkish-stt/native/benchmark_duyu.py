"""End-to-end local Python/ONNX Duyu-small reference benchmark on a real Turkish clip.
Source code is obtained from a pinned Apache-2.0 model revision; never load
unreviewed arbitrary HF revisions or use trust_remote_code.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import wave

from huggingface_hub import snapshot_download
from benchmark import F, REF, fetch, wer

REPO = "KaanAydinli/Duyu-small"
REVISION = "5b7f76972cd82ae6c2d66ec8e8c7b6782d5c8803"
FILES = [
    "duyu_small/*", "int8/*", "passes.npz", "fbank.npz", "spm.model",
    "engine.json", "config.json", "pyproject.toml", "LICENSE",
]

def main():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        source = root / "source"
        snapshot_download(repo_id=REPO, revision=REVISION,
                          allow_patterns=FILES, local_dir=source)
        if not (source / "LICENSE").is_file():
            raise RuntimeError("Duyu Apache-2.0 attribution file is missing")
        with fetch(F) as res:
            audio = res.read(2_000_000)
        digest = hashlib.sha1(f"blob {len(audio)}\0".encode() + audio).hexdigest()
        if digest != "47ae56f2500f69d6d0e56d6e1b0f87c6967246fe":
            raise RuntimeError("Audio fixture checksum mismatch")
        wav = root / "speech.wav"
        wav.write_bytes(audio)
        with wave.open(str(wav), "rb") as w:
            duration = w.getnframes() / w.getframerate()
        sys.path.insert(0, str(source))
        os.environ["DUYU_SMALL_DIR"] = str(source)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import stt
        loaded = stt.MODELS["duyu-small"]
        started = time.perf_counter()
        runner = stt.load_runner(loaded, offline=True)
        cold = time.perf_counter() - started
        results = []
        for i in range(2):
            started = time.perf_counter()
            result = runner(wav)
            elapsed = time.perf_counter() - started
            results.append({"text": result, "seconds": elapsed, "rtf": elapsed/duration, "wer": wer(REF, result)})
        # Share the same reviewed ground-truth corpus evaluator as native
        # and Balbal backends, reusing Duyu's warm ONNX model instance.
        from vendor_corpus import evaluate
        from argparse import Namespace
        manifest = root / "reviewed.jsonl"
        manifest.write_text(json.dumps({"wav": wav.name, "text": REF},
                                       ensure_ascii=False) + "\n", encoding="utf-8")
        evaluate(Namespace(
            model="duyu-small", manifest=manifest,
            output=Path("metrics/duyu-corpus.json"),
            max_wer=.8, max_cer=None,
        ), runner=runner)
        metrics = {"model": "duyu-small", "repo": REPO, "revision": REVISION,
                   "cold_load_seconds": cold, "audio_seconds": duration, "results": results}
        Path("metrics").mkdir(exist_ok=True)
        Path("metrics/duyu.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=True, indent=2))
        if any(not row["text"].strip() or row["wer"] > .8 for row in results):
            raise RuntimeError("Duyu-small failed real Turkish speech smoke test")

if __name__ == "__main__":
    main()
