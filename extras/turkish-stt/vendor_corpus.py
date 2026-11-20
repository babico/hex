#!/usr/bin/env python3
"""Offline multi-clip Turkish STT evaluation for Hex's five Python vendor adapters.

Requires reviewed local 16 kHz mono PCM16 WAVs and a JSONL transcript
manifest. The adapter is initialized once; no model is installed or license
accepted by this tool. All output files remain local.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import time
import wave

import stt
from native.validate_whisper_conversion import error_counts, load_manifest

MODELS = ("balbal-v2", "balbal-manas", "duyu-small",
          "turkmedstt", "turkspeak-itu")
MAX_SAMPLES = 16000 * 600


def audio_samples(path: Path) -> int:
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(),
                wav.getcomptype()) != (16000, 1, 2, "NONE"):
            raise ValueError(f"Expected mono 16 kHz PCM16 WAV: {path}")
        count = wav.getnframes()
        if not 0 < count <= MAX_SAMPLES:
            raise ValueError(f"Audio sample count out of bounds: {path}")
        return count


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def summary(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Corpus must contain audio")
    totals = {
        "clips": len(rows), "reference_words": 0,
        "reference_characters": 0, "word_errors": 0,
        "character_errors": 0, "audio_seconds": 0.0, "decode_seconds": 0.0,
    }
    for row in rows:
        edits, words = error_counts(row["ground_truth"], row["text"])
        chars, characters = error_counts(row["ground_truth"], row["text"],
                                         characters=True)
        totals["word_errors"] += edits
        totals["reference_words"] += words
        totals["character_errors"] += chars
        totals["reference_characters"] += characters
        totals["audio_seconds"] += row["audio_seconds"]
        totals["decode_seconds"] += row["decode_seconds"]
    if not totals["reference_words"] or not totals["reference_characters"]:
        raise ValueError("Reference corpus must contain speech")
    totals["wer"] = totals["word_errors"] / totals["reference_words"]
    totals["cer"] = totals["character_errors"] / totals["reference_characters"]
    totals["rtf"] = totals["decode_seconds"] / totals["audio_seconds"]
    return totals


def evaluate(args: argparse.Namespace) -> dict:
    if args.model not in MODELS:
        raise ValueError(f"Unsupported vendor evaluation model: {args.model}")
    for attr in ("max_wer", "max_cer"):
        value = getattr(args, attr)
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"{attr} must be between 0 and 1")
    clips = load_manifest(args.manifest)
    # Validate all audio before initializing expensive vendor engines.
    samples = [audio_samples(path) for path, _ in clips]
    model = stt.MODELS[args.model]
    with contextlib.redirect_stdout(sys.stderr):
        runner = stt.load_runner(model, offline=True)
    results = []
    for (path, truth), count in zip(clips, samples):
        start = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            result = runner(path)
        elapsed = time.perf_counter() - start
        if not isinstance(result, str) or not result.strip():
            raise RuntimeError(f"{args.model} returned no speech for {path}")
        edits, words = error_counts(truth, result)
        chars, characters = error_counts(truth, result, characters=True)
        seconds = count / 16000
        results.append({
            "wav": str(path), "wav_sha256": digest(path),
            "ground_truth": truth, "text": result.strip(),
            "wer": edits / words, "cer": chars / characters,
            "audio_seconds": seconds, "decode_seconds": elapsed,
            "rtf": elapsed / seconds,
        })
    totals = summary(results)
    thresholded = args.max_wer is not None or args.max_cer is not None
    passed = ((args.max_wer is None or totals["wer"] <= args.max_wer)
              and (args.max_cer is None or totals["cer"] <= args.max_cer)
              ) if thresholded else None
    report = {
        "model": args.model, "engine": "persistent Python vendor adapter",
        "repository": model["repository"], "adapter": model["adapter"],
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": digest(args.manifest),
        "model_revision_verified": False,
        "clips": results, "aggregate": totals,
        "max_wer": args.max_wer, "max_cer": args.max_cer,
        "accuracy_gate_applied": thresholded, "passed": passed,
        "note": "No model weights were downloaded by the evaluator. Verify vendor cache revisions separately. A short corpus does not establish population accuracy.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    if passed is False:
        raise SystemExit(f"Vendor corpus WER/CER exceeded acceptance thresholds: {args.output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-wer", type=float)
    parser.add_argument("--max-cer", type=float)
    report = evaluate(parser.parse_args())
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
