"""Offline parity gate for a license-reviewed Whisper checkpoint and native GGUF.

Requires the same pinned Python conversion environment, a built transcribe-cli,
and one or more locally supplied 16 kHz mono PCM16 WAVs. Never downloads weights
or audio; original checkpoints are not modified; models are not auto-registered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import time
import unicodedata
import wave
from pathlib import Path


def normalize(text: str) -> str:
    """Casefold, ignore punctuation, preserve Turkish letters and digits."""
    # Turkish dotted and dotless I are different letters. Unicode casefold
    # alone maps capital ASCII I to i, incorrectly merging "I" with "İ".
    folded = (unicodedata.normalize("NFKC", text)
              .translate(str.maketrans({"I": "ı", "İ": "i"}))
              .casefold().replace("\u0307", ""))
    return " ".join(re.findall(r"\w+", folded))


def error_counts(reference: str, prediction: str, *, characters: bool = False) -> tuple[int, int]:
    """Levenshtein edit count and reference length for corpus-level micro WER/CER."""
    a = list(normalize(reference)) if characters else normalize(reference).split()
    b = list(normalize(prediction)) if characters else normalize(prediction).split()
    previous = list(range(len(b) + 1))
    for i, token in enumerate(a, 1):
        current = [i]
        for j, candidate in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j-1] + (candidate != token)))
        previous = current
    return previous[-1], len(a)


def error_rate(reference: str, prediction: str, *, characters: bool = False) -> float:
    edits, count = error_counts(reference, prediction, characters=characters)
    return edits / count if count else float(bool(edits))


def compare(reference: str, native: str, max_parity_wer: float,
            expected: str | None = None) -> dict:
    if not 0.0 <= max_parity_wer <= 1.0:
        raise ValueError("Expected max parity WER between 0 and 1")
    if not normalize(reference) or not normalize(native):
        raise ValueError("Both reference and native transcripts must contain speech")
    metrics = {
        "reference": reference.strip(), "native": native.strip(),
        "parity_wer": error_rate(reference, native),
        "parity_cer": error_rate(reference, native, characters=True),
        "max_parity_wer": max_parity_wer,
    }
    if expected is not None:
        metrics["ground_truth"] = expected
        metrics["reference_wer"] = error_rate(expected, reference)
        metrics["native_wer"] = error_rate(expected, native)
    metrics["passed"] = metrics["parity_wer"] <= max_parity_wer
    return metrics


def read_audio(path: Path):
    import numpy as np
    with wave.open(str(path), "rb") as stream:
        if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth(),
                stream.getcomptype()) != (16000, 1, 2, "NONE"):
            raise ValueError(f"{path} must be mono 16 kHz PCM16 WAV")
        frames = stream.getnframes()
        if not 0 < frames <= 16000 * 30:
            raise ValueError(f"{path} must contain between 0 and 30 seconds of audio")
        return np.frombuffer(stream.readframes(frames), dtype="<i2").astype("float32") / 32768.0


def load_reference(checkpoint: Path):
    # Load large Whisper checkpoints exactly once for an entire corpus.
    # Reconstructing the Transformers pipeline for every WAV can exhaust
    # memory and masks warm-inference latency.
    from transformers import pipeline
    return pipeline("automatic-speech-recognition", model=str(checkpoint),
                    device="cpu", trust_remote_code=False)


def reference_text(recognizer, audio) -> str:
    result = recognizer({"array": audio, "sampling_rate": 16000},
                        generate_kwargs={"language": "turkish", "task": "transcribe"})
    if not isinstance(result, dict) or not isinstance(result.get("text"), str):
        raise ValueError("Unexpected Transformers reference output")
    return result["text"].strip()


def native_text(executable: Path, gguf: Path, wav: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="hex-gguf-parity-") as temp:
        output = Path(temp) / "transcript.txt"
        command = [str(executable), "-m", str(gguf), "-l", "tr", "-q",
                   "--output", str(output), str(wav)]
        result = subprocess.run(command, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=480,
                                check=False)
        if result.returncode != 0 or not output.is_file():
            raise RuntimeError("Native GGUF transcription failed: " + result.stderr[-1500:])
        return output.read_text(encoding="utf-8").strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[tuple[Path, str]]:
    """Read reviewed local JSONL {"wav":"relative.wav","text":"..."} rows."""
    base = path.resolve().parent
    clips = []
    seen = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid manifest JSON on line {lineno}: {error}") from error
        if not isinstance(record, dict) or not isinstance(record.get("wav"), str) \
                or not isinstance(record.get("text"), str) or not normalize(record["text"]):
            raise ValueError(f"Manifest line {lineno} requires wav and nonempty text")
        candidate = Path(record["wav"])
        wav = (candidate if candidate.is_absolute() else base / candidate).resolve()
        if wav in seen:
            raise ValueError(f"Duplicate WAV in manifest line {lineno}: {wav}")
        if not wav.is_file():
            raise FileNotFoundError(f"Missing manifest WAV on line {lineno}: {wav}")
        clips.append((wav, record["text"]))
        seen.add(wav)
    if not clips:
        raise ValueError("The evaluation manifest must contain at least one clip")
    return clips


def corpus_accuracy(rows: list[dict]) -> dict:
    """Micro-averaged accuracy weighted by Turkish reference token counts."""
    totals = {"words": 0, "characters": 0}
    for engine in ("reference", "native"):
        totals[f"{engine}_word_errors"] = 0
        totals[f"{engine}_character_errors"] = 0
    for row in rows:
        truth = row["ground_truth"]
        for characters, unit in ((False, "word"), (True, "character")):
            key = "characters" if characters else "words"
            for engine in ("reference", "native"):
                edits, count = error_counts(truth, row[engine], characters=characters)
                totals[f"{engine}_{unit}_errors"] += edits
                if engine == "reference":
                    totals[key] += count
    if totals["words"] == 0 or totals["characters"] == 0:
        raise ValueError("Corpus contains no reference speech")
    for engine in ("reference", "native"):
        totals[f"{engine}_wer"] = totals[f"{engine}_word_errors"] / totals["words"]
        totals[f"{engine}_cer"] = totals[f"{engine}_character_errors"] / totals["characters"]
    totals["clips"] = len(rows)
    return totals


def run(args: argparse.Namespace) -> None:
    gguf, checkpoint, binary = args.gguf.resolve(), args.checkpoint.resolve(), args.native_cli.resolve()
    if not gguf.is_file() or not (checkpoint / "model.safetensors").is_file() or not binary.is_file():
        raise FileNotFoundError("Expected GGUF, staged safetensors checkpoint, and native transcribe-cli")
    if args.max_native_wer is not None and not 0.0 <= args.max_native_wer <= 1.0:
        raise ValueError("Expected max native WER between 0 and 1")
    if args.max_native_wer is not None and args.manifest is None:
        raise ValueError("Native WER threshold requires a ground-truth manifest")
    clips = load_manifest(args.manifest) if args.manifest else [
        (path.resolve(), None) for path in args.wav
    ]
    if not clips:
        raise ValueError("At least one locally reviewed WAV is required")
    recognizer = load_reference(checkpoint)
    results = []
    for wav, truth in clips:
        samples = read_audio(wav)
        started = time.perf_counter()
        source = reference_text(recognizer, samples)
        source_time = time.perf_counter() - started
        started = time.perf_counter()
        target = native_text(binary, gguf, wav)
        target_time = time.perf_counter() - started
        comparison = compare(source, target, args.max_parity_wer, expected=truth)
        comparison.update({"wav": str(wav), "reference_seconds": source_time,
                           "native_seconds": target_time})
        results.append(comparison)
    aggregate = corpus_accuracy(results) if args.manifest else None
    passed = all(row["passed"] for row in results)
    if args.max_native_wer is not None:
        passed = passed and aggregate["native_wer"] <= args.max_native_wer
    metrics = {"model": args.model, "runtime": "transcribe.cpp GGUF",
               "checkpoint_sha256": sha256(checkpoint / "model.safetensors"),
               "gguf_sha256": sha256(gguf), "clips": results,
               "aggregate": aggregate, "max_native_wer": args.max_native_wer,
               "passed": passed,
               "note": "One short corpus is not production accuracy validation."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if not metrics["passed"]:
        raise SystemExit("GGUF reference parity failed; do not import this model")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--native-cli", type=Path, required=True)
    audio = parser.add_mutually_exclusive_group(required=True)
    audio.add_argument("--wav", type=Path, nargs="+")
    audio.add_argument("--manifest", type=Path,
                       help="JSONL with local WAV paths and ground-truth Turkish text")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-parity-wer", type=float, default=0.35)
    parser.add_argument("--max-native-wer", type=float, default=None,
                        help="Optional aggregate corpus WER limit, requires --manifest")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
