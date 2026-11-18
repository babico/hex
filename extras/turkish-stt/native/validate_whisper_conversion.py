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
    folded = unicodedata.normalize("NFKC", text).casefold().replace("\u0307", "")
    return " ".join(re.findall(r"\w+", folded))


def error_rate(reference: str, prediction: str, *, characters: bool = False) -> float:
    a = list(normalize(reference)) if characters else normalize(reference).split()
    b = list(normalize(prediction)) if characters else normalize(prediction).split()
    if not a:
        return 0.0 if not b else 1.0
    previous = list(range(len(b) + 1))
    for i, token in enumerate(a, 1):
        current = [i]
        for j, candidate in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j-1] + (candidate != token)))
        previous = current
    return previous[-1] / len(a)


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


def reference_text(checkpoint: Path, audio) -> str:
    from transformers import pipeline
    recognizer = pipeline("automatic-speech-recognition", model=str(checkpoint),
                          device="cpu", trust_remote_code=False)
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


def run(args: argparse.Namespace) -> None:
    gguf, checkpoint, binary = args.gguf.resolve(), args.checkpoint.resolve(), args.native_cli.resolve()
    if not gguf.is_file() or not (checkpoint / "model.safetensors").is_file() or not binary.is_file():
        raise FileNotFoundError("Expected GGUF, staged safetensors checkpoint, and native transcribe-cli")
    results = []
    for clip in args.wav:
        wav = clip.resolve()
        samples = read_audio(wav)
        started = time.perf_counter()
        source = reference_text(checkpoint, samples)
        source_time = time.perf_counter() - started
        started = time.perf_counter()
        target = native_text(binary, gguf, wav)
        target_time = time.perf_counter() - started
        comparison = compare(source, target, args.max_parity_wer)
        comparison.update({"wav": str(wav), "reference_seconds": source_time,
                           "native_seconds": target_time})
        results.append(comparison)
    if not results:
        raise ValueError("At least one locally reviewed WAV is required")
    metrics = {"model": args.model, "runtime": "transcribe.cpp GGUF",
               "checkpoint_sha256": sha256(checkpoint / "model.safetensors"),
               "gguf_sha256": sha256(gguf), "clips": results,
               "passed": all(row["passed"] for row in results),
               "note": "Single or small-corpus parity is not production WER validation."}
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
    parser.add_argument("--wav", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-parity-wer", type=float, default=0.35)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
