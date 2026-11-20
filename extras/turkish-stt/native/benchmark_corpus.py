#!/usr/bin/env python3
"""Evaluate locally reviewed Turkish WAVs against a resident native Sherpa model.

No implicit downloads, no model license acceptance, and no network access.
Manifest format is one JSON object per line:
{"wav":"relative/to/manifest.wav","text":"ground-truth Turkish transcript"}
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import threading
import time
import wave

from validate_whisper_conversion import error_counts, load_manifest


MODELS = ("seda-v2", "seda-tiny", "ufakzeka-kulak", "mihu-v2")
MAX_SAMPLES = 16000 * 600


class ResidentWorker:
    """One native decoder for all clips, bounded by per-clip deadlines."""

    def __init__(self, executable: Path, model: str, weights: Path):
        self.process = subprocess.Popen(
            [str(executable.resolve()), "--model", model,
             "--weights", str(weights.resolve()), "--language", "tr", "--serve", "1"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, bufsize=0,
        )
        self.input = queue.Queue(maxsize=1)
        self.output = queue.Queue(maxsize=1)

        def write():
            try:
                while True:
                    command = self.input.get()
                    if command is None:
                        break
                    self.process.stdin.write(command + b"\n")
                    self.process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass

        def read():
            while True:
                frame = self.process.stdout.readline(4 * 1024 * 1024 + 1)
                if not frame or len(frame) > 4 * 1024 * 1024:
                    self.output.put(None)
                    break
                self.output.put(frame)

        threading.Thread(target=write, daemon=True).start()
        threading.Thread(target=read, daemon=True).start()
        try:
            ready = self.receive(time.monotonic() + 90)
            if ready.get("ready") is not True or ready.get("pcm_protocol") != 1:
                raise RuntimeError(f"Native STT model failed to load: {ready.get('error', ready)}")
        except BaseException:
            self.close()
            raise

    def receive(self, deadline: float) -> dict:
        try:
            raw = self.output.get(timeout=max(0.001, deadline - time.monotonic()))
        except queue.Empty as exc:
            raise TimeoutError("Native model did not respond before deadline") from exc
        if raw is None:
            raise RuntimeError("Native model exited without a complete protocol frame")
        frame = json.loads(raw.decode("utf-8"))
        if not isinstance(frame, dict):
            raise ValueError("Invalid native model protocol frame")
        if "error" in frame:
            raise RuntimeError("Native decoder failed: " + str(frame["error"])[:512])
        return frame

    def send(self, command: bytes, deadline: float) -> dict:
        if time.monotonic() >= deadline:
            raise TimeoutError("Native decoding deadline exceeded")
        try:
            self.input.put_nowait(command)
        except queue.Full as exc:
            raise TimeoutError("Native worker input pipe is blocked") from exc
        return self.receive(deadline)

    def transcribe(self, path: Path, sample_count: int) -> str:
        duration = sample_count / 16000
        deadline = time.monotonic() + min(1500, 180 + 2 * duration)
        if self.send(b"_HEX_CLIP_BEGIN_", deadline).get("text") != "":
            raise RuntimeError("Native worker rejected PCM start")
        with wave.open(str(path), "rb") as audio:
            while payload := audio.readframes(32768):
                frame = self.send(b"_HEX_CLIP_PCM_" + payload.hex().encode("ascii"), deadline)
                if frame.get("text") != "":
                    raise RuntimeError("Unexpected native PCM transfer response")
        response = self.send(b"_HEX_CLIP_END_", deadline)
        result = response.get("text")
        if not isinstance(result, str):
            raise RuntimeError("Native decoder did not return text")
        return result.strip()

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def validated_audio(path: Path) -> int:
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getframerate(), audio.getsampwidth(),
                audio.getcomptype()) != (1, 16000, 2, "NONE"):
            raise ValueError(f"Expected mono 16 kHz PCM16 WAV: {path}")
        count = audio.getnframes()
        if not 0 < count <= MAX_SAMPLES:
            raise ValueError(f"Audio must contain 1 to {MAX_SAMPLES} samples: {path}")
        return count


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Evaluation contains no clips")
    totals = {"clips": len(rows), "reference_words": 0,
              "reference_characters": 0, "word_errors": 0,
              "character_errors": 0, "audio_seconds": 0.0, "decode_seconds": 0.0}
    for row in rows:
        words, word_count = error_counts(row["ground_truth"], row["text"])
        characters, char_count = error_counts(row["ground_truth"], row["text"], characters=True)
        totals["reference_words"] += word_count
        totals["reference_characters"] += char_count
        totals["word_errors"] += words
        totals["character_errors"] += characters
        totals["audio_seconds"] += row["audio_seconds"]
        totals["decode_seconds"] += row["decode_seconds"]
    if not totals["reference_words"] or not totals["reference_characters"]:
        raise ValueError("Ground-truth corpus must contain Turkish text")
    totals["wer"] = totals["word_errors"] / totals["reference_words"]
    totals["cer"] = totals["character_errors"] / totals["reference_characters"]
    totals["rtf"] = totals["decode_seconds"] / totals["audio_seconds"]
    return totals


def evaluate(args: argparse.Namespace) -> dict:
    if args.model not in MODELS:
        raise ValueError("Unsupported native Sherpa model")
    for name in ("max_wer", "max_cer"):
        value = getattr(args, name)
        if value is not None and not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    clips = load_manifest(args.manifest)
    worker_path = args.worker.resolve(strict=True)
    weights = args.weights.resolve(strict=True)
    if not weights.is_dir():
        raise ValueError("Weights must be a local directory")
    results = []
    with ResidentWorker(worker_path, args.model, weights) as worker:
        for wav, ground_truth in clips:
            count = validated_audio(wav)
            start = time.perf_counter()
            text = worker.transcribe(wav, count)
            seconds = time.perf_counter() - start
            if not text:
                raise RuntimeError(f"Native decoder produced no text for {wav}")
            word_errors, words = error_counts(ground_truth, text)
            character_errors, characters = error_counts(
                ground_truth, text, characters=True
            )
            results.append({
                "wav": str(wav), "ground_truth": ground_truth, "text": text,
                "wer": word_errors / words, "cer": character_errors / characters,
                "audio_seconds": count / 16000, "decode_seconds": seconds,
                "rtf": seconds / (count / 16000),
            })
    aggregate = summarize(results)
    passed = ((args.max_wer is None or aggregate["wer"] <= args.max_wer)
              and (args.max_cer is None or aggregate["cer"] <= args.max_cer))
    report = {
        "model": args.model, "runtime": "Sherpa-ONNX resident worker",
        "worker_sha256": hashlib.sha256(worker_path.read_bytes()).hexdigest(),
        "manifest": str(args.manifest.resolve()), "clips": results,
        "aggregate": aggregate, "max_wer": args.max_wer,
        "max_cer": args.max_cer, "passed": passed,
        "note": "Report covers only locally reviewed audio; never infer broader accuracy.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    if not passed:
        raise SystemExit(f"Native corpus accuracy below required threshold: {args.output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-wer", type=float)
    parser.add_argument("--max-cer", type=float)
    report = evaluate(parser.parse_args())
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
