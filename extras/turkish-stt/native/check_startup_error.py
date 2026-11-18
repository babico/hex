#!/usr/bin/env python3
"""Check the compiled native worker reports a missing model as one JSON frame."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: check_startup_error.py PATH_TO_WORKER")
    worker = Path(sys.argv[1]).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="hex-stt-missing-") as missing:
        result = subprocess.run(
            [
                str(worker), "--model", "seda-tiny", "--language", "tr",
                "--weights", missing, "--serve", "1",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    if result.returncode == 0:
        raise AssertionError("Worker unexpectedly initialized without model weights")
    lines = result.stdout.splitlines()
    if len(lines) != 1:
        raise AssertionError(f"Expected one JSON startup frame, got {len(lines)}")
    message = json.loads(lines[0])
    if message.get("ready") is not False:
        raise AssertionError(f"Worker did not reject the missing weights: {message}")
    if "Missing model file" not in message.get("error", ""):
        raise AssertionError(f"Startup error did not identify missing model weights: {message}")
    print("Missing native model weights reported as a structured error")


if __name__ == "__main__":
    main()
