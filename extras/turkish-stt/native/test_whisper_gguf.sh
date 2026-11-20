#!/usr/bin/env bash
# Verify actual native Whisper->GGUF conversion, Q8_0 and Turkish inference.
set -euo pipefail
base="$RUNNER_TEMP/hex-whisper-gguf"
mkdir -p "$base" metrics
git clone --no-checkout https://github.com/handy-computer/transcribe.cpp.git "$base/transcribe"
git -C "$base/transcribe" checkout --detach adfc50210a3f026a655a3e19937d90ef2bf0728c
cd "$base/transcribe"
uv run --project scripts/envs/whisper scripts/convert-whisper.py \
    openai/whisper-tiny "$base/whisper-tiny-reference.gguf" \
    --repo-id openai/whisper-tiny --variant whisper-tiny
test -s "$base/whisper-tiny-reference.gguf"
cmake -S . -B "$base/build" -DCMAKE_BUILD_TYPE=Release \
    -DTRANSCRIBE_BUILD_TESTS=OFF -DTRANSCRIBE_BUILD_EXAMPLES=ON \
    -DTRANSCRIBE_BUILD_TOOLS=ON
cmake --build "$base/build" --config Release --target transcribe-cli transcribe-quantize --parallel 4
quantizer=$(find "$base/build" -type f -name transcribe-quantize -print -quit)
cli=$(find "$base/build" -type f -name transcribe-cli -print -quit)
test -x "$quantizer" && test -x "$cli"
"$quantizer" "$base/whisper-tiny-reference.gguf" "$base/whisper-tiny-Q8_0.gguf" --quant Q8_0
test -s "$base/whisper-tiny-Q8_0.gguf"
python3 - "$base" <<'PY'
import hashlib, pathlib, sys, urllib.request, wave
base = pathlib.Path(sys.argv[1])
url = "https://raw.githubusercontent.com/atasoglu/seda/67fdbfc5e98919dbaecb8c7b63622b6d453c1c93/tests/data/speech.wav"
with urllib.request.urlopen(url, timeout=120) as response:
    audio = response.read(2_000_000)
actual = hashlib.sha1(f"blob {len(audio)}\0".encode() + audio).hexdigest()
assert actual == "47ae56f2500f69d6d0e56d6e1b0f87c6967246fe", actual
(base / "speech.wav").write_bytes(audio)
with wave.open(str(base / "speech.wav"), "rb") as clip:
    assert (clip.getnchannels(), clip.getframerate(), clip.getsampwidth()) == (1, 16000, 2)
PY
"$cli" -m "$base/whisper-tiny-reference.gguf" -l tr -q "$base/speech.wav" > "$base/reference.txt"
"$cli" -m "$base/whisper-tiny-Q8_0.gguf" -l tr -q "$base/speech.wav" > "$base/quantized.txt"
cd "$GITHUB_WORKSPACE"
python3 - "$base" "$GITHUB_SHA" <<'PY'
import hashlib, json, pathlib, sys
base = pathlib.Path(sys.argv[1])
def details(which):
    path = base / f"whisper-tiny-{which}.gguf"
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return {"size": path.stat().st_size, "sha256": h.hexdigest()}
reference = (base / "reference.txt").read_text(encoding="utf-8").strip()
quantized = (base / "quantized.txt").read_text(encoding="utf-8").strip()
assert reference and quantized, (reference, quantized)
report = {
    "fixture": "Google FLEURS Turkish via atasoglu/seda, CC BY 4.0",
    "source": "openai/whisper-tiny",
    "converter": "handy-computer/transcribe.cpp@adfc50210a3f026a655a3e19937d90ef2bf0728c",
    "head": sys.argv[2], "reference_text": reference, "quantized_text": quantized,
    "reference": details("reference"), "quantized": details("Q8_0"),
}
pathlib.Path("metrics").mkdir(exist_ok=True)
pathlib.Path("metrics/whisper-gguf.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
PY
