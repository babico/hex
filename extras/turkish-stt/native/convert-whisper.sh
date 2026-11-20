#!/bin/sh
# Convert a locally downloaded, license-reviewed Whisper checkpoint.
# Python is used at conversion time only; the result runs in transcribe.cpp.
set -eu
if [ "$#" -ne 3 ] && [ "$#" -ne 4 ]; then
  echo "usage: $0 MODEL_ID LOCAL_HF_CHECKPOINT OUTPUT_DIRECTORY [TURKISH_16KHZ_MONO_WAV_OR_JSONL_MANIFEST]" >&2
  exit 2
fi
case "$1" in
  turkmedstt|turkspeak-itu) ;;
  *) echo "unsupported model ID: $1" >&2; exit 2 ;;
esac
id=$1
case "$id" in
  turkmedstt) repo_id=turkmedstt/whisper-large-v3-turkish-general ;;
  turkspeak-itu) repo_id=RsGoksel/turkspeak-lv3-ITU_Mainframe ;;
esac
audio=""
audio_mode="--wav"
if [ "$#" -eq 4 ]; then
  test -f "$4" || { echo "No audio or manifest file: $4" >&2; exit 2; }
  audio=$(CDPATH= cd -- "$(dirname -- "$4")" && pwd)/$(basename -- "$4")
  case "$audio" in
    *.wav|*.WAV) audio_mode="--wav" ;;
    *.jsonl) audio_mode="--manifest" ;;
    *) echo "Expected a 16 kHz WAV or JSONL evaluation manifest: $audio" >&2; exit 2 ;;
  esac
fi
checkpoint=$(CDPATH= cd -- "$2" && pwd)
mkdir -p "$3"
output=$(CDPATH= cd -- "$3" && pwd)
test -f "$checkpoint/config.json"
# Some author checkpoints ship vocab.json+merges.txt but no fast tokenizer.json.
# Generate tokenizer.json in a separate staged folder; never mutate source weights.
test -f "$checkpoint/model.safetensors"
helpers=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
helper="$helpers/prepare_whisper_checkpoint.py"
for tool in git cmake uv; do command -v "$tool" >/dev/null; done
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT HUP INT TERM
git clone --no-checkout https://github.com/handy-computer/transcribe.cpp.git "$work/transcribe"
git -C "$work/transcribe" checkout --detach adfc50210a3f026a655a3e19937d90ef2bf0728c
cd "$work/transcribe"
uv run --project scripts/envs/whisper python "$helper" "$checkpoint" "$work/staged"
uv run --project scripts/envs/whisper scripts/convert-whisper.py "$work/staged" "$output/$id-reference.gguf" --repo-id "$repo_id" --variant whisper-large-v3
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target transcribe-quantize transcribe-cli --parallel 4
quantizer=$(find build -type f \( -name transcribe-quantize -o -name transcribe-quantize.exe \) | head -n 1)
test -n "$quantizer"
"$quantizer" "$output/$id-reference.gguf" "$output/$id-Q8_0.gguf" --quant Q8_0
echo "Converted: $output/$id-Q8_0.gguf"
shasum -a 256 "$output/$id-Q8_0.gguf" > "$output/$id-Q8_0.gguf.sha256"
if [ -n "$audio" ]; then
  cli=$(find build -type f \( -name transcribe-cli -o -name transcribe-cli.exe \) | head -n 1)
  test -n "$cli" || { echo "Native transcribe-cli is missing" >&2; exit 1; }
  # Genuine author checkpoint vs quantized native C++ inference, on the
  # exact same local WAV. The gate fails closed when outputs diverge.
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --project scripts/envs/whisper python \
    "$helpers/validate_whisper_conversion.py" \
    --model "$id" --checkpoint "$work/staged" \
    --gguf "$output/$id-Q8_0.gguf" --native-cli "$cli" \
    "$audio_mode" "$audio" --output "$output/$id-parity.json"
  echo "GGUF reference parity gate passed on $audio"
else
  echo "WARNING: GGUF is unvalidated. Rerun with a Turkish WAV or a ground-truth JSONL manifest before importing."
fi

echo "To install for native inference ONLY after passing reference parity:"
echo "  hex model import '$output/$id-Q8_0.gguf' --name '$id'"
