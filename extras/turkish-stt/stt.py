#!/usr/bin/env python3
"""Local Turkish STT companion for Hex. Vendor Python engines run in a persistent worker."""
import argparse
import json
import os
from pathlib import Path
import sys
import wave

REGISTRY = os.environ.get("HEX_TURKISH_STT_REGISTRY")
if REGISTRY is None:
    REGISTRY = Path(__file__).with_name("models.json").read_text(encoding="utf-8")
MODELS = {m["id"]: m for m in json.loads(REGISTRY)["models"]}

def samples(path):
    import numpy as np
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
            raise ValueError("Expected mono 16 kHz PCM16 WAV")
        if wav.getnframes() > 16_000 * 600:
            raise ValueError("Audio longer than ten minutes")
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype("float32") / 32768.0

def snapshot(model, offline, **kwargs):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=model["repository"], local_files_only=offline, **kwargs)

def load_runner(model, offline=False):
    """Return one reusable transcribe(path)->str callback. Load weights only once."""
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    kind = model["adapter"]
    if kind == "balbal":
        import balbal
        name = "balbal-v2-manas" if model.get("variant") == "manas" else "balbal-v2"
        recognizer = balbal.load(name)
        return lambda path: recognizer.transcribe(str(path))["text"]
    if kind == "mihu":
        import mihu_stt
        # The vendor API maintains its own model cache; do not spawn a process
        # for every clip, even if it internally creates a new decoder.
        return lambda path: mihu_stt.transcribe(str(path))
    if kind == "seda":
        from sedalib import Recognizer, read_wav
        recognizer = Recognizer.from_pretrained(model["repository"], mode=model["variant"], punctuate=True)
        def transcribe(path):
            audio, rate = read_wav(str(path))
            return recognizer.transcribe(audio, rate)
        return transcribe
    if kind == "duyu":
        from duyu_small import DuyuSmall
        directory = os.environ.get("DUYU_SMALL_DIR")
        if not directory:
            raise RuntimeError("Set DUYU_SMALL_DIR to the reviewed, locally installed Duyu-small checkout")
        recognizer = DuyuSmall(directory)
        return lambda path: recognizer.transcribe(samples(path), lang="mix")
    if kind == "kulak":
        import sherpa_onnx
        root = Path(snapshot(model, offline, allow_patterns=["sherpa-onnx-tdt/*"])) / "sherpa-onnx-tdt"
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(root / "encoder.int8.onnx"), decoder=str(root / "decoder.int8.onnx"),
            joiner=str(root / "joiner.int8.onnx"), tokens=str(root / "tokens.txt"),
            sample_rate=16000, feature_dim=80, decoding_method="greedy_search", model_type="nemo_transducer",
        )
        def transcribe(path):
            stream = recognizer.create_stream()
            stream.accept_waveform(16000, samples(path) * 8)
            recognizer.decode_stream(stream)
            return stream.result.text
        return transcribe
    if kind == "whisper":
        import torch
        from transformers import pipeline
        device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        directory = snapshot(model, offline, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])
        recognizer = pipeline("automatic-speech-recognition", model=directory, device=device, trust_remote_code=False)
        def transcribe(path):
            result = recognizer({"array": samples(path), "sampling_rate": 16000}, chunk_length_s=30,
                generate_kwargs={"language": "turkish", "task": "transcribe"})
            return result["text"]
        return transcribe
    raise ValueError(f"Unknown adapter: {kind}")

def recognize(model, path, offline=False):
    return load_runner(model, offline)(path)

def serve(model, offline=False, input_stream=None, output_stream=None):
    """One audio path per line -> one JSON result. Starts only after weights load."""
    input_stream = input_stream if input_stream is not None else sys.stdin
    output_stream = output_stream if output_stream is not None else sys.stdout
    runner = load_runner(model, offline)
    output_stream.write('{"ready":true}\n')
    output_stream.flush()
    for line in input_stream:
        path = line.rstrip("\r\n")
        if not path or len(path) > 32768:
            raise ValueError("Invalid local audio path")
        try:
            result = runner(Path(path))
            if not isinstance(result, str):
                raise TypeError("Adapter did not return text")
            response = {"text": result.strip()}
        except Exception as error:
            # A failing inference is recoverable on the next recording. The
            # host resets its worker on errors, so no stale decode is reused.
            response = {"error": str(error)[:1024]}
        output_stream.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
        output_stream.flush()
    return 0

def main(argv=None):
    if hasattr(sys.stdout, "reconfigure") and sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    install = sub.add_parser("install", help="Install the adapter's Python dependencies in this environment")
    install.add_argument("model", choices=MODELS)
    download = sub.add_parser("download", help="Explicitly download/cache this model's weights")
    download.add_argument("model", choices=MODELS)
    run = sub.add_parser("transcribe")
    run.add_argument("model", choices=MODELS)
    run.add_argument("wav", type=Path)
    run.add_argument("--offline", action="store_true")
    server = sub.add_parser("serve", help="Keep an engine loaded across clips; use from Hex")
    server.add_argument("model", choices=MODELS)
    server.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "list":
        print(json.dumps(list(MODELS.values()), ensure_ascii=False, indent=2))
        return 0
    model = MODELS[args.model]
    if args.command == "install":
        import subprocess
        return subprocess.call([sys.executable, "-m", "pip", "install", *model["dependencies"]])
    try:
        if args.command == "download":
            # Explicitly called by the user in a trusted Python environment.
            load_runner(model, offline=False)
            print(json.dumps({"model": args.model, "cached": True}, ensure_ascii=False))
            return 0
        if args.command == "serve":
            return serve(model, offline=args.offline)
        if not args.wav.is_file():
            parser.error("WAV file does not exist")
        result = recognize(model, args.wav.resolve(), args.offline)
        if not isinstance(result, str):
            raise TypeError("Adapter did not return text")
        print(json.dumps({"model": args.model, "text": result.strip()}, ensure_ascii=False))
        return 0
    except Exception as error:
        print(f"{args.model}: {error}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
