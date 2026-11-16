#!/usr/bin/env python3
"""Experimental local, whole-clip STT companion for Hex. No cloud inference."""
import argparse
import json
import os
from pathlib import Path
import subprocess
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
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype("float32") / 32768.0

def snapshot(model, offline, **kwargs):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=model["repository"], local_files_only=offline, **kwargs)

def recognize(model, path, offline=False):
    kind = model["adapter"]
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if kind == "balbal":
        # Invoke the console entry point from this interpreter's installed package.
        args = ["run", "--json"]
        if model.get("variant"):
            args += ["--model", model["variant"]]
        args.append(str(path))
        # Capture the command's stdout; leave diagnostics on stderr.
        code = "from importlib.metadata import entry_points; next(iter(entry_points(group='console_scripts', name='balbal'))).load()()"
        result = subprocess.run([sys.executable, "-c", code, *args], check=True, capture_output=True, text=True, encoding="utf-8")
        response = json.loads(result.stdout)
        return response["text"]
    if kind == "mihu":
        import mihu_stt
        return mihu_stt.transcribe(str(path))
    if kind == "seda":
        from sedalib import Recognizer, read_wav
        audio, rate = read_wav(str(path))
        recognizer = Recognizer.from_pretrained(model["repository"], mode=model["variant"], punctuate=True)
        return recognizer.transcribe(audio, rate)
    if kind == "duyu":
        # The engine is maintained by the model author. Require explicit installation;
        # never import executable Python directly from a downloaded weights snapshot.
        from duyu_small import DuyuSmall
        directory = os.environ.get("DUYU_SMALL_DIR")
        if not directory:
            raise RuntimeError("Set DUYU_SMALL_DIR to the reviewed, locally installed Duyu-small checkout")
        return DuyuSmall(directory).transcribe(samples(path), lang="mix")
    if kind == "kulak":
        import sherpa_onnx
        root = Path(snapshot(model, offline, allow_patterns=["sherpa-onnx-tdt/*"])) / "sherpa-onnx-tdt"
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(root / "encoder.int8.onnx"), decoder=str(root / "decoder.int8.onnx"),
            joiner=str(root / "joiner.int8.onnx"), tokens=str(root / "tokens.txt"),
            sample_rate=16000, feature_dim=80, decoding_method="greedy_search", model_type="nemo_transducer",
        )
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, samples(path) * 8)
        recognizer.decode_stream(stream)
        return stream.result.text
    if kind == "whisper":
        import torch
        from transformers import pipeline
        device = 0 if torch.cuda.is_available() else (-1)
        directory = snapshot(model, offline, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])
        pipe = pipeline("automatic-speech-recognition", model=directory, device=device, trust_remote_code=False)
        result = pipe({"array": samples(path), "sampling_rate": 16000}, chunk_length_s=30,
                      generate_kwargs={"language":"turkish", "task":"transcribe"})
        return result["text"]
    raise ValueError(f"Unknown adapter: {kind}")

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    install = sub.add_parser("install", help="Install selected adapter's Python dependencies")
    install.add_argument("model", choices=MODELS)
    run = sub.add_parser("transcribe")
    run.add_argument("model", choices=MODELS)
    run.add_argument("wav", type=Path)
    run.add_argument("--offline", action="store_true", help="Request cached weights only; support depends on vendor adapter")
    args = parser.parse_args(argv)
    if args.command == "list":
        print(json.dumps(list(MODELS.values()), ensure_ascii=False, indent=2))
        return 0
    model = MODELS[args.model]
    if args.command == "install":
        return subprocess.call([sys.executable, "-m", "pip", "install", *model["dependencies"]])
    if not args.wav.is_file():
        parser.error("WAV file does not exist")
    try:
        text = recognize(model, args.wav.resolve(), args.offline)
        if not isinstance(text, str):
            raise TypeError("Adapter did not return text")
        print(json.dumps({"model": args.model, "text": text.strip()}, ensure_ascii=False))
        return 0
    except Exception as error:
        print(f"{args.model}: {error}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
