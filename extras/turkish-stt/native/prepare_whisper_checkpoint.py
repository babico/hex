"""Stage offline Whisper checkpoint without copying 3 GB of weights.

A missing tokenizer.json is built locally from the author's vocab and merges.
The original model directory is never modified.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import struct

REQUIRED = (
    "config.json", "generation_config.json", "preprocessor_config.json",
    "model.safetensors", "vocab.json", "merges.txt",
)

def validate_safetensors(path: Path) -> None:
    with path.open("rb") as stream:
        header_len = stream.read(8)
        if len(header_len) != 8:
            raise ValueError("model.safetensors is empty or is an LFS pointer")
        length = struct.unpack("<Q", header_len)[0]
        if not 0 < length <= 16 * 1024 * 1024 or length + 8 >= path.stat().st_size:
            raise ValueError("model.safetensors header is invalid or weights have not been downloaded")
        header = json.loads(stream.read(length))
    keys = [key for key in header if key != "__metadata__"]
    if not any(key.startswith("model.encoder.") for key in keys):
        raise ValueError("Checkpoint lacks Whisper encoder tensors (is this an unmerged LoRA?)")
    if not any(key.startswith("model.decoder.") for key in keys):
        raise ValueError("Checkpoint lacks Whisper decoder tensors (is this an unmerged LoRA?)")

def generate_fast_tokenizer(staged: Path) -> None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(staged), local_files_only=True, use_fast=True, trust_remote_code=False
    )
    if not tokenizer.is_fast:
        raise ValueError("Cannot build a fast Whisper tokenizer from the local checkpoint")
    tokenizer.save_pretrained(str(staged))

def stage(source: Path, target: Path, tokenizer_builder=None) -> bool:
    source = source.resolve()
    target = target.resolve()
    if target == source or source in target.parents:
        raise ValueError("Staged output must be outside the source model directory")
    for name in REQUIRED:
        if not (source / name).is_file():
            raise FileNotFoundError(f"Required Whisper checkpoint file missing: {name}")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "whisper":
        raise ValueError("Only merged Whisper checkpoints are supported")
    validate_safetensors(source / "model.safetensors")
    if target.exists() and any(target.iterdir()):
        raise ValueError("Staging directory must be empty")
    target.mkdir(parents=True, exist_ok=True)
    for entry in source.iterdir():
        if not entry.is_file():
            continue
        destination = target / entry.name
        if entry.name == "model.safetensors":
            try:
                os.link(entry, destination)  # same filesystem, no duplicate
            except OSError:
                destination.symlink_to(entry)  # cross-filesystem
        else:
            shutil.copy2(entry, destination)
    generated = not (target / "tokenizer.json").is_file()
    if generated:
        (tokenizer_builder or generate_fast_tokenizer)(target)
    tokenizer_file = target / "tokenizer.json"
    if not tokenizer_file.is_file():
        raise ValueError("Fast Whisper tokenizer generation failed")
    tokenizer_json = json.loads(tokenizer_file.read_text(encoding="utf-8"))
    if tokenizer_json.get("model", {}).get("type") != "BPE":
        raise ValueError("Expected tokenizer.json with a BPE model")
    return generated

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("staged", type=Path)
    args = parser.parse_args()
    generated = stage(args.source, args.staged)
    print(f"Whisper checkpoint staged offline; fast tokenizer generated={generated}")

if __name__ == "__main__":
    main()
