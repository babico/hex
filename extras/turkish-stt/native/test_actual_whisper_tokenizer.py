"""Check real HF tokenizer files without downloading weights or remote code."""
from pathlib import Path
import json
import tempfile
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from prepare_whisper_checkpoint import generate_fast_tokenizer

REPOS = (
    "turkmedstt/whisper-large-v3-turkish-general",
    "RsGoksel/turkspeak-lv3-ITU_Mainframe",
)

def main():
    with tempfile.TemporaryDirectory() as temporary:
        for index, repo in enumerate(REPOS):
            source = Path(snapshot_download(
                repo_id=repo, allow_patterns=("*.json", "*.txt"),
                local_dir=Path(temporary) / f"source-{index}"
            ))
            staged = Path(temporary) / f"generated-{index}"
            staged.mkdir()
            for entry in source.iterdir():
                if entry.is_file():
                    (staged / entry.name).write_bytes(entry.read_bytes())
            generated = not (staged / "tokenizer.json").exists()
            if generated:
                generate_fast_tokenizer(staged)
            loaded = AutoTokenizer.from_pretrained(
                str(staged), local_files_only=True, use_fast=True, trust_remote_code=False
            )
            assert loaded.is_fast
            tokenizer_json = json.loads((staged / "tokenizer.json").read_text())
            assert tokenizer_json["model"]["type"] == "BPE"
            example = "Bugün Türkçe konuşma tanıma testi yapıyoruz."
            assert len(loaded.encode(example)) > 4
            print(json.dumps({
                "repository": repo, "generated": generated,
                "tokens": len(loaded.encode(example)),
            }))

if __name__ == "__main__":
    main()
