"""Download reviewed SDK assets by SHA-256 and build the local worker."""
import argparse
import hashlib
from pathlib import Path
import platform
import subprocess
import tarfile
import tempfile
from urllib.request import urlretrieve

ASSETS = {
    "linux-x64": ("linux-x64-shared-no-tts-lib", "bf2d998c8b07012cd5098f3b92673bc1333fd9b927767d7cb664be8190d8bc0b"),
    "linux-arm64": ("linux-aarch64-shared-cpu-lib", "fb98b80628a383909bf83626daca2e0d0ffc93bf1f7d222eaaa4282b388f072f"),
    "macos-arm64": ("osx-arm64-shared-no-tts-lib", "f3e0cbd86cc3f38dad30c97921b40e9a8bcc6f2c943777eb76ad77176993e417"),
    "windows-x64": ("win-x64-shared-MD-Release-no-tts-lib", "b90992b888710715d613a4fedcd75d5b4db294ef64f3194e4f77ea0a8b387441"),
    "windows-arm64": ("win-arm64-shared-MD-Release-no-tts-lib", "bdd1e9a1334d2f79ce64fac4cc417d88a954f961f6055056c213c9f5321efdd4"),
}
VERSION = "v1.13.8"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=ASSETS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    suffix, digest = ASSETS[options.target]
    archive_name = f"sherpa-onnx-{VERSION}-{suffix}.tar.bz2"
    output = options.output.resolve()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        archive = root / archive_name
        urlretrieve(f"https://github.com/k2-fsa/sherpa-onnx/releases/download/{VERSION}/{archive_name}", archive)
        with archive.open("rb") as data:
            if hashlib.file_digest(data, "sha256").hexdigest() != digest:
                raise RuntimeError("Sherpa SDK integrity check failed")
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(root / "sdk", filter="data")
        headers = list((root / "sdk").rglob("cxx-api.h"))
        headers = [h for h in headers if h.parent.name == "c-api" and h.parent.parent.name == "sherpa-onnx"]
        if len(headers) != 1:
            raise RuntimeError("Could not locate sherpa-onnx SDK header")
        sdk = headers[0].parents[3]
        build = root / "build"
        source = Path(__file__).resolve().parent
        flags = []
        if options.target.startswith("windows"):
            flags = ["-A", "ARM64" if options.target.endswith("arm64") else "x64"]
        subprocess.run(["cmake", "-S", str(source), "-B", str(build),
                        "-DCMAKE_BUILD_TYPE=Release", f"-DSHERPA_ROOT={sdk}", *flags], check=True)
        subprocess.run(["cmake", "--build", str(build), "--config", "Release", "--parallel", "4"], check=True)
        subprocess.run(["cmake", "--install", str(build), "--config", "Release", "--prefix", str(output)], check=True)
        executable = output / ("hex-sherpa-worker.exe" if options.target.startswith("windows") else "hex-sherpa-worker")
        subprocess.run([str(executable), "--help"], check=True)
        # Retain any legal notices distributed with the SDK.
        notices = output / "sdk-notices"
        for path in sdk.iterdir():
            if path.is_file() and ("LICENSE" in path.name.upper() or "NOTICE" in path.name.upper()):
                notices.mkdir(exist_ok=True)
                (notices / path.name).write_bytes(path.read_bytes())

if __name__ == "__main__":
    main()
