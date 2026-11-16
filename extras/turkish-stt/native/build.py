"""Download reviewed SDK assets by SHA-256 and build the local worker."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tarfile
import tempfile
from urllib.request import urlretrieve

ASSETS = {
    "linux-x64": ("linux-x64-shared-no-tts", "d0f96c8b65c6cd0974fada22737e337de81bc8cd2abbec2e39caf358b1eec5fc"),
    "linux-arm64": ("linux-aarch64-shared-cpu", "4e3734f82bc1379fd91f219f5869c7e9d03b7a4f7561907d8abca4849c51a789"),
    "macos-arm64": ("osx-arm64-shared-no-tts", "91b96512c4fa1960f8a9ed5360a6c8dda53a4b5015d0590244f14086a234557a"),
    "windows-x64": ("win-x64-shared-MD-Release-no-tts", "876e6b89b8cf84a3a1b375a397507f2cfe9c227c2a945411a11a668475fcb5d3"),
    "windows-arm64": ("win-arm64-shared-MD-Release-no-tts", "39446be8a2a933308bb0b4b7a77fd21fdffaa68d3d73fc6f1e63634bb42d9cca"),
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
        # Some official distributions omit headers. Always compile against
        # the vendored, version-matched C API instead of a C++ ABI.
        sdk_dirs = [p for p in (root / "sdk").iterdir() if p.is_dir() and (p / "lib").is_dir()]
        if len(sdk_dirs) != 1:
            raise RuntimeError("Expected one SDK directory with libraries")
        sdk = sdk_dirs[0]
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
