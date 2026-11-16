"""Build a portable app archive with native backend next to the app."""
import argparse
import pathlib
import shutil
import tempfile
import zipfile

def one(folder, predicate):
    matches = [p for p in folder.rglob("*") if p.is_file() and predicate(p)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one executable, got {matches}")
    return matches[0]

p=argparse.ArgumentParser()
p.add_argument("--target",required=True,choices=["linux-x64","linux-arm64","windows-x64","windows-arm64"])
p.add_argument("--desktop",type=pathlib.Path,required=True)
p.add_argument("--native",type=pathlib.Path,required=True)
p.add_argument("--output",type=pathlib.Path,required=True)
a=p.parse_args()
windows=a.target.startswith("windows")
arch=a.target.split("-")[-1]
with tempfile.TemporaryDirectory() as tmp:
    root=pathlib.Path(tmp)/"hex-turkish-stt"
    root.mkdir()
    if windows:
        for ext in (".exe",".com"):
            src=one(a.desktop,lambda f:f.name.startswith("hex-") and not f.name.startswith("hex-setup-") and f.name.endswith("-windows-"+arch+ext))
            shutil.copy2(src,root/("hex"+ext))
    else:
        src=one(a.desktop,lambda f:f.name.startswith("hex-") and f.name.endswith("-linux-"+arch))
        shutil.copy2(src,root/"hex")
        (root/"hex").chmod(0o755)
    native_worker=one(a.native,lambda f:f.name==("hex-sherpa-worker.exe" if windows else "hex-sherpa-worker"))
    shutil.copytree(native_worker.parent,root/"turkish-stt",ignore=shutil.ignore_patterns("SHA256SUMS"),symlinks=False)
    (root/"turkish-stt"/native_worker.name).chmod(0o755)
    (root/"README.txt").write_text("Experimental unsigned HEX Turkish STT build. Extract the entire archive before running. Keep turkish-stt beside hex. Model weights download separately; Mihu requires acceptance of its license. No model weights are redistributed. Linux: chmod +x hex turkish-stt/hex-sherpa-worker if your unzip tool strips permissions.\n",encoding="utf-8")
    a.output.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.output/f"hex-turkish-stt-{a.target}.zip","w",compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for path in sorted(root.rglob("*")):
            if path.is_file(): z.write(path,path.relative_to(root.parent))
