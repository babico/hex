"""Real speech smoke benchmark; measures cold load and two warm decodes.
Only public data-only models; the Mihu community license requires manual acceptance.
Fixture: google/fleurs Turkish dev, CC BY 4.0, via atasoglu/seda.
"""
import argparse
import hashlib
import json
import pathlib
import queue
import re
import sys
import subprocess
import tempfile
import threading
import time
import urllib.request
import wave

F = "https://raw.githubusercontent.com/atasoglu/seda/67fdbfc5e98919dbaecb8c7b63622b6d453c1c93/tests/data/speech.wav"
REF = "bazı festivaller küçük çocuklu aileler için özel kamp alanlarına sahiptir"
SEDA = ["encoder.int8.onnx", "decoder.onnx", "joiner.int8.onnx", "tokens.txt"]
MODELS = {
    "seda-v2": ("atasoglu/seda-v0.2", "main", SEDA),
    "seda-tiny": ("atasoglu/seda-v0.2-tiny", "main", SEDA),
    "ufakzeka-kulak": ("ufakai/ufakzeka-kulak", "886194bb06f71460169cd741e1fd8ed515fd9d1c", ["sherpa-onnx-tdt/"+n for n in ["encoder.int8.onnx","decoder.int8.onnx","joiner.int8.onnx","tokens.txt"]]),
}

def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent":"hex-turkish-stt-ci"}), timeout=120)

def download(model, directory):
    repo, rev, names = MODELS[model]
    with fetch(f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true") as r:
        meta = json.load(r)
    sha = meta["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}",sha) or (rev != "main" and sha != rev):
        raise ValueError("Invalid revision")
    entries = {s["rfilename"]:s for s in meta["siblings"]}
    for name in names:
        entry = entries[name]
        size = entry["size"]
        expected = (entry.get("lfs") or {}).get("sha256")
        if not 0 < size < 2_000_000_000 or (name.endswith(".onnx") and not expected):
            raise ValueError("Missing model integrity metadata")
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        total = 0
        with fetch(f"https://huggingface.co/{repo}/resolve/{sha}/{name}") as r, path.open("wb") as out:
            while chunk := r.read(1024*1024):
                total += len(chunk)
                if total > size:
                    raise ValueError("Size overflow")
                digest.update(chunk)
                out.write(chunk)
        if total != size or (expected and expected != digest.hexdigest()):
            raise ValueError("Model integrity error: "+name)
    return sha

def words(t):
    return re.findall(r"\w+", t.translate(str.maketrans("Iİ","ıi")).lower())

def wer(ref, text):
    a, b = words(ref), words(text)
    prev = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        curr = [i]
        for j, y in enumerate(b, 1):
            curr.append(min(curr[-1]+1, prev[j]+1, prev[j-1]+(x!=y)))
        prev = curr
    return prev[-1]/len(a)

def benchmark(args):
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        with fetch(F) as r:
            audio = r.read(2_000_000)
        if hashlib.sha1(f"blob {len(audio)}\0".encode()+audio).hexdigest() != "47ae56f2500f69d6d0e56d6e1b0f87c6967246fe":
            raise ValueError("Fixture integrity error")
        wav = root/"speech.wav"
        wav.write_bytes(audio)
        with wave.open(str(wav)) as w:
            assert (w.getnchannels(),w.getframerate(),w.getsampwidth()) == (1,16000,2)
            duration = w.getnframes()/w.getframerate()
        revision = download(args.model,root/"weights")
        start = time.perf_counter()
        proc = subprocess.Popen([str(args.worker.resolve()),"--weights",str(root/"weights"),"--model",args.model,"--language","tr","--serve","1"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,encoding="utf-8",bufsize=1)
        q = queue.Queue()
        def reader():
            for line in proc.stdout: q.put(line)
            q.put(None)
        threading.Thread(target=reader,daemon=True).start()
        def recv():
            line = q.get(timeout=120)
            if line is None: raise RuntimeError("Worker exited")
            return json.loads(line)
        try:
            assert recv().get("ready") is True
            cold = time.perf_counter()-start
            results = []
            for _ in range(2):
                start = time.perf_counter()
                proc.stdin.write(str(wav)+"\n")
                proc.stdin.flush()
                text = recv()["text"]
                elapsed = time.perf_counter()-start
                results.append({"text":text,"wer":wer(REF,text),"seconds":elapsed,"rtf":elapsed/duration})
            if args.model in ("seda-v2", "seda-tiny"):
                # True incremental Sherpa online decode: initial partials
                # come from PCM16 chunks, not retranscribing whole WAVs.
                import array
                import sys
                proc.stdin.write("_HEX_BEGIN_\n")
                proc.stdin.flush()
                assert recv()["text"] == ""
                partials = []
                with wave.open(str(wav)) as recording:
                    while chunk := recording.readframes(4096):
                        pcm = array.array("h")
                        pcm.frombytes(chunk)
                        # Preserve little-endian input when testing on any OS.
                        if sys.byteorder != "little":
                            pcm.byteswap()
                        payload = b"".join(int(x).to_bytes(2, "little", signed=True) for x in pcm)
                        proc.stdin.write("_HEX_PCM_" + payload.hex() + "\n")
                        proc.stdin.flush()
                        partials.append(recv()["text"])
                proc.stdin.write("_HEX_END_\n")
                proc.stdin.flush()
                final = recv()["text"]
                metrics_partial = {"partial_events": len(partials), "nonempty_partials": sum(bool(s.strip()) for s in partials), "final": final, "wer": wer(REF, final)}
                if not final.strip() or wer(REF, final) > .8:
                    raise RuntimeError("Incremental streaming decoder failed")
            else:
                metrics_partial = None
            proc.stdin.close()
            assert proc.wait(timeout=20)==0
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        metrics = {"model":args.model,"revision":revision,"reference":REF,"fixture":F,"attribution":"Google FLEURS, Turkish dev, CC BY 4.0","audio_seconds":duration,"cold_load_seconds":cold,"decodes":results,"streaming":metrics_partial}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps(metrics,ensure_ascii=False,indent=2))
        if any(not words(x["text"]) or x["wer"] > .8 for x in results) or words(results[0]["text"]) != words(results[1]["text"]):
            raise RuntimeError("Real speech or repeated decode smoke test failed")

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p=argparse.ArgumentParser()
    p.add_argument("--worker",type=pathlib.Path,required=True)
    p.add_argument("--model",choices=MODELS,required=True)
    p.add_argument("--output",type=pathlib.Path,required=True)
    benchmark(p.parse_args())
