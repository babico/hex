# Experimental Turkish STT fork

Nine models are wired into Hex's Rust catalog, model picker and whole-clip transcriber. Existing GGUF models remain unchanged and the default is still upstream Whisper. These are experimental: no claim of inference or build success until CI and actual recorded-speech tests pass.

## Python-free Sherpa backend (four models)

Mihu v2, Seda v0.2, Seda Tiny, UfakZeka Kulak use the C++ Sherpa-ONNX 1.13.8 worker. Its five architecture builds download checksum-pinned SDK assets; model weights are not redistributed. The worker loads the provided local weights, uses CPU inference, emits JSON, and does not require Python at runtime. Seda uses beam search without the vendor's LM/formatter; its output and WER are not equivalent to sedalib accurate mode. Kulak uses the model author's gain-8 TDT setting for Sherpa 1.13.8.

Build the worker (Python is only required for the SDK build helper; CMake and native C++ compiler required):

```sh
python extras/turkish-stt/native/build.py --target linux-x64 --output native-dist
```

Targets: linux-x64, linux-arm64, windows-x64, windows-arm64, macos-arm64. Keep the worker and shared SDK libraries together. Manually download each model's weights into a local folder from its repository (see models.json) after checking its license. For Mihu use encoder.int8.onnx, decoder.int8.onnx, joiner.int8.onnx and tokens.txt; for Seda use encoder.int8.onnx, decoder.onnx, joiner.int8.onnx and tokens.txt; for Kulak supply sherpa-onnx-tdt directory or its parent.

```sh
hex turkish-stt register mihu-v2 /absolute/path/native-dist/hex-sherpa-worker --weights /absolute/path/mihu
hex model list
hex model use mihu-v2
```

On Windows use the .exe and absolute paths. The native picker lists the models; after registration select the model in Settings if the app is running. Restart the app if its installed-state cache has not refreshed. Unregister with `hex turkish-stt unregister MODEL`. Registration checks the executable path, not model integrity or inference compatibility. The model picker's Download button now installs the four managed native models from Hugging Face, verifies graph SHA-256 values and records the immutable revision. Mihu first requires reviewing its community license and explicitly running `hex turkish-stt accept-license mihu-v2`. Managed registrations use one persistent worker per loaded model; manually registered executables retain the one-process-per-clip fallback. No live partials are exposed yet.

## Other models: existing vendor adapter fallback

Balbal Manas and Balbal v2 use the upstream package (custom ops + KenLM), Duyu its model-author Python engine (multi-stage ONNX + tokenizer), TurkMed and TurkSpeak Transformers until verified native ports or transcribe.cpp conversions exist. Their Python adapters are not yet validated on any platform. Python 3.11/3.12 virtual environments are recommended. Install selected dependencies and download/cache weights explicitly; check the actual file-based inference before registering:

```sh
hex turkish-stt list
hex turkish-stt install balbal-manas
hex turkish-stt transcribe balbal-manas speech.wav
hex turkish-stt register-python balbal-manas /absolute/path/to/venv/bin/python
hex model use balbal-manas
```

Set HEX_STT_PYTHON to the virtual environment interpreter for companion CLI commands. The registered interpreter is used for dictation, requesting offline mode. This is not a network sandbox; only explicitly trusted local executables/packages should be registered. Duyu requires a reviewed local engine checkout, installed with pip install -e . and DUYU_SMALL_DIR set for the app. Vendor dependencies and weights need pinned versions and real tests before production. Balbal and Mihu have custom license terms; review the applicable terms before downloading or redistributing.

## Build and validation

The fork workflow builds unsigned desktop artifacts for Linux x64/ARM64, Windows x64/ARM64, macOS ARM64 and separate native worker bundles for all five. It reuses upstream build scripts but does not use upstream release service credentials. It runs on branch pushes. The registry/dispatch tests are mocked; worker smoke checks load the executable and call --help. The workflow now benchmarks two real Turkish decodes in the same process on Linux x64 for Seda v2, Seda Tiny and Kulak (FLEURS fixture, CC BY 4.0), records cold load, warm latency, WER and RTF, and builds combined portable archives on Linux and Windows plus a macOS app/DMG containing the worker. The Mihu benchmark is deliberately excluded from unattended downloads until license acceptance is handled. The fork desktop workflow now also includes the native runtime in Linux DEB/RPM and unsigned Windows installers; these paths must pass package CI before release. Conversion helper: `sh extras/turkish-stt/native/convert-whisper.sh turkmedstt /local/checkpoint /output` (also accepts `turkspeak-itu`); its output must be validated against the original model before publishing. Native Balbal/Duyu ports, end-to-end microphone tests, live partials, conversion verification, complete installer distribution and benchmark parity remain open. No model weights bundled. Do not present this draft as production-ready.

## Vendor fallback (persistent Python worker)

Vendor models now use the same line-delimited, bounded JSON worker protocol as native Sherpa. A registered Python virtual environment loads weights once and remains warm across dictation clips; if a worker errors or times out, Hex kills and resets it. First install the vendor adapter and **explicitly cache model weights** with the interpreter you intend to register:

```sh
python extras/turkish-stt/stt.py install balbal-manas
python extras/turkish-stt/stt.py download balbal-manas
hex turkish-stt register-python balbal-manas /absolute/path/to/venv/bin/python
hex model use balbal-manas
```

The bundled script is embedded in Hex, so no repository checkout is required at runtime. On Windows specify `venv\\Scripts\\python.exe`. Balbal 0.6.3 has wheels for Linux x64, macOS ARM64 and Windows x64; other targets remain unsupported until native ports or compatible vendor wheels are available. Duyu requires author engine installation from a reviewed checkout and DUYU_SMALL_DIR; download alone is not a substitute. Whisper fallback requires PyTorch/Transformers and significant memory. A Python worker is a **vendor fallback**, not native inference. Model weights are never bundled into HEX installers.

Cross-platform real-speech CI now runs Seda v2, Seda Tiny and Kulak on Linux x64/ARM64, Windows x64/ARM64 and macOS ARM64 rather than only Linux x64. A single fixture checks decoder correctness and repetition, not comprehensive WER parity.

## Additional acceptance tests

Linux x64 and Windows x64/ARM64 now exercise the actual `hex model install seda-tiny`, `hex model check seda-tiny` and `hex model transcribe ... --model seda-tiny` commands against a pinned Turkish FLEURS fixture; this checks the real Rust bridge, artifact downloader, plugin registration and subprocess inference. macOS remains bundle/DMG verification because a headless microphone entitlement cannot be simulated as a normal user. A separate Ubuntu reference test downloads the author-pinned Apache-2.0 Duyu source and int8 weights in an ephemeral runner, then performs two real transcriptions via the vendored Python adapter, recording cold/warm latency and WER. Nothing from the remote model executes in Hex on ordinary installation; the Duyu user must separately install its author runtime.

For TurkMed/TurkSpeak, the conversion script now labels the output as reference dtype, records the Hugging Face repo attribution in GGUF, produces Q8_0 and a SHA-256, and prints the exact `hex model import` command. It does not automatically import unvalidated, multi-GB models or claim reference parity.

## Native incremental streaming (Seda + Mihu)

The Sherpa worker now also accepts a bounded local control stream: `_HEX_BEGIN_`, `_HEX_PCM_<hex-encoded little-endian PCM16>`, `_HEX_END_`. Native Seda v2, Seda Tiny and Mihu online recognizers emit provisional text as soon as the decoder makes it available, then final text after end-of-input. The stream is integrated into HEX's existing live-text/partial infrastructure with cancellation and model switching; provisional text remains **tentative**, not a confirmed committed prefix. Kulak is an offline NeMo TDT model and continues using whole-clip inference, not misleading fake streaming. Python vendor engines also retain whole-clip semantics. The real-speech CI checks incremental decoder final text for both Seda models on five platforms. Do not enable this feature for models without validated online decoding support.

## Versioned app updates

Managed native models keep their immutable verified weights independently of
the installed Hex executable version. On startup Hex locates the worker bundled
with the **currently running** app instead of trusting the absolute path saved
by an older Windows installer, macOS app, or Linux package. User-registered
custom workers and Python virtual environments retain their exact executable
paths and are never silently redirected. This fixes native model load failures
when a previous app version directory is removed.

## Local Whisper checkpoint compatibility

The TurkMedSTT General Hugging Face repository provides `vocab.json` and `merges.txt` but not `tokenizer.json`, which the transcribe.cpp GGUF converter requires. `convert-whisper.sh` now stages the checkpoint outside the author's download directory and generates a fast tokenizer **offline** using the pinned Transformers conversion environment. The original 3 GB `model.safetensors` is linked rather than copied. Checkpoints must include merged Whisper encoder/decoder tensors; incomplete Git-LFS pointers and adapter-only LoRA weights are rejected. The actual 3 GB GGUF conversion and reference transcript parity are still unverified; this fixes prerequisite file preparation only.
