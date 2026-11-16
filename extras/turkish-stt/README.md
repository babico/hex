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

On Windows use the .exe and absolute paths. The native picker lists the models; after registration select the model in Settings if the app is running. Restart the app if its installed-state cache has not refreshed. Unregister with `hex turkish-stt unregister MODEL`. Registration checks the executable path, not model integrity or inference compatibility. No automatic weights or dependency download is performed by the picker. Initial implementation runs a process per clip, not a persistent model or live partials.

## Other models: existing vendor adapter fallback

Balbal Manas and Balbal v2 use the upstream package (custom ops + KenLM), Duyu its model-author Python engine (multi-stage ONNX + tokenizer), TurkMed and TurkSpeak Transformers until verified native ports or transcribe.cpp conversions exist. Their Python adapters are not yet validated on any platform. Python 3.11/3.12 virtual environments are recommended. Install selected dependencies and download/cache weights explicitly; check the actual file-based inference before registering:

```sh
hex turkish-stt list
hex turkish-stt install balbal-manas
hex turkish-stt transcribe balbal-manas speech.wav
hex turkish-stt register-python balbal-manas /absolute/path/to/venv/bin/python
hex model use balbal-manas
```

Set HEX_STT_PYTHON to the virtual environment interpreter for companion CLI commands. The registered interpreter is used for dictation, requesting offline mode. This is not a network sandbox; only explicitly trusted local executables/packages should be registered. Duyu requires a reviewed local engine checkout, installed with pip install -e . and DUYU_SMALL_DIR set for the app. Vendor dependencies and weights need pinned versions and real tests before production; Balbal weights are non-commercial and Mihu has a custom Community License.

## Build and validation

The fork workflow builds unsigned desktop artifacts for Linux x64/ARM64, Windows x64/ARM64, macOS ARM64 and separate native worker bundles for all five. It reuses upstream build scripts but does not use upstream release service credentials. It runs on branch pushes. The registry/dispatch tests are mocked; worker smoke checks load the executable and call --help. Actual inference tests, native Balbal/Duyu implementation, verified Whisper conversion, native installers containing companion workers, license notices review, auto-download, streaming/persistent workers, job cancellation and benchmark parity are still required. No model weights bundled. Do not present this draft as production-ready.
