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

## GGUF/reference inference parity gate

The optional fourth argument of `native/convert-whisper.sh` is a locally supplied
16 kHz mono PCM16 Turkish WAV. After conversion to Q8_0 the tool builds the
pinned transcribe.cpp CLI, runs the **original** local Transformers Whisper
checkpoint and the **native** converted GGUF on the same audio, computes
casefolded/normalized Turkish WER and CER, and writes
`MODEL-parity.json` next to the quantized model. If the reference/native
WER exceeds 35%, the command fails and must not be imported. This is a
per-clip porting gate, not proof of production quality or model license clearance.

```sh
sh extras/turkish-stt/native/convert-whisper.sh \
  turkmedstt /local/whisper-checkpoint /local/gguf-output /local/turkish-16khz.wav
```

Without the fourth WAV, only conversion/quantization is performed and the
result is explicitly **unvalidated**; do not treat it as ready. Actual
TurkMedSTT and TurkSpeak full-weight end-to-end inference parity still needs
to pass on their own weights and a broader audio corpus.

## Native PCM transfer without temporary WAV files

The managed Sherpa worker now advertises a `pcm_protocol` capability and accepts bounded
`_HEX_CLIP_BEGIN_`, `_HEX_CLIP_PCM_` and `_HEX_CLIP_END_` frames. HEX transfers
samples directly to the resident native process, avoiding narrow C library path failures
on Windows home directories containing Turkish/Unicode characters, and avoiding an
extra unencrypted temporary WAV on disk. Maximum clip length remains 10 minutes,
with 32,768-sample chunks; canceled or failed transfers kill/reset the worker. Native
workers from older bundles retain the legacy on-disk fallback until upgraded; custom
executables and registered Python engines retain their existing interface.

## Remove managed models and reclaim disk space

The model picker now supports removing a downloaded external model when it is
not active or preparing. `hex turkish-stt remove MODEL` also removes managed
Sherpa weights from the exact immutable revision folder under Hex's own model
directory (after refusing symlinks, redirects and paths outside the owned tree).
Manually registered weights, executables and Python virtual environments are
**never** deleted; only their registration is forgotten. Use
`hex turkish-stt unregister MODEL` to forget a registration without touching
its managed weight files. If a model is still mapped or a download holds the
lock, removal can fail safely; close the engine and retry. If an interrupted
download never created a model registration, the same `remove MODEL` command
reclaims partial and superseded revisions under Hex's own managed weights
tree. It never removes user-owned model folders or Python virtual environments.

## Unicode model weights on Windows

Sherpa's precompiled Windows model loader uses a narrow path interface and
does not handle all UTF-8 model weight directory names. Hex's native worker
keeps the original UTF-8 path for filesystem access but stages the four
reviewed data files under a random, ASCII-only directory within the user's
temporary folder when the path contains non-ASCII characters. It attempts
NTFS hard links first (no weight duplication), falls back to a temporary copy
if the source is on another drive, and cleans up on worker exit. If the user's
TEMP itself contains Unicode, set `HEX_STT_ASCII_SCRATCH_DIR` to a writable
ASCII-only path. Native five-platform tests intentionally load all three
Sherpa models from a `Türkçe_Ağırlıklar` folder.

## Worker recovery and managed-model integrity

The native Sherpa worker and the optional persistent Python vendor worker return
structured JSON errors when model initialization fails. Hex surfaces those
errors in model preparation instead of just reporting that the worker vanished.
A failed/canceled stream or clip disposes of the worker before another
recording; backend errors do not silently reuse its decoder state.

Native PCM frames pass through a one-slot writer queue, keeping a stuck child
stdin pipe off Hex's transcription thread. Whole-clip requests use an **overall**
deadline that scales with the audio duration (up to 25 minutes), rather than
renewing an entire timeout for every PCM frame. Streaming still uses individual
request deadlines. Vendor Python and older WAV-only workers have a
separate duration-scaled timeout (up to 45 minutes) for slower inference.
These bounded limits are not guarantees that all hardware will decode long
recordings within the allotted time.

Managed weights require a receipt matching their source repository, immutable
revision, exact required files, file sizes, and SHA-256 values. Download,
verification and removal reject symlinked files/directories under the Hex-owned
model tree rather than following paths outside it. Registration cleanup works
even if the previous app worker, Python virtual environment, or model revision
directory has already disappeared; manually supplied data is never deleted.
On Linux and Windows, model selection now validates managed and Python
worker initialization **before** saving the new choice; a canceled
preparation terminates the loading child. macOS retains its existing
pre-activation load check. WAV-only manually registered tools still
require a real-audio check.

The fork's CI additionally runs the `transcription::external` Rust unit tests
on Linux, alongside the cross-platform Python/native checks and actual
recorded-speech tests. A new test's presence alone is not proof of a passing CI
run; inspect the current branch run before marking the feature complete.

## Independent Turkish corpus evaluation

The earlier smoke jobs prove decoding on one known sentence; they are **not** a
population accuracy measurement. The native evaluator can now score any
locally reviewed JSONL manifest (one row per 16 kHz mono PCM16 WAV, paths
relative to the manifest):

```jsonl
{"wav":"audio/01.wav","text":"Iğdır ve İstanbul hakkında konuşuyoruz."}
{"wav":"audio/02.wav","text":"Türkçe konuşma tanıma testi."}
```

The manifest is supplied by the user; the evaluator does not download audio,
accept licenses, or touch the selected weights. Use the same manifest across
models and hardware for a meaningful comparison:

```sh
python extras/turkish-stt/native/benchmark_corpus.py \
  --worker /absolute/native-dist/hex-sherpa-worker \
  --weights /absolute/seda-tiny-weights \
  --model seda-tiny --manifest /absolute/corpus.jsonl \
  --output /absolute/metrics/seda-tiny.json \
  --max-wer 0.30 --max-cer 0.15
```

The worker stays loaded across the corpus and uses bounded, in-memory PCM
frames (also safe for Unicode WAV paths on Windows). Each clip records
transcript, ground truth, decoding time and real-time factor; the report
includes **micro-averaged** WER/CER, total errors and a reproducible worker
SHA-256. Thresholds are explicit and optional. The single FLEURS fixture in
the Linux x64 CI exercises this evaluator's real native path; it does not
stand in for a large independent corpus.

The existing Whisper converter also accepts a JSONL manifest as the optional
fourth argument:

```sh
sh extras/turkish-stt/native/convert-whisper.sh \
  turkmedstt /local/checkpoint /local/output /local/corpus.jsonl
```

This runs the original Transformers checkpoint and converted Q8_0 GGUF against
each clip, recording per-clip reference/native parity and reference/native WER
against ground truth. The report includes an aggregate corpus WER/CER, and
the Transformers reference is loaded only once. You can use the validator
directly with `--max-native-wer` to reject a converted model that exceeds a
specified corpus error rate. No full TurkMed/TurkSpeak conversion or corpus
accuracy is claimed until that exact checkpoint and manifest pass.
