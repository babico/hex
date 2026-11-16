# Experimental Turkish STT companion

The fork adds nine local-model entries through an independent CLI. This is an initial implementation, **not yet a replacement for Hex's native dictation engine or its Settings picker**. The existing models and defaults are unchanged.

## Quick start

Use Python 3.11 or 3.12 in a virtual environment. Run from the repository before compiling:

```sh
python extras/turkish-stt/stt.py list
python extras/turkish-stt/stt.py install balbal-manas
python extras/turkish-stt/stt.py transcribe balbal-manas speech.wav
```

After building the fork:

```sh
hex turkish-stt list
hex turkish-stt install balbal-manas
hex turkish-stt transcribe balbal-manas speech.wav
hex turkish-stt transcribe balbal-manas speech.wav --offline
```

Set HEX_STT_PYTHON to the absolute interpreter path in the selected virtual environment. On Windows the fallback interpreter is python; elsewhere python3. List works without Python. Input is mono 16 kHz PCM16 WAV; convert other input with your chosen audio tool. Dependencies install only on explicit request; weights may download on the first invocation. Offline mode requests cached weights via the Hugging Face environment/API, but is not an operating-system network sandbox.

## Model status

All registry entries have experimental, unverified inference adapters. Balbal's upstream wheels are documented for Linux x64, macOS ARM64 and Windows x64; other architectures need native wheel validation. Duyu requires a reviewed local checkout of its engine, installed with pip install -e . and DUYU_SMALL_DIR pointing to the checkout. Mihu v2 uses its own Community License: review before redistribution. Whisper entries use full Transformers models, rather than an unverified GGUF conversion. No weights are bundled. Model downloads are not pinned in this prototype; pin revisions and checksums before production.

## Build

Use Actions → Turkish STT fork checks and unsigned builds. Push checks exercise the registry and mocked dispatch on Linux x64/ARM64, Windows x64/ARM64, and macOS ARM64. The manual desktop build reuses upstream's unsigned Linux/Windows jobs and builds an unsigned macOS app and DMG. If GitHub does not expose branch-only manual workflows yet, merge the reviewed workflow into the default branch first or run the upstream build scripts locally. No upstream release service credentials or signing keys are used; the build jobs do not publish a release. Packages are not proof of model wheel availability. Native macOS signing/notarization and Windows installer signing are separate release tasks.

## Remaining native integration

- Extend ModelRuntime, TranscriptionModelId and ModelDefinition with a real external runtime. Never pass ONNX weights to the GGUF engine; keep existing validated GGUF paths intact.
- Connect the new catalog to src/rune_ui/models.rs, speech_model.rs, preparation and control/model listing; reflect explicit installation and platform compatibility.
- Implement persistent worker lifecycle in macOS Transcriber and Linux/Windows LocalTranscriber; route engine, service, CLI and meeting requests without loading a dummy GGUF model. Support cancel, timeouts, offline policy and bounded inputs; no invented timestamps.
- Use per-model verified revisions/artifacts, hash checks and licensed redistribution; isolate vendor environments. Add actual integration fixtures and cross-platform inference checks. Enable streaming only after partial/final parity is tested.
- Keep the stock Whisper model as default until Turkish and TR/EN coding benchmark results demonstrate a better default. Measure WER, code-term accuracy, warm/cold latency and memory on actual devices.

The companion is a development scaffold with callable adapters, not a tested full integration.
