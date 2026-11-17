//! Experimental local external speech runtime. No shell or implicit downloads.
use std::ffi::OsString;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use color_eyre::eyre::{Result, bail, eyre, WrapErr};
use serde::{Deserialize, Serialize};
use super::models::TranscriptionSelection;
pub(crate) mod install;
mod worker;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExternalModelId {
    BalbalManas,
    BalbalV2,
    MihuV2,
    DuyuSmall,
    Kulak,
    SedaV2,
    SedaTiny,
    TurkMed,
    TurkSpeak,
}
impl ExternalModelId {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BalbalManas => "balbal-manas",
            Self::BalbalV2 => "balbal-v2",
            Self::MihuV2 => "mihu-v2",
            Self::DuyuSmall => "duyu-small",
            Self::Kulak => "ufakzeka-kulak",
            Self::SedaV2 => "seda-v2",
            Self::SedaTiny => "seda-tiny",
            Self::TurkMed => "turkmedstt",
            Self::TurkSpeak => "turkspeak-itu",
        }
    }
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "balbal-manas" => Some(Self::BalbalManas),
            "balbal-v2" => Some(Self::BalbalV2),
            "mihu-v2" => Some(Self::MihuV2),
            "duyu-small" => Some(Self::DuyuSmall),
            "ufakzeka-kulak" => Some(Self::Kulak),
            "seda-v2" => Some(Self::SedaV2),
            "seda-tiny" => Some(Self::SedaTiny),
            "turkmedstt" => Some(Self::TurkMed),
            "turkspeak-itu" => Some(Self::TurkSpeak),
            _ => None,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct Registration {
    executable: PathBuf,
    args: Vec<String>,
    python: bool,
    #[serde(default)]
    managed_weights: Option<PathBuf>,
}

pub fn registration_path(id: ExternalModelId) -> Result<PathBuf> {
    Ok(super::models::models_dir()?.join("external").join(format!("{}.json", id.as_str())))
}

fn read_registration(id: ExternalModelId) -> Result<Registration> {
    let bytes = fs::read(registration_path(id)?)
        .wrap_err_with(|| format!("Register {} first: hex turkish-stt register-python {} /absolute/path/to/python", id.as_str(), id.as_str()))?;
    let config: Registration = serde_json::from_slice(&bytes)?;
    if !config.executable.is_absolute() || !config.executable.is_file() {
        bail!("External STT executable is missing or is not an absolute file path");
    }
    Ok(config)
}

pub fn registered(id: ExternalModelId) -> bool { read_registration(id).is_ok() }

pub fn register(id: ExternalModelId, executable: &Path, args: Vec<String>, python: bool) -> Result<()> {
    // Preserve virtualenv interpreter symlinks; resolving them loses the venv.
    let executable = std::path::absolute(executable)?;
    if !executable.is_file() { bail!("Expected an executable file"); }
    let path = registration_path(id)?;
    fs::create_dir_all(path.parent().ok_or_else(|| eyre!("Missing model directory"))?)?;
    super::models::write_json_atomically(&path, &Registration { executable, args, python, managed_weights: None })
}

pub(super) fn register_managed(id: ExternalModelId, executable: &Path, dir: &Path) -> Result<()> {
    let weights = dir.to_str().ok_or_else(|| eyre!("Model folder must be UTF-8"))?;
    super::models::write_json_atomically(&registration_path(id)?, &Registration { executable: executable.to_path_buf(), args: vec!["--weights".into(), weights.into()], python: false, managed_weights: Some(dir.to_path_buf()) })
}
pub fn verify_registration(id: ExternalModelId, canceled: &std::sync::atomic::AtomicBool) -> Result<PathBuf> {
    let config = read_registration(id)?;
    if let Some(dir) = &config.managed_weights { install::verify(dir, Some(canceled))?; }
    registration_path(id)
}
pub fn unregister(id: ExternalModelId) -> Result<()> {
    match fs::remove_file(registration_path(id)?) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}

pub fn cli(args: &[OsString]) -> Result<bool> {
    let action = args.first().and_then(|a| a.to_str()).unwrap_or("");
    if !matches!(action, "register" | "register-python" | "unregister" | "accept-license") { return Ok(false); }
    let id = args.get(1).and_then(|a| a.to_str()).and_then(ExternalModelId::parse)
        .ok_or_else(|| eyre!("Unknown model; run hex turkish-stt list"))?;
    if action == "accept-license" {
        if args.len() != 2 { bail!("Usage: hex turkish-stt accept-license MODEL"); }
        install::accept_license(id)?;
        return Ok(true);
    }
    if action == "unregister" {
        if args.len() != 2 { bail!("Usage: hex turkish-stt unregister MODEL"); }
        unregister(id)?;
        return Ok(true);
    }
    let executable = args.get(2).ok_or_else(|| eyre!("Expected executable path"))?;
    let extra = args[3..].iter().map(|a| a.to_str().map(str::to_owned)
        .ok_or_else(|| eyre!("Backend arguments must be UTF-8"))).collect::<Result<Vec<_>>>()?;
    if action == "register-python" && !extra.is_empty() { bail!("Unexpected Python arguments"); }
    register(id, Path::new(executable), extra, action == "register-python")?;
    println!("Registered {}; select it in Hex Settings or use hex model use {}.", id.as_str(), id.as_str());
    Ok(true)
}

pub struct ExternalRunner {
    pub selection: TranscriptionSelection,
    id: ExternalModelId,
    config: Registration,
    worker: std::sync::Mutex<Option<worker::Worker>>,
}

struct Scratch(PathBuf);
impl Scratch {
    fn create() -> Result<Self> {
        let base = super::models::models_dir()?.join("external");
        fs::create_dir_all(&base)?;
        let mut nonce = [0_u8; 16];
        getrandom::fill(&mut nonce).map_err(|e| eyre!("Could not generate scratch directory name: {e}"))?;
        let name = nonce.iter().map(|n| format!("{n:02x}")).collect::<String>();
        let path = base.join(format!("clip-{name}"));
        let mut builder = fs::DirBuilder::new();
        #[cfg(unix)]
        { use std::os::unix::fs::DirBuilderExt; builder.mode(0o700); }
        builder.create(&path)?;
        Ok(Self(path))
    }
}
impl Drop for Scratch {
    fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); }
}

pub struct ExternalStream<'a> {
    parent: &'a ExternalRunner,
    cancel: transcribe_cpp::CancelToken,
    partial: String,
    complete: bool,
}
impl ExternalStream<'_> {
    pub fn feed(&mut self, samples: &[f32]) -> Result<()> {
        for chunk in samples.chunks(4096) {
            if self.cancel.is_cancelled() { bail!("Live transcription canceled"); }
            let mut guard = self.parent.worker.lock().map_err(|_| eyre!("External STT lock poisoned"))?;
            let answer = guard.as_mut().ok_or_else(|| eyre!("External STT worker disappeared"))?.feed_live(chunk, &self.cancel);
            if answer.is_err() { *guard = None; }
            self.partial = answer?;
        }
        Ok(())
    }
    pub fn finish(&mut self) -> Result<String> {
        let mut guard = self.parent.worker.lock().map_err(|_| eyre!("External STT lock poisoned"))?;
        let answer = guard.as_mut().ok_or_else(|| eyre!("External STT worker disappeared"))?.end_live(&self.cancel);
        if answer.is_err() { *guard = None; }
        self.complete = true;
        answer
    }
    pub fn partial(&self) -> &str { &self.partial }
}
impl Drop for ExternalStream<'_> {
    fn drop(&mut self) {
        // Incomplete streams are never shared with the next recording.
        if !self.complete && let Ok(mut guard) = self.parent.worker.lock() { *guard = None; }
    }
}

impl ExternalRunner {
    pub fn can_stream(&self) -> bool {
        self.config.managed_weights.is_some()
            && matches!(self.id, ExternalModelId::SedaV2 | ExternalModelId::SedaTiny | ExternalModelId::MihuV2)
    }
    pub fn begin_stream(&self, cancel: &transcribe_cpp::CancelToken) -> Result<ExternalStream<'_>> {
        if !self.can_stream() { bail!("This external backend cannot stream"); }
        self.prewarm()?;
        let mut guard = self.worker.lock().map_err(|_| eyre!("External STT lock poisoned"))?;
        let started = guard.as_mut().ok_or_else(|| eyre!("External STT worker disappeared"))?.start_live(cancel);
        if started.is_err() { *guard = None; }
        started?;
        Ok(ExternalStream { parent: self, cancel: cancel.clone(), partial: String::new(), complete: false })
    }
    pub fn load(selection: &TranscriptionSelection) -> Result<Self> {
        let super::models::TranscriptionModelId::External(id) = selection.model else { bail!("Expected external STT model"); };
        let config = read_registration(id)?;
        if let Some(dir) = &config.managed_weights { install::verify(dir, None)?; }
        Ok(Self { selection: selection.clone(), id, config, worker: std::sync::Mutex::new(None) })
    }

    pub fn prewarm(&self) -> Result<()> {
        if self.config.managed_weights.is_none() && !self.config.python { return Ok(()); }
        let mut guard = self.worker.lock().map_err(|_| eyre!("External worker lock poisoned"))?;
        if guard.as_mut().is_some_and(worker::Worker::running) { return Ok(()); }
        *guard = None;
        *guard = Some(worker::Worker::spawn(&self.config, self.id, &self.selection.language)?);
        Ok(())
    }

    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        if samples.is_empty() { return Ok(String::new()); }
        if samples.len() > 16_000 * 600 { bail!("External STT clip exceeds ten minutes"); }
        let scratch = Scratch::create()?;
        let wav = scratch.0.join("audio.wav");
        let response = scratch.0.join("result.json");
        let spec = hound::WavSpec { channels: 1, sample_rate: 16_000, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
        let mut writer = hound::WavWriter::create(&wav, spec)?;
        for value in samples {
            let sample = if value.is_finite() { (*value * 32768.0).clamp(-32768.0, 32767.0) as i16 } else { 0 };
            writer.write_sample(sample)?;
        }
        writer.finalize()?;
        if self.config.managed_weights.is_some() || self.config.python {
            self.prewarm()?;
            let path = wav.to_str().ok_or_else(|| eyre!("Audio path must be UTF-8"))?;
            let mut guard = self.worker.lock().map_err(|_| eyre!("External worker lock poisoned"))?;
            let result = guard.as_mut().ok_or_else(|| eyre!("External worker missing"))?.request(path);
            if result.is_err() { *guard = None; }
            return result;
        }
        let mut command = Command::new(&self.config.executable);
        if self.config.python {
            command.arg("-c").arg(include_str!("../../extras/turkish-stt/stt.py"))
                .args(["transcribe", self.id.as_str()]).arg(&wav).arg("--offline")
                .env("HEX_TURKISH_STT_REGISTRY", include_str!("../../extras/turkish-stt/models.json"));
        } else {
            command.args(&self.config.args).args(["--model", self.id.as_str(), "--language", &self.selection.language, "--wav"]).arg(&wav);
        }
        #[cfg(windows)]
        { use std::os::windows::process::CommandExt; command.creation_flags(0x08000000); }
        let mut child = command.stdin(Stdio::null()).stdout(Stdio::from(File::create(&response)?))
            .stderr(Stdio::null()).spawn().wrap_err("Could not launch registered STT backend")?;
        let deadline = Instant::now() + Duration::from_secs(120);
        let result: Result<_> = loop {
            match child.try_wait() {
                Ok(Some(status)) => break Ok(status),
                Ok(None) if super::gguf_session::job_cancel().is_some_and(|token| token.is_cancelled()) => break Err(eyre!("External STT canceled")),
                Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(25)),
                Ok(None) => break Err(eyre!("External STT backend timed out")),
                Err(e) => break Err(e.into()),
            }
        };
        if result.is_err() { let _ = child.kill(); let _ = child.wait(); }
        if !result?.success() { bail!("External STT backend failed; test with hex turkish-stt transcribe"); }
        if fs::metadata(&response)?.len() > 4 * 1024 * 1024 { bail!("External STT response exceeds 4 MiB"); }
        let result: serde_json::Value = serde_json::from_slice(&fs::read(&response)?)?;
        Ok(result.get("text").and_then(serde_json::Value::as_str)
            .ok_or_else(|| eyre!("External backend must return a JSON text field"))?.trim().to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stable_ids() {
        let mut count = 0;
        for model in super::super::models::MODELS {
            if let super::super::models::TranscriptionModelId::External(id) = model.id {
                count += 1;
                assert_eq!(ExternalModelId::parse(id.as_str()), Some(id));
                assert_eq!(model.id.as_str().parse::<super::super::models::TranscriptionModelId>().unwrap(), model.id);
                assert!(model.supports_language("tr"));
                assert!(!model.streams);
            }
        }
        assert_eq!(count, 9);
    }
}
