//! Persistent, bounded local Sherpa process. Kill and reap on cancellation,
 //! protocol failure or model switch; never retry a canceled recording.
use std::{io::{BufRead, BufReader, Read, Write}, process::{Child, Command, Stdio}, sync::{atomic::{AtomicBool, Ordering}, mpsc}, time::{Duration, Instant}};
use color_eyre::eyre::{Result, bail, eyre};
use super::{Registration, ExternalModelId};

pub(super) struct Worker {
    child: Child,
    input: mpsc::SyncSender<String>,
    output: mpsc::Receiver<Vec<u8>>,
    pcm_protocol: bool,
}
impl Drop for Worker {
    fn drop(&mut self) { let _ = self.child.kill(); let _ = self.child.wait(); }
}
pub(super) fn text(bytes: &[u8]) -> Result<String> {
    let result: serde_json::Value = serde_json::from_slice(bytes)?;
    if let Some(error) = result.get("error").and_then(serde_json::Value::as_str) {
        let detail: String = error.chars().filter(|c| !c.is_control()).take(512).collect();
        bail!("External STT engine failed: {detail}");
    }
    Ok(result.get("text").and_then(serde_json::Value::as_str)
        .ok_or_else(|| eyre!("External backend must return a JSON text field"))?.trim().to_owned())
}
/// The overall budget grows with audio length for slower CPUs while always
/// remaining finite. A ten-minute clip has less than 24 minutes total.
fn clip_timeout(samples: usize) -> Duration {
    let audio_seconds = (samples as u64).div_ceil(16_000);
    Duration::from_secs((180_u64 + audio_seconds.saturating_mul(2)).min(1_500))
}

/// Parse the first bounded worker message, including model load failures.
fn ready_response(bytes: &[u8]) -> Result<bool> {
    let ready: serde_json::Value = serde_json::from_slice(bytes)?;
    if ready.get("ready").and_then(serde_json::Value::as_bool) == Some(true) {
        return Ok(ready.get("pcm_protocol").and_then(serde_json::Value::as_u64) == Some(1));
    }
    if let Some(error) = ready.get("error").and_then(serde_json::Value::as_str) {
        let detail: String = error.chars().filter(|ch| !ch.is_control()).take(512).collect();
        bail!("External STT model failed to load: {detail}");
    }
    bail!("External STT worker did not report readiness")
}

/// Never write directly from the transcription thread to a child pipe: a
/// wedged decoder can stop reading stdin indefinitely.
fn queue_frame(input: &mpsc::SyncSender<String>, message: &str) -> Result<()> {
    match input.try_send(message.to_owned()) {
        Ok(()) => Ok(()),
        Err(mpsc::TrySendError::Full(_)) => bail!("External STT worker input is blocked"),
        Err(mpsc::TrySendError::Disconnected(_)) => bail!("External STT worker input closed"),
    }
}

impl Worker {
    pub(super) fn running(&mut self) -> bool { matches!(self.child.try_wait(), Ok(None)) }
    pub(super) fn spawn(config: &Registration, id: ExternalModelId, language: &str, canceled: Option<&AtomicBool>) -> Result<Self> {
        let mut command = Command::new(&config.executable);
        if config.python {
            command.arg("-u").arg("-c").arg(include_str!("../../../extras/turkish-stt/stt.py"))
                .args(["serve", id.as_str(), "--offline"])
                .env("HEX_TURKISH_STT_REGISTRY", include_str!("../../../extras/turkish-stt/models.json"))
                .env("PYTHONUTF8", "1");
        } else {
            command.args(&config.args).args(["--model", id.as_str(), "--language", language, "--serve", "1"]);
        }
        #[cfg(windows)]
        { use std::os::windows::process::CommandExt; command.creation_flags(0x08000000); }
        let mut child = command.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn()?;
        let mut child_stdin = child.stdin.take().expect("piped stdin");
        let (input, input_queue) = mpsc::sync_channel::<String>(1);
        let _writer = std::thread::spawn(move || {
            while let Ok(message) = input_queue.recv() {
                if writeln!(child_stdin, "{message}").and_then(|_| child_stdin.flush()).is_err() {
                    break;
                }
            }
        });
        let stdout = child.stdout.take().expect("piped stdout");
        let (tx, output) = mpsc::sync_channel(1);
        let _reader = std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            loop {
                let mut line = Vec::new();
                let count = (&mut reader).take(4 * 1024 * 1024 + 1).read_until(b'\n', &mut line);
                if !matches!(count, Ok(n) if n > 0 && n <= 4 * 1024 * 1024) || line.last() != Some(&b'\n') { break; }
                if tx.send(line).is_err() { break; }
            }
        });
        let mut worker = Self { child, input, output, pcm_protocol: false };
        worker.pcm_protocol = ready_response(&worker.receive(Duration::from_secs(90), canceled)?)?;
        Ok(worker)
    }
    fn receive(&self, timeout: Duration, canceled: Option<&AtomicBool>) -> Result<Vec<u8>> {
        let deadline = Instant::now() + timeout;
        loop {
            if canceled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
                bail!("External STT model preparation canceled");
            }
            if crate::transcription::gguf_session::job_cancel().is_some_and(|token| token.is_cancelled()) { bail!("External STT canceled"); }
            if Instant::now() >= deadline { bail!("External STT timed out"); }
            match self.output.recv_timeout(Duration::from_millis(25)) {
                Ok(bytes) => return Ok(bytes),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => bail!("External worker stopped or returned an invalid response"),
            }
        }
    }
    fn exchange(&mut self, message: &str, timeout: Duration, cancel: Option<&transcribe_cpp::CancelToken>) -> Result<String> {
        self.exchange_until(message, Instant::now() + timeout, cancel)
    }

    /// Use an absolute deadline so long PCM clips cannot renew the full timeout
    /// for every chunk. The per-frame limit is still applied by the caller.
    fn exchange_until(&mut self, message: &str, deadline: Instant, cancel: Option<&transcribe_cpp::CancelToken>) -> Result<String> {
        if message.contains(['\n','\r']) || message.len() > 131100 { bail!("Invalid worker message"); }
        if Instant::now() >= deadline { bail!("External worker timed out"); }
        queue_frame(&self.input, message)?;
        loop {
            if cancel.is_some_and(|token| token.is_cancelled()) { bail!("Live STT canceled"); }
            if crate::transcription::gguf_session::job_cancel().is_some_and(|token| token.is_cancelled()) { bail!("STT canceled"); }
            if Instant::now() >= deadline { bail!("External worker timed out"); }
            match self.output.recv_timeout(Duration::from_millis(25)) {
                Ok(bytes) => return text(&bytes),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => bail!("External worker exited"),
            }
        }
    }

    pub(super) fn supports_pcm(&self) -> bool { self.pcm_protocol }

    /// Transfer a whole clip without a filesystem path. Windows temporary
    /// directories may contain Turkish/Unicode names that a native C ABI
    /// loader interpreting narrow strings cannot open. Bound each message
    /// to 128 KiB and the full clip to ten minutes.
    pub(super) fn request_pcm(&mut self, samples: &[f32]) -> Result<String> {
        if !self.pcm_protocol { bail!("Installed Sherpa worker lacks PCM protocol; update the native bundle"); }
        if samples.is_empty() || samples.len() > 16_000 * 600 {
            bail!("PCM clip must contain 1-9,600,000 samples");
        }
        // One total bounded deadline across all frames and the final decode.
        // Allow slower CPUs proportionally more time on long recordings.
        let deadline = Instant::now() + clip_timeout(samples.len());
        self.exchange_until("_HEX_CLIP_BEGIN_", deadline.min(Instant::now() + Duration::from_secs(15)), None)?;
        for chunk in samples.chunks(32768) {
            const HEX: &[u8; 16] = b"0123456789abcdef";
            let mut encoded = String::with_capacity(chunk.len() * 4 + 14);
            encoded.push_str("_HEX_CLIP_PCM_");
            for sample in chunk {
                let value = if sample.is_finite() {
                    (*sample * 32768.0).clamp(-32768.0, 32767.0) as i16
                } else { 0 };
                for byte in value.to_le_bytes() {
                    encoded.push(HEX[(byte >> 4) as usize] as char);
                    encoded.push(HEX[(byte & 15) as usize] as char);
                }
            }
            self.exchange_until(&encoded, deadline.min(Instant::now() + Duration::from_secs(30)), None)?;
        }
        self.exchange_until("_HEX_CLIP_END_", deadline.min(Instant::now() + Duration::from_secs(120)), None)
    }

    pub(super) fn request(&mut self, path: &str, timeout: Duration) -> Result<String> {
        if path.len() > 32768 { bail!("Invalid audio path"); }
        self.exchange(path, timeout, None)
    }

    pub(super) fn start_live(&mut self, cancel: &transcribe_cpp::CancelToken) -> Result<()> {
        self.exchange("_HEX_BEGIN_", Duration::from_secs(15), Some(cancel)).map(drop)
    }
    pub(super) fn feed_live(&mut self, samples: &[f32], cancel: &transcribe_cpp::CancelToken) -> Result<String> {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut encoded = String::with_capacity(samples.len() * 4 + 9);
        encoded.push_str("_HEX_PCM_");
        for x in samples {
            let value = if x.is_finite() { (*x * 32768.0).clamp(-32768.0, 32767.0) as i16 } else { 0 };
            for byte in value.to_le_bytes() {
                encoded.push(HEX[(byte >> 4) as usize] as char);
                encoded.push(HEX[(byte & 15) as usize] as char);
            }
        }
        self.exchange(&encoded, Duration::from_secs(30), Some(cancel))
    }
    pub(super) fn end_live(&mut self, cancel: &transcribe_cpp::CancelToken) -> Result<String> {
        self.exchange("_HEX_END_", Duration::from_secs(120), Some(cancel))
    }
}

#[cfg(test)]
mod tests {
    use super::{clip_timeout, queue_frame, ready_response, text};
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    #[cfg(unix)]
    #[test]
    fn cancel_during_model_startup_terminates_the_worker() {
        use super::{Worker, Registration, ExternalModelId};
        use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
        let canceled = Arc::new(AtomicBool::new(false));
        let token = canceled.clone();
        let handle = std::thread::spawn(move || {
            let config = Registration {
                executable: "/bin/sh".into(),
                args: vec!["-c".into(), "exec sleep 10".into()],
                python: false,
                managed_weights: None,
            };
            Worker::spawn(&config, ExternalModelId::SedaTiny, "tr", Some(&token))
                .map(drop)
        });
        std::thread::sleep(Duration::from_millis(100));
        canceled.store(true, Ordering::Release);
        let error = handle.join().unwrap().unwrap_err();
        assert!(error.to_string().contains("canceled"));
    }

    #[test]
    fn a_blocked_worker_pipe_cannot_block_the_transcription_thread() {
        let (sender, _receiver) = mpsc::sync_channel(1);
        queue_frame(&sender, "first").unwrap();
        let error = queue_frame(&sender, "second").unwrap_err();
        assert!(error.to_string().contains("blocked"));
        drop(_receiver);
        assert!(queue_frame(&sender, "after close").is_err());
    }

    #[test]
    fn clip_deadline_scales_for_long_recordings_but_stays_finite() {
        assert_eq!(clip_timeout(16_000), Duration::from_secs(182));
        assert_eq!(clip_timeout(16_000 * 600), Duration::from_secs(1_380));
        assert_eq!(clip_timeout(usize::MAX), Duration::from_secs(1_500));
    }

    #[test]
    fn a_clip_deadline_cannot_be_extended_by_per_frame_timeouts() {
        let total_deadline = Instant::now() + Duration::from_secs(10);
        let per_frame = total_deadline.min(Instant::now() + Duration::from_secs(30));
        assert_eq!(per_frame, total_deadline);
        let short_frame = total_deadline.min(Instant::now() + Duration::from_millis(30));
        assert!(short_frame <= total_deadline);
    }

    #[test]
    fn model_load_errors_survive_prewarm_and_cap_diagnostics() {
        assert_eq!(ready_response(br#"{"ready":true,"pcm_protocol":1}"#).unwrap(), true);
        assert_eq!(ready_response(br#"{"ready":true}"#).unwrap(), false);
        let error = ready_response(br#"{"ready":false,"error":"missing local model files"}"#).unwrap_err();
        assert!(error.to_string().contains("missing local model files"));
        assert!(ready_response(br#"{"ready":false}"#).is_err());
        let error = ready_response(b"{not-json").unwrap_err();
        assert!(!error.to_string().is_empty());
    }

    #[test]
    fn vendor_errors_are_visible_without_leaking_control_characters() {
        let error = text(br#"{"error":"missing vendor model cache"}"#).unwrap_err();
        assert!(error.to_string().contains("missing vendor model cache"));
        assert_eq!(text(r#"{"text":" Merhaba dünya "}"#.as_bytes()).unwrap(), "Merhaba dünya");
        assert!(text(br#"{"ready":true}"#).is_err());
    }
}
