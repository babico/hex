//! Persistent, bounded local Sherpa process. Kill and reap on cancellation,
 //! protocol failure or model switch; never retry a canceled recording.
use std::{io::{BufRead, BufReader, Read, Write}, process::{Child, ChildStdin, Command, Stdio}, sync::mpsc, time::{Duration, Instant}};
use color_eyre::eyre::{Result, bail, eyre};
use super::{Registration, ExternalModelId};

pub(super) struct Worker {
    child: Child,
    input: ChildStdin,
    output: mpsc::Receiver<Vec<u8>>,
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
impl Worker {
    pub(super) fn running(&mut self) -> bool { matches!(self.child.try_wait(), Ok(None)) }
    pub(super) fn spawn(config: &Registration, id: ExternalModelId, language: &str) -> Result<Self> {
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
        let input = child.stdin.take().expect("piped stdin");
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
        let worker = Self { child, input, output };
        let ready: serde_json::Value = serde_json::from_slice(&worker.receive(Duration::from_secs(90))?)?;
        if ready.get("ready").and_then(serde_json::Value::as_bool) != Some(true) { bail!("External worker was not ready"); }
        Ok(worker)
    }
    fn receive(&self, timeout: Duration) -> Result<Vec<u8>> {
        let deadline = Instant::now() + timeout;
        loop {
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
        if message.contains(['\n','\r']) || message.len() > 131100 { bail!("Invalid worker message"); }
        writeln!(self.input, "{message}")?;
        self.input.flush()?;
        let deadline = Instant::now() + timeout;
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

    pub(super) fn request(&mut self, path: &str) -> Result<String> {
        if path.len() > 32768 { bail!("Invalid audio path"); }
        self.exchange(path, Duration::from_secs(120), None)
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
    use super::text;

    #[test]
    fn vendor_errors_are_visible_without_leaking_control_characters() {
        let error = text(br#"{"error":"missing vendor model cache"}"#).unwrap_err();
        assert!(error.to_string().contains("missing vendor model cache"));
        assert_eq!(text(r#"{"text":" Merhaba dünya "}"#.as_bytes()).unwrap(), "Merhaba dünya");
        assert!(text(br#"{"ready":true}"#).is_err());
    }
}
