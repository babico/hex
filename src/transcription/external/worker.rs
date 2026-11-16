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
    Ok(result.get("text").and_then(serde_json::Value::as_str)
        .ok_or_else(|| eyre!("External backend must return a JSON text field"))?.trim().to_owned())
}
impl Worker {
    pub(super) fn running(&mut self) -> bool { matches!(self.child.try_wait(), Ok(None)) }
    pub(super) fn spawn(config: &Registration, id: ExternalModelId, language: &str) -> Result<Self> {
        let mut command = Command::new(&config.executable);
        command.args(&config.args).args(["--model", id.as_str(), "--language", language, "--serve", "1"]);
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
    pub(super) fn request(&mut self, path: &str) -> Result<String> {
        if path.contains(['\n', '\r']) || path.len() > 32768 { bail!("Invalid audio path"); }
        writeln!(self.input, "{path}")?;
        self.input.flush()?;
        text(&self.receive(Duration::from_secs(120))?)
    }
}
