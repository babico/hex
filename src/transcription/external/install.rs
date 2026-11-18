//! Local downloads of data-only models. Never downloads or executes code.
use std::{fs, path::{Path, PathBuf}, sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering}, time::Duration};
use color_eyre::eyre::{Result, bail, eyre};
use serde::{Deserialize, Serialize};
use super::{ExternalModelId, registration_path};
use crate::transcription::models::{self, ModelPreparationStage};

const MIHU: &[&str] = &["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"];
const SEDA: &[&str] = &["encoder.int8.onnx", "decoder.onnx", "joiner.int8.onnx", "tokens.txt"];
const KULAK: &[&str] = &["sherpa-onnx-tdt/encoder.int8.onnx", "sherpa-onnx-tdt/decoder.int8.onnx", "sherpa-onnx-tdt/joiner.int8.onnx", "sherpa-onnx-tdt/tokens.txt"];

fn source(id: ExternalModelId) -> Option<(&'static str, &'static str, &'static [&'static str])> {
    use ExternalModelId::*;
    Some(match id {
        MihuV2 => ("mihuai/turkish-stt-66m-v2", "bc2f77287cfc3ec2568cf5ba9f9181e1a43bfff4", MIHU),
        // Seda's immutable revision is resolved and recorded on installation.
        SedaV2 => ("atasoglu/seda-v0.2", "main", SEDA),
        SedaTiny => ("atasoglu/seda-v0.2-tiny", "main", SEDA),
        Kulak => ("ufakai/ufakzeka-kulak", "886194bb06f71460169cd741e1fd8ed515fd9d1c", KULAK),
        _ => return None,
    })
}

pub(crate) fn supported(id: ExternalModelId) -> bool { source(id).is_some() }

pub(super) fn worker() -> Result<PathBuf> {
    let name = if cfg!(windows) { "hex-sherpa-worker.exe" } else { "hex-sherpa-worker" };
    let exe = std::env::current_exe()?;
    let mut paths = vec![exe.parent().ok_or_else(|| eyre!("No executable directory"))?.join("turkish-stt").join(name)];
    if let Some(dir) = std::env::var_os("HEX_STT_NATIVE_DIR") { paths.insert(0, PathBuf::from(dir).join(name)); }
    #[cfg(target_os = "linux")]
    paths.push(PathBuf::from("/usr/lib/hex/turkish-stt").join(name));
    paths.into_iter().find(|p| p.is_absolute() && p.is_file()).ok_or_else(|| eyre!("Native worker not found. Install the matching fork bundle or set HEX_STT_NATIVE_DIR."))
}

#[derive(Deserialize)]
struct Hub { sha: String, siblings: Vec<Sibling> }
#[derive(Deserialize)]
struct Sibling { rfilename: String, size: Option<u64>, lfs: Option<Lfs> }
#[derive(Deserialize)]
struct Lfs { sha256: String }
#[derive(Serialize, Deserialize)]
struct Receipt { repo: String, revision: String, files: Vec<Stored> }
#[derive(Serialize, Deserialize)]
struct Stored { name: String, bytes: u64, sha256: String }

fn check(canceled: &AtomicBool) -> Result<()> {
    if canceled.load(Ordering::Acquire) { bail!("Model installation canceled"); } Ok(())
}
fn acknowledgement(id: ExternalModelId) -> Result<PathBuf> {
    Ok(models::models_dir()?.join("external").join(format!("{}.license-accepted", id.as_str())))
}
pub(crate) fn accept_license(id: ExternalModelId) -> Result<()> {
    if id != ExternalModelId::MihuV2 { bail!("Only Mihu requires this acknowledgement"); }
    let path = acknowledgement(id)?;
    fs::create_dir_all(path.parent().ok_or_else(|| eyre!("Missing model directory"))?)?;
    crate::durable::write(&path, b"mihu Community License 1.0; user acknowledgement\n")?;
    Ok(())
}

pub(crate) fn install(id: ExternalModelId, canceled: &AtomicBool, downloaded: &AtomicU64, stage: &AtomicU8) -> Result<PathBuf> {
    let (repo, revision, required) = source(id).ok_or_else(|| eyre!("Register the model's vendor runtime first"))?;
    let executable = worker()?;
    if id == ExternalModelId::MihuV2 && !acknowledgement(id)?.is_file() {
        bail!("Powered by mihu. Review https://huggingface.co/mihuai/turkish-stt-66m-v2/blob/main/LICENSE; if eligible and you accept, run hex turkish-stt accept-license mihu-v2, then choose the model again.");
    }
    check(canceled)?;
    ModelPreparationStage::Downloading.store(stage);
    let url = format!("https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true");
    let agent = crate::net::agent(&url, |cfg| cfg.https_only(true).timeout_recv_response(Some(Duration::from_secs(60))).timeout_recv_body(Some(Duration::from_secs(60))));
    let mut response = agent.get(&url).header("User-Agent", "hex-turkish-stt").call()?;
    let meta: Hub = serde_json::from_reader(response.body_mut().with_config().limit(8 << 20).reader())?;
    if meta.sha.len() != 40 || !meta.sha.bytes().all(|b| b.is_ascii_hexdigit()) || (revision != "main" && meta.sha != revision) { bail!("Invalid model revision"); }
    let dir = models::models_dir()?.join("external").join("weights").join(id.as_str()).join(&meta.sha);
    fs::create_dir_all(&dir)?;
    let _lock = models::lock_downloads(&dir, canceled, || {})?;
    let mut names = required.to_vec();
    for extra in ["README.md", "LICENSE", "ATTRIBUTION.md", "SHA256SUMS"] {
        if meta.siblings.iter().any(|s| s.rfilename == extra) { names.push(extra); }
    }
    let mut done = 0;
    let mut files = Vec::new();
    for name in names {
        check(canceled)?;
        let info = meta.siblings.iter().find(|s| s.rfilename == name).ok_or_else(|| eyre!("Missing model file: {name}"))?;
        let size = info.size.ok_or_else(|| eyre!("Missing file size: {name}"))?;
        if size == 0 || size > 2_000_000_000 { bail!("Invalid file size: {name}"); }
        let digest = info.lfs.as_ref().map(|l| l.sha256.as_str());
        if name.ends_with(".onnx") && !digest.is_some_and(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())) { bail!("No published ONNX SHA-256: {name}"); }
        let target = dir.join(name);
        fs::create_dir_all(target.parent().ok_or_else(|| eyre!("Invalid model path"))?)?;
        let good = fs::metadata(&target).is_ok_and(|m| m.len() == size)
            && digest.is_some_and(|d| models::sha256_hex(&target, Some(canceled)).is_ok_and(|actual| actual == d));
        if !good {
            let partial = target.with_file_name(format!("{}.partial", target.file_name().ok_or_else(|| eyre!("No file name"))?.to_string_lossy()));
            if fs::metadata(&partial).is_ok_and(|m| m.len() > size) { fs::remove_file(&partial)?; }
            let url = format!("https://huggingface.co/{repo}/resolve/{}/{name}", meta.sha);
            crate::net::download(&crate::net::Download { url: &url, user_agent: "hex-turkish-stt", size: Some(size), canceled: Some(canceled) }, &partial,
                |n| downloaded.store(done + n, Ordering::Relaxed))?;
            let actual = models::sha256_hex(&partial, Some(canceled))?;
            if fs::metadata(&partial)?.len() != size || digest.is_some_and(|d| actual != d) {
                let _ = fs::remove_file(&partial);
                bail!("Model checksum or size mismatch: {name}");
            }
            if target.exists() { fs::remove_file(&target)?; }
            fs::rename(&partial, &target)?;
        }
        files.push(Stored { name: name.to_owned(), bytes: size, sha256: models::sha256_hex(&target, Some(canceled))? });
        done += size; downloaded.store(done, Ordering::Relaxed);
    }
    ModelPreparationStage::Verifying.store(stage);
    models::write_json_atomically(&dir.join("receipt.json"), &Receipt { repo: repo.to_owned(), revision: meta.sha, files })?;
    verify(&dir, Some(canceled))?;
    if id == ExternalModelId::MihuV2 { crate::durable::write(&dir.join("NOTICE"), &fs::read(dir.join("LICENSE"))?)?; }
    check(canceled)?;
    super::register_managed(id, &executable, &dir)?;
    registration_path(id)
}

/// Only the direct immutable revision subdirectory of Hex's own
/// managed weight tree may be removed. Do not follow arbitrary registration
/// paths or delete folders belonging to manually registered backends.
fn owned_revision_dir(root: &Path, dir: &Path) -> bool {
    let Some(revision) = dir.file_name().and_then(|name| name.to_str()) else { return false; };
    dir.parent() == Some(root)
        && revision.len() == 40
        && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
}

pub(super) fn remove_managed(id: ExternalModelId, dir: &Path) -> Result<()> {
    let root = models::models_dir()?.join("external").join("weights").join(id.as_str());
    if !owned_revision_dir(&root, dir) {
        bail!("Refusing to delete a weight directory outside Hex's managed model revision");
    }
    if fs::symlink_metadata(&root)?.file_type().is_symlink()
        || fs::symlink_metadata(dir)?.file_type().is_symlink()
    {
        bail!("Refusing to follow symlinked native STT model directories");
    }
    let canonical_root = root.canonicalize()?;
    if dir.canonicalize()?.parent() != Some(canonical_root.as_path()) {
        bail!("Refusing to delete model weights outside Hex's managed folder");
    }
    // The same download lock prevents removing files during an install or
    // verification. Keep it held while deleting the directory's contents.
    let canceled = AtomicBool::new(false);
    let lock = models::lock_downloads(dir, &canceled, || {})?;
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if entry.file_name().to_string_lossy() == ".download.lock" { continue; }
        if entry.file_type()?.is_dir() {
            fs::remove_dir_all(entry.path())?;
        } else {
            fs::remove_file(entry.path())?;
        }
    }
    drop(lock);
    fs::remove_file(dir.join(".download.lock"))?;
    fs::remove_dir(dir)?;
    Ok(())
}

#[cfg(test)]
mod remove_tests {
    use super::*;
    #[test]
    fn managed_receipts_must_match_the_repository_revision_and_required_files() {
        let revision = "a".repeat(40);
        let dir = Path::new("models/external/weights/seda-tiny").join(&revision);
        let files = SEDA.iter().map(|name| Stored {
            name: (*name).to_owned(), bytes: 42, sha256: "b".repeat(64),
        }).collect();
        let mut receipt = Receipt {
            repo: "atasoglu/seda-v0.2-tiny".into(), revision: revision.clone(), files,
        };
        assert!(validate_receipt(&dir, &receipt).is_ok());
        receipt.repo = "someone-else/model".into();
        assert!(validate_receipt(&dir, &receipt).is_err());
        receipt.repo = "atasoglu/seda-v0.2-tiny".into();
        receipt.revision = "c".repeat(40);
        assert!(validate_receipt(&dir, &receipt).is_err());
        receipt.revision = revision;
        receipt.files.pop();
        assert!(validate_receipt(&dir, &receipt).is_err());
    }

    #[test]
    fn only_exact_managed_revision_subdirectories_are_removable() {
        let root = Path::new("models/external/weights/seda-tiny");
        let revision = "a".repeat(40);
        assert!(owned_revision_dir(root, &root.join(&revision)));
        assert!(!owned_revision_dir(root, root));
        assert!(!owned_revision_dir(root, &root.join("different")));
        assert!(!owned_revision_dir(root, &root.join(&revision).join("nested")));
        assert!(!owned_revision_dir(root, &root.join("..").join("elsewhere")));
    }
}

/// A receipt must describe this model and immutable source revision, not
/// simply contain hashes chosen by whoever last wrote receipt.json.
fn validate_receipt(dir: &Path, receipt: &Receipt) -> Result<()> {
    let id = dir.parent()
        .and_then(Path::file_name)
        .and_then(|part| part.to_str())
        .and_then(ExternalModelId::parse)
        .ok_or_else(|| eyre!("Invalid managed STT model directory"))?;
    let (expected_repo, _, required) = source(id)
        .ok_or_else(|| eyre!("Model has no managed download manifest"))?;
    let revision = dir.file_name().and_then(|part| part.to_str())
        .ok_or_else(|| eyre!("Missing model revision"))?;
    if receipt.repo != expected_repo || receipt.revision != revision
        || revision.len() != 40 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        bail!("Model receipt repository or revision mismatch");
    }

    let extras = ["README.md", "LICENSE", "ATTRIBUTION.md", "SHA256SUMS"];
    let mut seen = std::collections::HashSet::new();
    for file in &receipt.files {
        if !required.contains(&file.name.as_str()) && !extras.contains(&file.name.as_str()) {
            bail!("Unrecognized managed model file: {}", file.name);
        }
        if !seen.insert(file.name.as_str()) {
            bail!("Duplicate managed model file: {}", file.name);
        }
        if file.bytes == 0 || file.bytes > 2_000_000_000 || file.sha256.len() != 64
            || !file.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            bail!("Invalid managed model file metadata: {}", file.name);
        }
    }
    for required_file in required {
        if !seen.contains(required_file) {
            bail!("Missing required managed model file: {required_file}");
        }
    }
    Ok(())
}

pub(super) fn verify(dir: &Path, canceled: Option<&AtomicBool>) -> Result<()> {
    let receipt: Receipt = serde_json::from_slice(&fs::read(dir.join("receipt.json"))?)?;
    validate_receipt(dir, &receipt)?;
    for file in receipt.files {
        let relative = Path::new(&file.name);
        if !relative.components().all(|part| matches!(part, std::path::Component::Normal(_))) { bail!("Unsafe model path"); }
        let path = dir.join(relative);
        if fs::metadata(&path)?.len() != file.bytes || models::sha256_hex(&path, canceled)? != file.sha256 { bail!("Model checksum failed: {}", path.display()); }
    }
    Ok(())
}
