//! One bounded triage bundle per failed attempt (the Runtime worker's `triage.py`): what a
//! person needs to explain the failure after its executor is gone, as one canonical JSON
//! file. Written and fsynced before the terminal names it, and observation only: nothing in
//! it overrides a terminal field. The CLI quotes `terminal.traceback`.
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Write},
    os::unix::fs::OpenOptionsExt,
    path::Path,
};
use tensorfs_core::sha256;

/// `TriageBundleRef.length` is bounded at 1 MiB by the protocol; fields are capped well below.
const MAX_TRACEBACK: usize = 256 << 10;
const MAX_TEXT: usize = 16 << 10;
/// Enough executor stderr to hold a crash's faulthandler stacks past the extension-module list
/// Python prints after them.
pub const MAX_STDERR: u64 = 64 << 10;

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct TriageRef {
    pub subject_id: String,
    /// Hex SHA-256 of the bundle's canonical bytes.
    pub sha256: String,
    pub length: u64,
}

/// What the machine knows of one failed attempt.
pub struct Facts<'a> {
    pub request_id: &'a str,
    pub attempt: u32,
    pub terminal: &'a str,
    pub origin: &'a str,
    pub code: &'a str,
    pub message: &'a str,
    pub traceback: &'a str,
    pub executor_pid: u32,
    pub stderr_tail: &'a str,
}

/// `text` within `limit` chars: whole, or its head and its tail around a marker. A failure's
/// first line names the operation and its last lines the cause (uv's error after its progress).
pub fn ends(text: &str, limit: usize) -> String {
    const MARK: &str = " [...] ";
    let count = text.chars().count();
    if count <= limit {
        return text.into();
    }
    let head = limit / 4;
    let tail = limit.saturating_sub(head + MARK.len());
    text.chars().take(head).chain(MARK.chars()).chain(text.chars().skip(count - tail)).collect()
}

fn capped(text: &str, limit: usize) -> &str {
    if text.len() <= limit {
        return text;
    }
    // Keep the end: a traceback's last lines name the failure.
    let mut start = text.len() - limit;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    &text[start..]
}

/// Write the bundle under `root/triage/` and return the reference the terminal carries; it
/// names this TensorD and, once an executor said hello, the Runtime and TensorFS it ran.
pub fn write(
    root: &Path,
    facts: &Facts,
    executor: Option<&crate::journal::ExecutorFacts>,
) -> io::Result<TriageRef> {
    let subject_id = format!("trb-{}", &uuid::Uuid::new_v4().simple().to_string()[..24]);
    let bundle = json!({
        "format": "cozy.machine.triage/1",
        "subject_id": subject_id,
        "request_id": facts.request_id,
        "attempt_ordinal": facts.attempt,
        "terminal": {
            "terminal": facts.terminal,
            "origin": facts.origin,
            "code": capped(facts.code, MAX_TEXT),
            "message": capped(facts.message, MAX_TEXT),
            "traceback": capped(facts.traceback, MAX_TRACEBACK),
        },
        "executor": {"pid": facts.executor_pid, "stderr_tail": capped(facts.stderr_tail, MAX_STDERR as usize)},
        "software": executor.map(|executor| executor.software(facts.attempt)).unwrap_or_else(|| {
            // A launch failure has no executor; only this TensorD was observed.
            crate::journal::ExecutorFacts::observed(0, String::new(), String::new()).software(facts.attempt)
        }),
    });
    let bytes = serde_json_canonicalizer::to_vec(&bundle).map_err(io::Error::other)?;
    let directory = root.join("triage");
    fs::create_dir_all(&directory)?;
    let path = directory.join(format!("{subject_id}.json"));
    let pending = directory.join(format!(".{subject_id}.pending"));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&pending)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    fs::rename(&pending, &path)?;
    File::open(&directory)?.sync_all()?;
    Ok(TriageRef {
        subject_id,
        sha256: sha256::hex(&sha256::digest(&bytes)),
        length: bytes.len() as u64,
    })
}

/// The bundle's bytes, checked against the reference its terminal carries.
pub fn read(root: &Path, triage: &TriageRef) -> io::Result<Vec<u8>> {
    if !triage
        .subject_id
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'-')
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "triage subject is not a plain name",
        ));
    }
    let bytes = fs::read(
        root.join("triage")
            .join(format!("{}.json", triage.subject_id)),
    )?;
    if bytes.len() as u64 != triage.length || sha256::hex(&sha256::digest(&bytes)) != triage.sha256
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "triage bundle differs from its terminal's reference",
        ));
    }
    Ok(bytes)
}
