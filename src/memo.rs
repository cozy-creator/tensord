//! The machine tier of memoized Model methods (`stage/1`): the small results of costly
//! encoders, shared by every run on this machine and scoped only by their key. An executor
//! asks before it computes (`stage_memo_lookup`) and offers what it computed
//! (`stage_memo_store`). The tier manages itself, with no verb: an entry expires the caches'
//! TTL after it was written, the least recently used go beyond `STORE_BYTES` and in a low
//! disk's covering plan (`reclaim`), and on a low disk a new entry is not written (a miss,
//! never an error).
use crate::{
    device_executor::{Answer, Frame, Kind},
    reclaim::{self, TTL},
};
use std::{
    collections::{HashMap, HashSet},
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
    sync::{Condvar, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};
use tensorfs_core::sha256;

/// The largest result the tier takes: worth storing is costly to produce and small.
pub const ENTRY_BYTES: u64 = 512 << 10;
const STORE_BYTES: u64 = 2 << 30;

pub struct Memo {
    root: PathBuf,
    state: Mutex<State>,
    settled: Condvar,
}

#[derive(Default)]
struct State {
    /// A key one run is computing: other lookups of it wait for the outcome.
    claims: HashMap<String, String>,
    /// (stage, numerics) scopes in which one key was computed twice, differently: reuse stops.
    disputed: HashSet<(String, String)>,
}

/// One entry on disk: `<root>/<key>/<sha256>-<cost µs>`.
struct Entry {
    path: PathBuf,
    sha256: String,
    cost_ms: f64,
}

fn key_ok(key: &str) -> bool {
    (16..=128).contains(&key.len()) && key.bytes().all(|b| b.is_ascii_alphanumeric())
}

impl Memo {
    pub fn open(root: PathBuf) -> io::Result<Self> {
        fs::create_dir_all(&root)?;
        Ok(Self {
            root,
            state: Mutex::default(),
            settled: Condvar::new(),
        })
    }

    fn entry(&self, key: &str) -> Option<Entry> {
        let path = fs::read_dir(self.root.join(key))
            .ok()?
            .flatten()
            .next()?
            .path();
        let name = path.file_name()?.to_str()?.to_owned();
        let (sha256, cost) = name.split_once('-')?;
        Some(Entry {
            path,
            sha256: sha256.into(),
            cost_ms: cost.parse::<u64>().ok()? as f64 / 1000.0,
        })
    }

    /// Answers a memo frame of `run`, whose executor reads and writes `spool`.
    pub fn answer(&self, run: &str, spool: &Path, frame: &Frame) -> Answer {
        let mut answer = Answer::ok(frame.seq);
        if !key_ok(&frame.key) {
            return Answer::refused(
                frame.seq,
                "memo_key_invalid",
                "a memo key is 16 to 128 letters and digits",
            );
        }
        match frame.kind {
            Kind::StageMemoLookup => match self.lookup(run, spool, frame) {
                Ok(Some((local, cost_ms))) => (answer.local, answer.cost_ms) = (local, cost_ms),
                Ok(None) => answer.disputed = self.disputed(frame),
                // An unreadable tier is a miss the run computes through.
                Err(error) => eprintln!("stage memo {}: {error}", &frame.key[..16]),
            },
            _ => {
                let stored = self.store(spool, frame);
                self.settle(&frame.key, run);
                match stored {
                    Ok(reason) => {
                        (answer.stored, answer.reason) = (reason.is_empty(), reason.into())
                    }
                    Err(error) => answer.reason = error.to_string(),
                }
                answer.disputed = answer.reason == "present" && self.disputed(frame);
            }
        }
        answer
    }

    fn disputed(&self, frame: &Frame) -> bool {
        let scope = (frame.stage.clone(), frame.numerics.clone());
        self.state.lock().unwrap().disputed.contains(&scope)
    }

    /// A verified copy in the run's spool with what it cost to produce, or a miss that claims
    /// the key for this run.
    fn lookup(&self, run: &str, spool: &Path, frame: &Frame) -> io::Result<Option<(String, f64)>> {
        let key = &frame.key;
        loop {
            let entry = {
                let mut state = self.state.lock().unwrap();
                loop {
                    if state
                        .disputed
                        .contains(&(frame.stage.clone(), frame.numerics.clone()))
                    {
                        return Ok(None);
                    }
                    if let Some(entry) = self.entry(key) {
                        break entry;
                    }
                    match state.claims.get(key) {
                        Some(owner) if owner != run => {
                            // Its producer stores, abandons, or its run ends: progress, not a clock.
                            state = self.settled.wait(state).unwrap();
                        }
                        _ => {
                            state.claims.insert(key.clone(), run.into());
                            return Ok(None);
                        }
                    }
                }
            };
            let mut raw = vec![];
            let read = File::open(&entry.path).and_then(|mut file| file.read_to_end(&mut raw));
            if read.is_err() || sha256::hex_digest(&raw) != entry.sha256 {
                // A missing or damaged entry is a miss, and goes.
                let _ = fs::remove_dir_all(self.root.join(key));
                continue;
            }
            let stamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos());
            let local = spool.join(format!("stage-memo-{}-{stamp}.safetensors", &key[..16]));
            OpenOptions::new()
                .write(true)
                .create_new(true) // never through a link
                .mode(0o644)
                .custom_flags(libc::O_NOFOLLOW)
                .open(&local)?
                .write_all(&raw)?;
            // Its last use, for the least-recently-used order.
            let _ = File::open(&entry.path).and_then(|file| file.set_modified(SystemTime::now()));
            return Ok(Some((local.to_string_lossy().into_owned(), entry.cost_ms)));
        }
    }

    /// Why the offered entry was not stored; empty when it was.
    fn store(&self, spool: &Path, frame: &Frame) -> io::Result<&'static str> {
        if frame.local.is_empty() {
            return Ok("declined"); // the executor's own decision; its reason is its own record
        }
        if frame.length == 0 || frame.length > ENTRY_BYTES {
            return Ok("too_large");
        }
        if reclaim::measure(&self.root)?.short().is_some() {
            return Ok("disk_low");
        }
        let Some(raw) = spooled(spool, Path::new(&frame.local), frame.length) else {
            return Ok("local");
        };
        let sha256 = sha256::hex_digest(&raw);
        let home = self.root.join(&frame.key);
        if let Some(existing) = self.entry(&frame.key) {
            if existing.sha256 != sha256 {
                let scope = (frame.stage.clone(), frame.numerics.clone());
                self.state.lock().unwrap().disputed.insert(scope);
            }
            return Ok("present");
        }
        // Written beside its home and renamed in whole: a reader never sees part of an entry.
        let staged = self
            .root
            .join(format!(".{}-{}", frame.key, std::process::id()));
        fs::create_dir_all(&staged)?;
        let cost_us = (frame.cost_ms.max(0.0) * 1000.0) as u64;
        fs::write(staged.join(format!("{sha256}-{cost_us}")), &raw)?;
        let _ = fs::remove_dir(&home); // an emptied home, if one is left
        match fs::rename(&staged, &home) {
            Ok(()) => Ok(""),
            Err(_) => {
                let _ = fs::remove_dir_all(&staged);
                Ok("present")
            }
        }
    }

    fn settle(&self, key: &str, run: &str) {
        let mut state = self.state.lock().unwrap();
        if state.claims.get(key).is_some_and(|owner| owner == run) {
            state.claims.remove(key);
            self.settled.notify_all();
        }
    }

    /// The run ended: whatever it was computing, its waiters compute themselves.
    pub fn release(&self, run: &str) {
        let mut state = self.state.lock().unwrap();
        let before = state.claims.len();
        state.claims.retain(|_, owner| owner != run);
        if state.claims.len() != before {
            self.settled.notify_all();
        }
    }

    /// Entries' homes, for a low disk's plan.
    pub fn homes(&self) -> Vec<PathBuf> {
        let homes = fs::read_dir(&self.root).into_iter().flatten().flatten();
        homes
            .filter(|home| {
                home.file_name()
                    .to_str()
                    .is_some_and(|key| self.entry(key).is_some())
            })
            .map(|home| home.path())
            .collect()
    }

    /// The TTL, then the cap; the least recently used go first. Answers how many went.
    pub fn sweep(&self) -> usize {
        let Ok(homes) = fs::read_dir(&self.root) else {
            return 0;
        };
        let now = SystemTime::now();
        let (mut kept, mut total, mut dropped) = (vec![], 0u64, 0);
        for home in homes.flatten() {
            let path = home.path();
            let entry = home.file_name().to_str().and_then(|key| self.entry(key));
            let facts = entry.and_then(|e| fs::metadata(&e.path).ok());
            // Written when its home appeared; used when its file was last touched.
            let written = fs::metadata(&path)
                .and_then(|m| m.modified())
                .unwrap_or(UNIX_EPOCH);
            match facts {
                Some(facts) if now.duration_since(written).is_ok_and(|age| age <= TTL) => {
                    total += facts.size();
                    kept.push((facts.modified().unwrap_or(UNIX_EPOCH), facts.size(), path));
                }
                // Expired, or a home left without its entry (an interrupted store).
                _ => dropped += usize::from(fs::remove_dir_all(&path).is_ok()),
            }
        }
        kept.sort();
        for (_, size, path) in kept {
            if total <= STORE_BYTES {
                break;
            }
            if fs::remove_dir_all(&path).is_ok() {
                (total, dropped) = (total - size, dropped + 1);
            }
        }
        dropped
    }
}

/// The executor's entry file: only from its own spool, never through a link, and exactly as
/// long as it said.
fn spooled(spool: &Path, local: &Path, length: u64) -> Option<Vec<u8>> {
    if local.parent() != Some(spool) {
        return None;
    }
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(local)
        .ok()?;
    if !file.metadata().ok()?.is_file() {
        return None;
    }
    let mut raw = vec![];
    file.take(length + 1).read_to_end(&mut raw).ok()?;
    (raw.len() as u64 == length).then_some(raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn frame(kind: Kind, key: &str, local: &Path, length: u64) -> Frame {
        Frame {
            kind,
            key: key.into(),
            stage: "encode".into(),
            numerics: "bf16".into(),
            local: local.to_string_lossy().into_owned(),
            length,
            cost_ms: 1500.25,
            ..Frame::default()
        }
    }

    #[test]
    fn a_stored_entry_is_a_hit_for_the_next_run_and_a_second_result_disputes_its_scope() {
        let root = std::env::temp_dir().join(format!("cm-memo-{}", uuid::Uuid::new_v4()));
        let spool = root.join("spool");
        fs::create_dir_all(&spool).unwrap();
        let memo = Arc::new(Memo::open(root.join("memo")).unwrap());
        let key = "k".repeat(64);
        let lookup = frame(Kind::StageMemoLookup, &key, Path::new(""), 0);
        // A miss claims the key: a second run's lookup waits for the first run's outcome.
        let miss = memo.answer("run-1", &spool, &lookup);
        assert!(miss.ok && miss.local.is_empty() && !miss.disputed);
        let (waiter, waiting_spool, waiting) = (memo.clone(), spool.clone(), lookup.clone());
        let second = std::thread::spawn(move || waiter.answer("run-2", &waiting_spool, &waiting));
        let computed = spool.join("computed.safetensors");
        fs::write(&computed, b"tensor bytes").unwrap();
        let stored = memo.answer(
            "run-1",
            &spool,
            &frame(Kind::StageMemoStore, &key, &computed, 12),
        );
        assert!(
            stored.stored && stored.reason.is_empty(),
            "{}",
            stored.reason
        );
        let hit = second.join().unwrap();
        assert_eq!(fs::read(&hit.local).unwrap(), b"tensor bytes");
        assert_eq!(hit.cost_ms, 1500.25);
        assert_eq!(Path::new(&hit.local).parent(), Some(spool.as_path()));
        // A file outside the spool, or one whose length differs, is never stored.
        let outside = root.join("outside");
        fs::write(&outside, b"tensor bytes").unwrap();
        let other = "o".repeat(64);
        for (local, length) in [(&outside, 12), (&computed, 11)] {
            let refused = memo.answer(
                "run-1",
                &spool,
                &frame(Kind::StageMemoStore, &other, local, length),
            );
            assert!(
                !refused.stored && refused.reason == "local",
                "{}",
                refused.reason
            );
        }
        // The same key computed again, differently: the first stays and reuse stops for the scope.
        fs::write(&computed, b"other bytes!").unwrap();
        let again = memo.answer(
            "run-3",
            &spool,
            &frame(Kind::StageMemoStore, &key, &computed, 12),
        );
        assert!(!again.stored && again.reason == "present" && again.disputed);
        assert!(memo.answer("run-3", &spool, &lookup).disputed);
        // A run that ends while computing frees its waiters to compute themselves.
        let fresh = frame(Kind::StageMemoLookup, &"f".repeat(64), Path::new(""), 0);
        let fresh = Frame {
            stage: "decode".into(),
            ..fresh
        };
        assert!(memo.answer("run-4", &spool, &fresh).local.is_empty());
        let (waiter, waiting_spool, waiting) = (memo.clone(), spool.clone(), fresh.clone());
        let third = std::thread::spawn(move || waiter.answer("run-5", &waiting_spool, &waiting));
        memo.release("run-4");
        assert!(third.join().unwrap().local.is_empty());
        // Within the TTL and the cap nothing goes; a low disk's plan takes an entry's home.
        assert_eq!(memo.sweep(), 0);
        let homes = memo.homes();
        assert_eq!(homes.len(), 1);
        fs::remove_dir_all(&homes[0]).unwrap();
        assert!(memo
            .answer(
                "run-6",
                &spool,
                &frame(Kind::StageMemoLookup, &key, Path::new(""), 0)
            )
            .local
            .is_empty());
        fs::remove_dir_all(&root).unwrap();
    }
}
