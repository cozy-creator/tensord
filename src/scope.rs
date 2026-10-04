//! An executor's containment scope (Runtime `proctree.ExecutorScope`): its own cgroup-v2
//! below the machine's where the host delegates one, else a process-tree token. Either
//! reaches descendants that call `setsid` or double-fork. Lifecycle containment, not a
//! sandbox: same-UID code can move out of a delegated cgroup or scrub its token.
use crate::process::{Exact, gone, process_birth, process_ended};
use std::{
    collections::BTreeSet,
    fs::{self, File, OpenOptions},
    io::{self, Read, Seek, Write},
    os::{fd::AsRawFd, unix::fs::MetadataExt},
    path::{Path, PathBuf},
    sync::Mutex,
    time::Duration,
};

const ROOT: &str = "/sys/fs/cgroup";
/// Runtime `child_env.EXECUTOR_SCOPE_ENV`: one complete inherited environment entry.
pub const TOKEN_ENV: &str = "COZY_EXECUTOR_SCOPE";

/// This process's live tree scopes: a token on no live scope is unclaimed.
static LIVE: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());

#[derive(Debug)]
pub struct Scope(Backend);

#[derive(Debug)]
enum Backend {
    /// One exact directory, protected against path reuse by its inode.
    Cgroup { relative: String, inode: u64 },
    /// `<namespace>-<id>` in the leader's environment, inherited by every descendant that
    /// keeps it. An escaped one is adopted and reaped by the machine's supervisor (the
    /// subreaper); a `/proc` census finds it by its token wherever it is parented.
    Tree { token: String },
}

/// Durable facts for one already-created containment scope, not sandbox authority.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Recovery {
    Cgroup {
        relative: String,
        inode: u64,
    },
    /// Supported token descendants keep their launched UID and inherited token.
    /// A token-scrubbing/UID-changing process is outside this best-effort boundary.
    Tree {
        token: String,
        uid: u32,
    },
}

impl Recovery {
    /// Strict observed emptiness. An unreadable relevant census is not an empty scope.
    pub fn empty(&self) -> io::Result<bool> {
        match self {
            Self::Cgroup { relative, inode } => {
                if !relative.starts_with('/')
                    || Path::new(relative).components().any(|part| {
                        !matches!(
                            part,
                            std::path::Component::RootDir | std::path::Component::Normal(_)
                        )
                    })
                {
                    return Err(io::Error::other("invalid recovered cgroup path"));
                }
                let expected = Path::new(ROOT).join(relative.trim_start_matches('/'));
                let path = match fs::metadata(&expected) {
                    Ok(metadata) if metadata.ino() == *inode => Some(expected),
                    Ok(_) => find_cgroup_inode(Path::new(ROOT), *inode)?,
                    Err(error) if gone(&error) => find_cgroup_inode(Path::new(ROOT), *inode)?,
                    Err(error) => return Err(error),
                };
                // The old inode is absent in a complete census. Kernel cgroups can only
                // be removed empty; scanning avoids mistaking a renamed group for gone.
                path.map_or(Ok(true), |path| count_cgroup(&path).map(|count| count == 0))
            }
            Self::Tree { token, uid } => {
                if token.is_empty()
                    || token.len() > 128
                    || !token
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
                {
                    return Err(io::Error::other("invalid recovered scope token"));
                }
                let (rows, unknown) = census_partial(Path::new("/proc"), *uid)?;
                if members(&rows, |value| value == token).any(|row| row.alive) {
                    return Ok(false); // A known live reader is enough to prohibit release.
                }
                match unknown {
                    Some(error) => Err(error),
                    None => Ok(true),
                }
            }
        }
    }
}

fn count_cgroup(path: &Path) -> io::Result<usize> {
    let mut total = fs::read_to_string(path.join("cgroup.procs"))?
        .lines()
        .filter(|line| !line.is_empty())
        .count();
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            total += count_cgroup(&entry.path())?;
        }
    }
    Ok(total)
}

fn find_cgroup_inode(path: &Path, inode: u64) -> io::Result<Option<PathBuf>> {
    if fs::metadata(path)?.ino() == inode {
        return Ok(Some(path.into()));
    }
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            match find_cgroup_inode(&entry.path(), inode) {
                Ok(Some(path)) => return Ok(Some(path)),
                Ok(None) => (),
                Err(error) if gone(&error) => (),
                Err(error) => return Err(error),
            }
        }
    }
    Ok(None)
}

/// Opaque, stable per state root, so a machine only ever sweeps its own scopes.
pub fn namespace(state: &Path) -> String {
    let path = state.canonicalize().unwrap_or_else(|_| state.to_path_buf());
    tensorfs_core::sha256::hex_digest(path.as_os_str().as_encoded_bytes())[..16].to_string()
}

fn own_cgroup() -> io::Result<PathBuf> {
    let entries = fs::read_to_string("/proc/self/cgroup")?;
    let relative = entries
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .ok_or_else(|| io::Error::other("no unified cgroup-v2 membership"))?;
    Ok(Path::new(ROOT).join(relative.trim().trim_start_matches('/')))
}

fn writable_unified() -> bool {
    if !Path::new(ROOT).join("cgroup.controllers").is_file() {
        return false;
    }
    let path = std::ffi::CString::new(ROOT).unwrap();
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: valid path and writable statvfs.
    unsafe { libc::statvfs(path.as_ptr(), &mut stat) == 0 && stat.f_flag & libc::ST_RDONLY == 0 }
}

impl Scope {
    /// A fresh scope: a cgroup where the unified hierarchy is writable and delegated, else a
    /// token (read-only or v1 hierarchies: Docker, RunPod).
    pub fn create(namespace: &str) -> io::Result<Self> {
        let id = &uuid::Uuid::new_v4().simple().to_string()[..16];
        if let Some(scope) = Self::cgroup(namespace, id)? {
            return Ok(scope);
        }
        let token = format!("{namespace}-{id}");
        LIVE.lock().unwrap().insert(token.clone());
        Ok(Self(Backend::Tree { token }))
    }

    fn cgroup(namespace: &str, id: &str) -> io::Result<Option<Self>> {
        if !writable_unified() {
            return Ok(None);
        }
        let path = own_cgroup()?.join(format!("cozy-executor-{namespace}-{id}"));
        if let Err(error) = fs::create_dir(&path) {
            return match error.kind() {
                io::ErrorKind::PermissionDenied | io::ErrorKind::ReadOnlyFilesystem => Ok(None),
                _ => Err(error),
            };
        }
        // An undelegated hierarchy shows here, before any executor exists.
        let usable = ["cgroup.procs", "cgroup.kill"]
            .iter()
            .all(|name| OpenOptions::new().write(true).open(path.join(name)).is_ok())
            && path.join("cgroup.events").is_file();
        if !usable {
            let _ = fs::remove_dir(&path);
            return Ok(None);
        }
        Ok(Some(Self(Backend::Cgroup {
            relative: format!("/{}", path.strip_prefix(ROOT).unwrap().display()),
            inode: fs::metadata(&path)?.ino(),
        })))
    }

    /// Snapshot after the receiver's launch UID is fixed. No process is created here.
    pub fn recovery(&self, uid: u32) -> Recovery {
        match &self.0 {
            Backend::Cgroup { relative, inode } => Recovery::Cgroup {
                relative: relative.clone(),
                inode: *inode,
            },
            Backend::Tree { token } => Recovery::Tree {
                token: token.clone(),
                uid,
            },
        }
    }

    /// Its cgroup (`/…`, as `0::` lines name it), when it is one.
    pub fn cgroup_relative(&self) -> Option<&str> {
        match &self.0 {
            Backend::Cgroup { relative, .. } => Some(relative),
            Backend::Tree { .. } => None,
        }
    }

    /// The entry its leader is launched with (a fork: in the environment it is forked with).
    pub fn environment(&self) -> Option<(&'static str, &str)> {
        match &self.0 {
            Backend::Cgroup { .. } => None,
            Backend::Tree { token } => Some((TOKEN_ENV, token)),
        }
    }

    fn path(relative: &str, inode: u64) -> io::Result<PathBuf> {
        let path = Path::new(ROOT).join(relative.trim_start_matches('/'));
        if fs::metadata(&path)?.ino() != inode {
            return Err(io::Error::other(format!(
                "executor cgroup {relative} was replaced"
            )));
        }
        Ok(path)
    }

    /// The trampoline joins a cgroup before it imports anything; a token needs nothing of it
    /// (the machine gives the process group).
    pub fn trampoline_args(&self) -> Vec<String> {
        match &self.0 {
            Backend::Cgroup { relative, inode } => vec![
                "--scope-backend".into(),
                "cgroup_v2".into(),
                "--cgroup".into(),
                relative.clone(),
                "--cgroup-inode".into(),
                inode.to_string(),
            ],
            Backend::Tree { .. } => vec!["--scope-backend".into(), "inherit".into()],
        }
    }

    /// Live processes in the scope.
    pub fn processes(&self) -> io::Result<usize> {
        match &self.0 {
            Backend::Cgroup { relative, inode } => count_cgroup(&Self::path(relative, *inode)?),
            Backend::Tree { token } => {
                let rows = census()?;
                Ok(members(&rows, |t| t == token)
                    .filter(|row| row.alive)
                    .count())
            }
        }
    }

    /// Move a running process into a cgroup scope (a fork starts in its parent's).
    pub fn adopt(&self, pid: u32) -> io::Result<()> {
        match &self.0 {
            Backend::Cgroup { relative, inode } => OpenOptions::new()
                .write(true)
                .open(Self::path(relative, *inode)?.join("cgroup.procs"))?
                .write_all(pid.to_string().as_bytes()),
            Backend::Tree { .. } => Ok(()),
        }
    }

    pub fn kill(&self) -> io::Result<()> {
        match &self.0 {
            Backend::Cgroup { relative, inode } => OpenOptions::new()
                .write(true)
                .open(Self::path(relative, *inode)?.join("cgroup.kill"))?
                .write_all(b"1"),
            Backend::Tree { token } => {
                let rows = census()?;
                for row in members(&rows, |t| t == token).filter(|row| row.alive) {
                    if let Some(exact) = row.exact()? {
                        exact.kill()?;
                    }
                }
                Ok(())
            }
        }
    }

    /// Kill whatever outlived the executor (after its leader is reaped), wait for it, and
    /// remove the scope. Returns how many processes were still inside.
    pub fn end(&self) -> io::Result<usize> {
        match &self.0 {
            Backend::Cgroup { relative, inode } => {
                let stragglers = self.processes()?;
                if stragglers > 0 {
                    self.kill()?;
                }
                let path = Self::path(relative, *inode)?;
                wait_empty(&path)?;
                remove_tree(&path)?;
                Ok(stragglers)
            }
            Backend::Tree { token } => {
                let stragglers = end_members(|t| t == token)?;
                LIVE.lock().unwrap().remove(token);
                // Anything of this machine's that no live scope claims goes now too.
                let namespace = token
                    .rsplit_once('-')
                    .map_or("", |(namespace, _)| namespace);
                Ok(stragglers + end_members(unclaimed(namespace))?)
            }
        }
    }

    /// At machine start every scope of this namespace belongs to an earlier run: kill, wait
    /// for and remove each. Returns how many processes they still held.
    pub fn sweep(namespace: &str) -> io::Result<usize> {
        let mut held = end_members(unclaimed(namespace))?;
        if !writable_unified() {
            return Ok(held);
        }
        let prefix = format!("cozy-executor-{namespace}-");
        for entry in fs::read_dir(own_cgroup()?)? {
            let entry = entry?;
            if !entry.file_name().to_string_lossy().starts_with(&prefix) {
                continue;
            }
            let scope = Self(Backend::Cgroup {
                relative: format!("/{}", entry.path().strip_prefix(ROOT).unwrap().display()),
                inode: entry.metadata()?.ino(),
            });
            held += scope.end()?;
        }
        Ok(held)
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        if let Backend::Tree { token } = &self.0 {
            LIVE.lock().unwrap().remove(token);
        }
    }
}

/// Block until nothing in the cgroup is alive: woken by `cgroup.events`, no deadline.
fn wait_empty(path: &Path) -> io::Result<()> {
    let mut events = File::open(path.join("cgroup.events"))?;
    loop {
        let mut text = String::new();
        events.rewind()?;
        events.read_to_string(&mut text)?;
        if text.lines().any(|line| line == "populated 0") {
            return Ok(());
        }
        let mut poll = libc::pollfd {
            fd: events.as_raw_fd(),
            events: libc::POLLPRI,
            revents: 0,
        };
        // SAFETY: one live descriptor; kernfs wakes it when the file changes.
        if unsafe { libc::poll(&mut poll, 1, -1) } < 0 {
            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::Interrupted {
                return Err(error);
            }
        }
    }
}

fn remove_tree(path: &Path) -> io::Result<()> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            remove_tree(&entry.path())?;
        }
    }
    fs::remove_dir(path)
}

/// One process as a census reads it.
struct Row {
    pid: u32,
    ppid: u32,
    start_ticks: u64,
    alive: bool,
    token: Option<String>,
}
impl Row {
    fn exact(&self) -> io::Result<Option<Exact>> {
        match process_birth(self.pid) {
            Ok(birth) if birth.start_ticks == self.start_ticks => Exact::open(&birth),
            Ok(_) => Ok(None),
            Err(error) if gone(&error) => Ok(None),
            Err(error) => Err(error),
        }
    }
}

/// Every process but this one and init, with its scope token where its environment is
/// readable (a zombie has none).
fn census() -> io::Result<Vec<Row>> {
    let own = std::process::id();
    let mut rows = Vec::new();
    for entry in fs::read_dir("/proc")? {
        let Some(pid) = entry?
            .file_name()
            .to_str()
            .and_then(|n| n.parse::<u32>().ok())
        else {
            continue;
        };
        if pid == own || pid == 1 {
            continue;
        }
        let Ok(stat) = fs::read_to_string(format!("/proc/{pid}/stat")) else {
            continue;
        };
        let Some((_, suffix)) = stat.rsplit_once(") ") else {
            continue;
        };
        let fields: Vec<_> = suffix.split_whitespace().collect();
        let (Some(state), Some(Ok(ppid)), Some(Ok(start_ticks))) = (
            fields.first(),
            fields.get(1).map(|f| f.parse()),
            fields.get(19).map(|f| f.parse()),
        ) else {
            continue;
        };
        let alive = !matches!(*state, "Z" | "X");
        let token = alive
            .then(|| fs::read(format!("/proc/{pid}/environ")).ok())
            .flatten()
            .and_then(|environ| {
                environ.split(|b| *b == 0).find_map(|entry| {
                    let value = entry
                        .strip_prefix(TOKEN_ENV.as_bytes())?
                        .strip_prefix(b"=")?;
                    String::from_utf8(value.to_vec()).ok()
                })
            });
        rows.push(Row {
            pid,
            ppid,
            start_ticks,
            alive,
            token,
        });
    }
    Ok(rows)
}

/// Strict proof only: ordinary best-effort containment teardown retains its existing
/// census. Read foreign UID metadata but do not require unrelated foreign environments.
#[cfg(test)]
fn census_strict(proc_root: &Path, uid: u32) -> io::Result<Vec<Row>> {
    let (rows, unknown) = census_partial(proc_root, uid)?;
    match unknown {
        Some(error) => Err(error),
        None => Ok(rows),
    }
}

fn census_partial(proc_root: &Path, uid: u32) -> io::Result<(Vec<Row>, Option<io::Error>)> {
    let mut rows = Vec::new();
    let mut unknown = None;
    for entry in fs::read_dir(proc_root)? {
        let entry = entry?;
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|name| name.parse::<u32>().ok())
        else {
            continue;
        };
        if pid == std::process::id() || pid == 1 {
            continue;
        }
        let read = |name: &str| fs::read(entry.path().join(name));
        let inspect = || -> io::Result<Row> {
            let stat = String::from_utf8(read("stat")?).map_err(io::Error::other)?;
            let (_, suffix) = stat
                .rsplit_once(") ")
                .ok_or_else(|| io::Error::other("invalid scope process stat"))?;
            let fields: Vec<_> = suffix.split_whitespace().collect();
            let field = |index| {
                fields
                    .get(index)
                    .copied()
                    .ok_or_else(|| io::Error::other("scope process stat truncated"))
            };
            let alive = !matches!(field(0)?, "Z" | "X");
            let ppid = field(1)?.parse().map_err(io::Error::other)?;
            let start_ticks = field(19)?.parse().map_err(io::Error::other)?;
            let token = if alive {
                let status = String::from_utf8(read("status")?).map_err(io::Error::other)?;
                let actual = status
                    .lines()
                    .find_map(|line| line.strip_prefix("Uid:"))
                    .and_then(|line| line.split_whitespace().nth(1))
                    .ok_or_else(|| io::Error::other("scope process status has no effective UID"))?
                    .parse::<u32>()
                    .map_err(io::Error::other)?;
                if actual == uid {
                    let environment = read("environ")?;
                    environment
                        .split(|byte| *byte == 0)
                        .find_map(|entry| {
                            entry.strip_prefix(TOKEN_ENV.as_bytes())?.strip_prefix(b"=")
                        })
                        .map(|value| String::from_utf8(value.to_vec()).map_err(io::Error::other))
                        .transpose()?
                } else {
                    None
                }
            } else {
                None
            };
            Ok(Row {
                pid,
                ppid,
                start_ticks,
                alive,
                token,
            })
        };
        match inspect() {
            Ok(row) => rows.push(row),
            Err(error) if gone(&error) => (),
            Err(error) => {
                if unknown.is_none() {
                    unknown = Some(io::Error::new(
                        error.kind(),
                        format!("scope census process {pid} is unreadable: {error}"),
                    ));
                }
            }
        }
    }
    Ok((rows, unknown))
}

/// Processes whose token `owned` accepts, and every descendant still parented below one.
fn members(rows: &[Row], owned: impl Fn(&str) -> bool) -> impl Iterator<Item = &Row> {
    let mut pids: BTreeSet<u32> = rows
        .iter()
        .filter(|row| row.token.as_deref().is_some_and(&owned))
        .map(|row| row.pid)
        .collect();
    loop {
        let before = pids.len();
        for row in rows {
            if pids.contains(&row.ppid) {
                pids.insert(row.pid);
            }
        }
        if pids.len() == before {
            break;
        }
    }
    rows.iter().filter(move |row| pids.contains(&row.pid))
}

/// A token of this namespace that no live scope of this process holds.
fn unclaimed(namespace: &str) -> impl Fn(&str) -> bool + '_ {
    move |token| {
        token
            .strip_prefix(namespace)
            .and_then(|rest| rest.strip_prefix('-'))
            .is_some_and(|id| !id.is_empty() && !id.contains('-'))
            && !LIVE.lock().unwrap().contains(token)
    }
}

/// Kill every member until a census finds none alive, then wait for each to exit. Its
/// parent reaps it: an escaped one's is the supervisor.
fn end_members(owned: impl Fn(&str) -> bool) -> io::Result<usize> {
    let mut signaled: BTreeSet<(u32, u64)> = BTreeSet::new();
    let mut killed: Vec<Exact> = Vec::new();
    let mut exiting = Vec::new();
    loop {
        let rows = census()?;
        let fresh: Vec<&Row> = members(&rows, &owned)
            .filter(|row| row.alive && !signaled.contains(&(row.pid, row.start_ticks)))
            .collect();
        if fresh.is_empty() {
            break;
        }
        for row in fresh {
            signaled.insert((row.pid, row.start_ticks));
            match row.exact()? {
                Some(exact) => {
                    exact.kill()?;
                    killed.push(exact);
                }
                // A leader already exiting has no pidfd: its exit is sampled below.
                None => exiting.push(process_birth(row.pid)),
            }
        }
    }
    for exact in &killed {
        exact.wait()?;
    }
    for birth in exiting.into_iter().flatten() {
        // The sample period is how often the exit is looked for, never a deadline.
        while !process_ended(&birth)? {
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    Ok(signaled.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};

    fn id() -> String {
        uuid::Uuid::new_v4().simple().to_string()[..16].to_string()
    }

    /// The leader detaches a daemon into its own session, then exits; the daemon's pid.
    fn detach(scope: &Scope) -> u32 {
        let mut command = Command::new("/bin/sh");
        command
            .args([
                "-c",
                "read go; setsid sleep 1000 </dev/null >/dev/null 2>&1 & echo $!",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped());
        if let Some((name, value)) = scope.environment() {
            command.env(name, value);
        }
        let mut leader = command.spawn().unwrap();
        scope.adopt(leader.id()).unwrap();
        leader.stdin.take().unwrap().write_all(b"go\n").unwrap();
        let mut daemon = String::new();
        leader
            .stdout
            .take()
            .unwrap()
            .read_to_string(&mut daemon)
            .unwrap();
        assert!(leader.wait().unwrap().success());
        daemon.trim().parse().unwrap()
    }

    /// A token scope even where a cgroup could be delegated.
    fn tree(namespace: &str, live: bool) -> Scope {
        let token = format!("{namespace}-{}", id());
        if live {
            LIVE.lock().unwrap().insert(token.clone());
        }
        Scope(Backend::Tree { token })
    }

    #[test]
    fn a_recovered_token_observes_a_setsid_reader_after_its_leader_ends() {
        let scope = tree("strict", true);
        let reader = process_birth(detach(&scope)).unwrap();
        struct EndReader(ProcessBirth);
        impl Drop for EndReader {
            fn drop(&mut self) {
                if let Ok(Some(exact)) = Exact::open(&self.0) {
                    let _ = exact.kill();
                    let _ = exact.wait();
                }
            }
        }
        let _cleanup = EndReader(reader.clone());
        let recovery = scope.recovery(unsafe { libc::geteuid() });
        let encoded = serde_json::to_vec(&recovery).unwrap();
        let recovered: Recovery = serde_json::from_slice(&encoded).unwrap();
        assert!(!recovered.empty().unwrap());
        scope.end().unwrap();
        assert!(process_ended(&reader).unwrap());
        match recovered.empty() {
            Ok(empty) => assert!(empty),
            Err(error) => eprintln!(
                "strict empty proof unavailable on this host; no release authority: {error}"
            ),
        }
    }

    #[test]
    fn strict_scope_census_refuses_unreadable_relevant_environment() {
        let root = std::env::temp_dir().join(format!("scope-unknown-{}", id()));
        let process = root.join("424242");
        fs::create_dir_all(&process).unwrap();
        let mut fields = vec!["0"; 20];
        fields[0] = "S";
        fields[1] = "1";
        fields[19] = "10";
        fs::write(
            process.join("stat"),
            format!("424242 (fixture) {}", fields.join(" ")),
        )
        .unwrap();
        let uid = unsafe { libc::geteuid() };
        fs::write(
            process.join("status"),
            format!("Uid: {uid} {uid} {uid} {uid}\n"),
        )
        .unwrap();
        // A directory makes the attempted environment read fail even for privileged tests.
        fs::create_dir(process.join("environ")).unwrap();
        assert!(census_strict(&root, uid).is_err());
        // An unrelated foreign UID cannot carry this supported same-UID scope token.
        fs::write(
            process.join("status"),
            format!("Uid: {} {} {} {}\n", uid + 1, uid + 1, uid + 1, uid + 1),
        )
        .unwrap();
        assert_eq!(census_strict(&root, uid).unwrap().len(), 1);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn end_reaches_a_setsid_descendant_under_either_backend() {
        let mut scopes = vec![tree("test", true)];
        match Scope::cgroup("test", &id()).unwrap() {
            Some(cgroup) => scopes.push(cgroup),
            None => eprintln!("no delegated cgroup-v2 here: the token alone is proved"),
        }
        for scope in scopes {
            let daemon = process_birth(detach(&scope)).unwrap();
            // The leader is gone; its setsid daemon is not, and the scope still counts it.
            assert_eq!(scope.processes().unwrap(), 1);
            assert_eq!(scope.end().unwrap(), 1);
            assert!(process_ended(&daemon).unwrap());
            if let Some(relative) = scope.cgroup_relative() {
                assert!(!Path::new(ROOT).join(&relative[1..]).exists());
            }
        }
    }

    #[test]
    fn an_unclaimed_token_of_the_namespace_is_swept_and_others_are_not() {
        let namespace = format!("sweep{}", id());
        let left = process_birth(detach(&tree(&namespace, false))).unwrap();
        let claimed = tree(&namespace, true);
        let held = process_birth(detach(&claimed)).unwrap();
        let other = tree(&format!("other{}", id()), true);
        let foreign = process_birth(detach(&other)).unwrap();
        assert_eq!(Scope::sweep(&namespace).unwrap(), 1);
        assert!(process_ended(&left).unwrap());
        assert!(!process_ended(&held).unwrap() && !process_ended(&foreign).unwrap());
        assert_eq!(claimed.end().unwrap() + other.end().unwrap(), 2);
    }
}
