//! CPU runner supervision. Scheduling policy remains with the machine owner.
use crate::journal::{
    Artifact, Execution, Invocation, Journal, Outcome, ProcessBirth, ProgressSnapshot,
    ResultRecord, State, SubmissionContext,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{HashMap, HashSet},
    ffi::CString,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::{
            ffi::OsStrExt,
            fs::{OpenOptionsExt, PermissionsExt},
            net::UnixStream,
            process::CommandExt,
        },
    },
    path::{Component, Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Condvar, Mutex},
    time::Duration,
};

pub const MAX_RUNNER_FRAME: usize = 1024 * 1024;

impl crate::native_inputs::IntakeJournal for Engine {
    fn begin_intake(
        &self,
        spec: crate::native_inputs::IntakeSpec,
    ) -> io::Result<crate::native_inputs::IntakeState> {
        self.journal.lock().unwrap().begin_intake(spec)
    }
    fn finish_intake(
        &self,
        actor: &str,
        retention: &str,
        receipt: Vec<u8>,
    ) -> io::Result<crate::native_inputs::IntakeState> {
        self.journal
            .lock()
            .unwrap()
            .settle_intake(actor, retention, Some(receipt), false)
    }
    fn abort_intake(
        &self,
        actor: &str,
        retention: &str,
    ) -> io::Result<crate::native_inputs::IntakeState> {
        self.journal
            .lock()
            .unwrap()
            .settle_intake(actor, retention, None, true)
    }
}

#[derive(Clone, Debug)]
pub struct RunnerConfig {
    /// Trusted immutable generation interpreter, resolved by the package installer.
    pub python: PathBuf,
    pub module: String,
    pub import_paths: Vec<PathBuf>,
    /// Already acquired shared environment-generation hold from the installer.
    /// The runner also holds its generation independently before package imports.
    pub generation_hold: Option<Arc<File>>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RunnerCommand<'a> {
    Invoke {
        execution_id: &'a str,
        #[serde(flatten)]
        invocation: &'a Invocation,
        output_root: &'a Path,
    },
    Cancel {
        execution_id: &'a str,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RunnerEvent {
    Ready {
        pid: u32,
        #[serde(default)]
        capabilities: Vec<String>,
    },
    Progress {
        execution_id: String,
        completed_units: u64,
        detail: String,
    },
    Result {
        execution_id: String,
        value: Value,
        #[serde(default)]
        artifacts: Vec<String>,
        #[serde(default)]
        asset_bindings: Vec<crate::journal::AssetBinding>,
    },
    Failed {
        execution_id: String,
        code: String,
        detail: String,
    },
    Canceled {
        execution_id: String,
    },
    #[serde(other)]
    Unknown,
}

pub fn write_command(stream: &mut UnixStream, command: &RunnerCommand<'_>) -> io::Result<()> {
    let frame = serde_json::to_vec(command)?;
    if frame.len() > MAX_RUNNER_FRAME {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "runner command exceeds frame limit",
        ));
    }
    stream.write_all(&(frame.len() as u32).to_be_bytes())?;
    stream.write_all(&frame)
}

pub fn read_event(stream: &mut UnixStream) -> io::Result<Option<RunnerEvent>> {
    let mut header = [0; 4];
    if stream.read(&mut header[..1])? == 0 {
        return Ok(None);
    }
    stream.read_exact(&mut header[1..])?;
    let length = u32::from_be_bytes(header) as usize;
    if length == 0 || length > MAX_RUNNER_FRAME {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid runner frame length",
        ));
    }
    let mut frame = vec![0; length];
    stream.read_exact(&mut frame)?;
    serde_json::from_slice(&frame)
        .map(Some)
        .map_err(io::Error::other)
}

type Writer = Arc<Mutex<UnixStream>>;

pub struct Engine {
    pub root: PathBuf,
    journal: Mutex<Journal>,
    active: Mutex<HashMap<String, Writer>>,
    owned: Mutex<HashSet<String>>,
    progress: Mutex<HashMap<String, ProgressSnapshot>>,
    activity: Mutex<u64>,
    activity_changed: Condvar,
}

impl Engine {
    pub fn open(root: &Path) -> io::Result<Arc<Self>> {
        fs::create_dir_all(root)?;
        fs::set_permissions(root, fs::Permissions::from_mode(0o700))?;
        for name in ["staging", "results"] {
            fs::create_dir_all(root.join(name))?;
        }
        File::open(root)?.sync_all()?;
        Ok(Arc::new(Self {
            root: root.to_path_buf(),
            journal: Mutex::new(Journal::open(root)?),
            active: Mutex::new(HashMap::new()),
            owned: Mutex::new(HashSet::new()),
            progress: Mutex::new(HashMap::new()),
            activity: Mutex::new(0),
            activity_changed: Condvar::new(),
        }))
    }

    pub fn submit(&self, key: &str, invocation: Invocation) -> io::Result<Execution> {
        let owned = self.owned.lock().unwrap();
        let progress = self.progress.lock().unwrap();
        let mut record = self.journal.lock().unwrap().accept(key, invocation)?;
        overlay_observation(&mut record, &owned, &progress);
        drop(progress);
        drop(owned);
        self.notify_activity();
        Ok(record)
    }
    pub fn get(&self, id: &str) -> io::Result<Execution> {
        let owned = self.owned.lock().unwrap();
        let progress = self.progress.lock().unwrap();
        let mut record = self.journal.lock().unwrap().get(id)?;
        overlay_observation(&mut record, &owned, &progress);
        Ok(record)
    }
    pub fn workspace_id(&self) -> String {
        self.journal.lock().unwrap().workspace_id().into()
    }
    pub fn installation(
        &self,
        actor: &str,
        alias: &str,
    ) -> io::Result<Option<crate::journal::Installation>> {
        self.journal.lock().unwrap().installation(actor, alias)
    }
    pub fn installation_for_generation(
        &self,
        actor: &str,
        generation: &str,
    ) -> io::Result<Option<crate::journal::Installation>> {
        self.journal
            .lock()
            .unwrap()
            .installation_for_generation(actor, generation)
    }
    pub fn installations(&self, actor: &str) -> io::Result<Vec<crate::journal::Installation>> {
        self.journal.lock().unwrap().installations(actor)
    }
    pub fn bind_installation(
        &self,
        record: crate::journal::Installation,
    ) -> io::Result<crate::journal::Installation> {
        self.journal.lock().unwrap().bind_installation(record)
    }
    pub fn public_terminal(&self, id: &str) -> io::Result<Option<crate::journal::PublicTerminal>> {
        self.journal.lock().unwrap().public_terminal(id)
    }
    pub fn acknowledge_collection(&self, id: &str) -> io::Result<Execution> {
        self.journal.lock().unwrap().acknowledge_collection(id)
    }
    pub fn native_output(&self, actor: &str, owner: &str) -> io::Result<Option<Vec<u8>>> {
        self.journal.lock().unwrap().native_output(actor, owner)
    }
    pub fn native_owner(&self, owner: &str) -> io::Result<Option<String>> {
        self.journal.lock().unwrap().native_owner(owner)
    }
    pub fn bind_native_output(&self, actor: &str, owner: &str, source: &[u8]) -> io::Result<()> {
        self.journal
            .lock()
            .unwrap()
            .bind_native_output(actor, owner, source)
    }
    pub fn commit_public_terminal(
        &self,
        id: &str,
        record: crate::journal::PublicTerminal,
    ) -> io::Result<crate::journal::PublicTerminal> {
        self.journal
            .lock()
            .unwrap()
            .commit_public_terminal(id, record)
    }

    pub fn submit_public(
        &self,
        context: SubmissionContext,
        invocation: Invocation,
    ) -> io::Result<Execution> {
        self.submit_public_on_boot(context, invocation, "")
    }
    pub fn submit_public_on_boot(
        &self,
        context: SubmissionContext,
        invocation: Invocation,
        boot: &str,
    ) -> io::Result<Execution> {
        let owned = self.owned.lock().unwrap();
        let progress = self.progress.lock().unwrap();
        let mut record = self
            .journal
            .lock()
            .unwrap()
            .accept_public_on_boot(context, invocation, boot)?;
        overlay_observation(&mut record, &owned, &progress);
        drop(progress);
        drop(owned);
        self.notify_activity();
        Ok(record)
    }

    pub fn get_public(&self, actor: &str, request_id: &str) -> io::Result<Execution> {
        let owned = self.owned.lock().unwrap();
        let progress = self.progress.lock().unwrap();
        let mut record = self.journal.lock().unwrap().get_public(actor, request_id)?;
        overlay_observation(&mut record, &owned, &progress);
        Ok(record)
    }

    pub fn list_actor(&self, actor: &str, limit: usize) -> io::Result<Vec<Execution>> {
        self.select(|journal| journal.list_actor(actor, limit))
    }

    pub fn actor_page(
        &self,
        actor: &str,
        after: u64,
        before: u64,
        newest: bool,
        states: &[String],
        limit: usize,
    ) -> io::Result<Vec<Execution>> {
        self.select(|journal| journal.actor_page(actor, after, before, newest, states, limit))
    }
    pub fn actor_head(&self, actor: &str) -> io::Result<u64> {
        self.journal.lock().unwrap().actor_head(actor)
    }
    pub fn close_submission(
        &self,
        actor: &str,
        submission_id: &str,
        request_id: &str,
        expected_workspace_id: &str,
    ) -> io::Result<Option<Execution>> {
        let owned = self.owned.lock().unwrap();
        let progress = self.progress.lock().unwrap();
        let mut record = self.journal.lock().unwrap().close_submission(
            actor,
            submission_id,
            request_id,
            expected_workspace_id,
        )?;
        if let Some(record) = &mut record {
            overlay_observation(record, &owned, &progress);
        }
        drop(progress);
        drop(owned);
        self.notify_activity();
        Ok(record)
    }

    pub fn list(&self) -> io::Result<Vec<Execution>> {
        let owned = self.owned.lock().unwrap();
        let progress = self.progress.lock().unwrap();
        let mut records = self.journal.lock().unwrap().list()?;
        for record in &mut records {
            overlay_observation(record, &owned, &progress);
        }
        Ok(records)
    }
    pub fn supervising(&self) -> usize {
        self.owned.lock().unwrap().len()
    }

    pub fn wait_for_environment(&self, id: &str, reason: Option<String>) -> io::Result<Execution> {
        let record = self
            .journal
            .lock()
            .unwrap()
            .wait_for_environment(id, reason)?;
        self.notify_activity();
        Ok(record)
    }

    pub fn ready(&self, limit: usize) -> io::Result<Vec<Execution>> {
        self.select(|journal| journal.ready(limit))
    }
    pub fn nonterminal(&self, limit: usize) -> io::Result<Vec<Execution>> {
        self.select(|journal| journal.nonterminal(limit))
    }
    pub fn active(&self, limit: usize) -> io::Result<Vec<Execution>> {
        self.select(|journal| journal.active(limit))
    }
    fn select(
        &self,
        query: impl FnOnce(&Journal) -> io::Result<Vec<Execution>>,
    ) -> io::Result<Vec<Execution>> {
        let owned = self.owned.lock().unwrap();
        let progress = self.progress.lock().unwrap();
        let mut records = query(&self.journal.lock().unwrap())?;
        for record in &mut records {
            overlay_observation(record, &owned, &progress);
        }
        Ok(records)
    }

    /// In-process wake cursor, separate from durable execution/status revisions.
    pub fn activity_epoch(&self) -> u64 {
        *self.activity.lock().unwrap()
    }

    /// Optional duration bounds observation only; it never changes execution state.
    pub fn wait_activity(&self, observed: u64, wait: Option<Duration>) -> u64 {
        let epoch = self.activity.lock().unwrap();
        if *epoch != observed || *epoch == u64::MAX {
            return *epoch;
        }
        let epoch = match wait {
            Some(wait) => {
                self.activity_changed
                    .wait_timeout_while(epoch, wait, |epoch| *epoch == observed)
                    .unwrap()
                    .0
            }
            None => self
                .activity_changed
                .wait_while(epoch, |epoch| *epoch == observed)
                .unwrap(),
        };
        *epoch
    }

    pub fn notify_activity(&self) {
        let mut epoch = self.activity.lock().unwrap();
        *epoch = epoch.saturating_add(1); // never wrap or reuse a wake cursor
        self.activity_changed.notify_all();
    }

    /// Dispatch only after the owner's admission and trusted generation resolution.
    pub fn dispatch(self: &Arc<Self>, id: &str, config: RunnerConfig) -> io::Result<bool> {
        let mut owned = self.owned.lock().unwrap();
        if !self.journal.lock().unwrap().claim(id)? {
            return Ok(false);
        }
        owned.insert(id.into());
        self.notify_activity();
        let engine = self.clone();
        let id = id.to_owned();
        let thread_id = id.clone();
        let launched = std::thread::Builder::new()
            .name(format!("execution-{id}"))
            .spawn(move || {
                if let Err(error) = engine.run(&thread_id, config) {
                    eprintln!("execution {thread_id}: {error}");
                }
                engine.progress.lock().unwrap().remove(&thread_id);
                engine.owned.lock().unwrap().remove(&thread_id);
                engine.notify_activity();
            });
        if let Err(error) = launched {
            owned.remove(&id);
            self.journal
                .lock()
                .unwrap()
                .defer_unstarted(&id, format!("supervisor launch failed: {error}"))?;
            self.notify_activity();
            return Err(error);
        }
        Ok(true)
    }

    /// The durable actor record precedes delivery. Transport loss never calls this.
    pub fn cancel(&self, id: &str, actor: &str) -> io::Result<Execution> {
        let record = {
            let owned = self.owned.lock().unwrap();
            let mut progress = self.progress.lock().unwrap();
            let mut record =
                self.journal
                    .lock()
                    .unwrap()
                    .cancel_observed(id, actor, progress.get(id))?;
            progress.remove(id);
            overlay_observation(&mut record, &owned, &progress);
            record
        };
        self.notify_activity();
        let writer = self.active.lock().unwrap().get(id).cloned();
        if let Some(writer) = writer {
            // Delivery failure does not revoke the already committed cancellation authority.
            let _ = write_command(
                &mut writer.lock().unwrap(),
                &RunnerCommand::Cancel { execution_id: id },
            );
        }
        Ok(record)
    }

    fn observe_progress(
        &self,
        id: &str,
        completed_units: u64,
        mut detail: String,
    ) -> io::Result<()> {
        // This map has at most one entry per supervised execution, never an event history.
        if !self.owned.lock().unwrap().contains(id) {
            return Ok(());
        }
        const MAX_DETAIL_BYTES: usize = 2048;
        if detail.len() > MAX_DETAIL_BYTES {
            let mut end = MAX_DETAIL_BYTES;
            while !detail.is_char_boundary(end) {
                end -= 1;
            }
            detail.truncate(end);
        }
        let mut progress = self.progress.lock().unwrap();
        let mut journal = self.journal.lock().unwrap();
        let mut record = journal.get(id)?;
        if record.state.terminal() {
            return Ok(());
        }
        if let Some(snapshot) = progress.get(id) {
            snapshot.overlay(&mut record);
        }
        if completed_units <= record.completed_units {
            return Ok(());
        }
        if record.revision >= record.revision_ceiling {
            record = journal.reserve_observations(id, progress.get(id))?;
        }
        progress.insert(
            id.into(),
            ProgressSnapshot {
                completed_units,
                detail,
                revision: record
                    .revision
                    .checked_add(1)
                    .ok_or_else(|| io::Error::other("execution observation cursor exhausted"))?,
            },
        );
        Ok(())
    }

    fn finish(&self, id: &str, outcome: Outcome) -> io::Result<Execution> {
        let mut progress = self.progress.lock().unwrap();
        let record = self
            .journal
            .lock()
            .unwrap()
            .finish_observed(id, outcome, progress.get(id))?;
        if record.state.terminal() {
            progress.remove(id);
        }
        drop(progress);
        self.notify_activity();
        Ok(record)
    }

    /// Reconcile a previous service incarnation without adopting or repeating authored work.
    /// A live/unknown process birth keeps its obligation charged and nonterminal.
    pub fn reconcile(&self) -> io::Result<Vec<Execution>> {
        let owned = self.owned.lock().unwrap();
        let mut journal = self.journal.lock().unwrap();
        let mut changed = false;
        for record in journal.active(usize::MAX)? {
            if !matches!(record.state, State::Starting | State::Running)
                || owned.contains(&record.id)
            {
                continue;
            }
            let ended = match &record.process {
                Some(birth) => match process_ended(birth) {
                    Ok(ended) => ended,
                    Err(error) => {
                        eprintln!(
                            "execution {}: cannot prove executor termination: {error}",
                            record.id
                        );
                        continue;
                    }
                },
                None => true,
            };
            if !ended {
                continue;
            }
            if record.state == State::Starting {
                journal.defer_unstarted(
                    &record.id,
                    "owner restarted before start authorization; no authored work dispatched"
                        .into(),
                )?;
            } else {
                journal.finish(&record.id, Outcome::Failed("owner lost before durable result custody; exact executor birth has ended; started work will not be replayed".into()))?;
            }
            changed = true;
        }
        drop(journal);
        drop(owned);
        if changed {
            self.notify_activity();
        }
        self.nonterminal(1024)
    }

    fn run(&self, id: &str, config: RunnerConfig) -> io::Result<()> {
        let _generation_hold = config.generation_hold.clone();
        let record = self.get(id)?;
        if record.cancel_actor.is_some() {
            self.finish(id, Outcome::Canceled)?;
            return Ok(());
        }
        let output_root = self.root.join("staging").join(id);
        let launch = (|| {
            fs::create_dir_all(&output_root)?;
            let (parent, runner) = UnixStream::pair()?;
            let stdout = File::create(output_root.join("stdout.log"))?;
            let stderr = File::create(output_root.join("stderr.log"))?;
            let child = spawn_runner(config, &runner, stdout, stderr)?;
            drop(runner);
            Ok::<_, io::Error>((child, parent))
        })();
        let (mut child, mut reader) = match launch {
            Ok(value) => value,
            Err(error) => {
                self.journal
                    .lock()
                    .unwrap()
                    .defer_unstarted(id, format!("runner launch failed: {error}"))?;
                self.notify_activity();
                return Ok(());
            }
        };
        let supervised = self.supervise(
            id,
            &record.invocation,
            &output_root,
            &mut child,
            &mut reader,
        );
        // Closing both socket directions is a cooperative EOF signal, not a process kill.
        self.active.lock().unwrap().remove(id);
        let _ = reader.shutdown(std::net::Shutdown::Both);
        drop(reader);
        let status = child.wait()?;
        match supervised {
            Ok(terminal) => {
                let outcome = match terminal {
                    RunnerEvent::Result {
                        value,
                        artifacts,
                        asset_bindings,
                        ..
                    } if status.success() => {
                        match self.custody(id, &output_root, value, artifacts) {
                            Ok(mut result) => {
                                let mut identities = HashSet::new();
                                for binding in &asset_bindings {
                                    if !identities.insert(&binding.asset_ref)
                                        || !result.artifacts.iter().any(|a| {
                                            a.name == binding.relative_path
                                                && a.length == binding.length
                                        })
                                    {
                                        return self.finish(id, Outcome::Failed("output binding does not name one held artifact".into())).map(|_| ());
                                    }
                                }
                                result.asset_bindings = asset_bindings;
                                Outcome::Completed(result)
                            }
                            Err(error) => {
                                Outcome::Failed(format!("result custody failed: {error}"))
                            }
                        }
                    }
                    RunnerEvent::Result { .. } => {
                        Outcome::Failed(format!("executor reported result but exited {status}"))
                    }
                    RunnerEvent::Failed { code, detail, .. } => {
                        Outcome::Failed(format!("{code}: {detail}"))
                    }
                    RunnerEvent::Canceled { .. } if self.get(id)?.cancel_actor.is_some() => {
                        Outcome::Canceled
                    }
                    RunnerEvent::Canceled { .. } => Outcome::Failed(
                        "runner canceled without durable cancellation authority".into(),
                    ),
                    _ => Outcome::Failed("runner ended without a terminal result".into()),
                };
                self.finish(id, outcome)?;
            }
            Err(error) => {
                if self.get(id)?.state == State::Starting {
                    self.journal.lock().unwrap().defer_unstarted(
                        id,
                        format!("runner ended before start authorization: {error}"),
                    )?;
                    self.notify_activity();
                } else {
                    self.finish(
                        id,
                        Outcome::Failed(format!("executor ended {status}: {error}")),
                    )?;
                }
            }
        }
        Ok(())
    }

    fn supervise(
        &self,
        id: &str,
        invocation: &Invocation,
        output_root: &Path,
        child: &mut Child,
        reader: &mut UnixStream,
    ) -> io::Result<RunnerEvent> {
        // Register identity immediately after spawn, before any package authorization.
        let birth = process_birth(child.id())?;
        self.journal.lock().unwrap().register_process(id, birth)?;
        self.notify_activity();
        let ready =
            read_event(reader)?.ok_or_else(|| io::Error::other("runner EOF before Ready"))?;
        match ready {
            RunnerEvent::Ready { pid, capabilities }
                if pid == child.id()
                    && capabilities.iter().any(|cap| cap == "runtime.author-cpu/1") => {}
            _ => {
                return Err(io::Error::other(
                    "runner did not offer CPU author capability for its actual PID",
                ))
            }
        }
        let writer = Arc::new(Mutex::new(reader.try_clone()?));
        {
            let mut stream = writer.lock().unwrap();
            // Running means authorization may have arrived, including a lost write ack.
            self.journal.lock().unwrap().running(id)?;
            self.notify_activity();
            write_command(
                &mut stream,
                &RunnerCommand::Invoke {
                    execution_id: id,
                    invocation,
                    output_root,
                },
            )?;
            self.active
                .lock()
                .unwrap()
                .insert(id.into(), writer.clone());
        }
        if self.get(id)?.cancel_actor.is_some() {
            write_command(
                &mut writer.lock().unwrap(),
                &RunnerCommand::Cancel { execution_id: id },
            )?;
        }
        loop {
            let event = read_event(reader)?
                .ok_or_else(|| io::Error::other("runner EOF before terminal result"))?;
            let event_id = match &event {
                RunnerEvent::Progress { execution_id, .. }
                | RunnerEvent::Result { execution_id, .. }
                | RunnerEvent::Failed { execution_id, .. }
                | RunnerEvent::Canceled { execution_id } => execution_id,
                RunnerEvent::Unknown => continue,
                RunnerEvent::Ready { .. } => {
                    return Err(io::Error::other("duplicate runner Ready"))
                }
            };
            if event_id != id {
                return Err(io::Error::other(
                    "runner event identifies another execution",
                ));
            }
            match event {
                RunnerEvent::Progress {
                    completed_units,
                    detail,
                    ..
                } => self.observe_progress(id, completed_units, detail)?,
                terminal => return Ok(terminal),
            }
        }
    }

    fn custody(
        &self,
        id: &str,
        output_root: &Path,
        value: Value,
        paths: Vec<String>,
    ) -> io::Result<ResultRecord> {
        let destination = self.root.join("results").join(id);
        fs::create_dir_all(&destination)?;
        let mut artifacts = Vec::new();
        for (index, name) in paths.into_iter().enumerate() {
            let mut source = open_artifact(output_root, Path::new(&name))?;
            let temporary = destination.join(format!("{index}.pending"));
            let mut output = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&temporary)?;
            let mut hash = tensorfs_core::sha256::Sha256::new();
            let mut length = 0;
            let mut buffer = [0; 64 * 1024];
            loop {
                let count = source.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                output.write_all(&buffer[..count])?;
                hash.update(&buffer[..count]);
                length += count as u64;
            }
            let digest = tensorfs_core::sha256::hex(&hash.finish());
            output.sync_all()?;
            output.set_permissions(fs::Permissions::from_mode(0o400))?;
            output.sync_all()?;
            let filename = format!("{index}-{digest}");
            fs::rename(&temporary, destination.join(&filename))?;
            artifacts.push(Artifact {
                name,
                path: format!("results/{id}/{filename}"),
                sha256: digest,
                length,
            });
        }
        File::open(&destination)?.sync_all()?;
        File::open(self.root.join("results"))?.sync_all()?;
        Ok(ResultRecord {
            value,
            artifacts,
            asset_bindings: vec![],
        })
    }

    /// Reads validate content identity; same-UID package execution is not a security sandbox.
    pub fn open_result(&self, id: &str, index: usize) -> io::Result<File> {
        let record = self.get(id)?;
        let artifact = record
            .result
            .and_then(|result| result.artifacts.get(index).cloned())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    "durable result artifact is unavailable",
                )
            })?;
        let mut file = open_artifact(&self.root, Path::new(&artifact.path))?;
        let mut hash = tensorfs_core::sha256::Sha256::new();
        let mut length = 0;
        let mut buffer = [0; 64 * 1024];
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            hash.update(&buffer[..count]);
            length += count as u64;
        }
        if length != artifact.length
            || tensorfs_core::sha256::hex(&hash.finish()) != artifact.sha256
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "durable result content changed",
            ));
        }
        use std::io::Seek;
        file.rewind()?;
        Ok(file)
    }
}

fn overlay_observation(
    record: &mut Execution,
    owned: &HashSet<String>,
    progress: &HashMap<String, ProgressSnapshot>,
) {
    if record.state.terminal() {
        return;
    }
    if !owned.contains(&record.id) && record.state == State::Running {
        // A new owner cannot reproduce lost volatile events. Its snapshot advances to
        // the old reservation ceiling so clients' numeric cursors never move backward.
        record.revision = record.revision.max(record.revision_ceiling);
    }
    if let Some(snapshot) = progress.get(&record.id) {
        snapshot.overlay(record);
    }
}

fn spawn_runner(
    config: RunnerConfig,
    socket: &UnixStream,
    stdout: File,
    stderr: File,
) -> io::Result<Child> {
    let fd = socket.as_raw_fd();
    let mut command = Command::new(config.python);
    command
        .arg("-m")
        .arg(config.module)
        .arg("--execution-fd")
        .arg(fd.to_string());
    if !config.import_paths.is_empty() {
        // PYTHONPATH is import-path configuration, never an execution-mode switch.
        command.env(
            "PYTHONPATH",
            std::env::join_paths(config.import_paths).map_err(io::Error::other)?,
        );
    }
    command
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));
    // SAFETY: fcntl is async-signal-safe; only this owned socket becomes inheritable.
    unsafe {
        command.pre_exec(move || {
            if libc::fcntl(fd, libc::F_SETFD, 0) < 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
    command.spawn()
}

pub fn process_birth(pid: u32) -> io::Result<ProcessBirth> {
    let (start_ticks, _) = process_stat(pid)?;
    Ok(ProcessBirth {
        pid,
        boot_id: fs::read_to_string("/proc/sys/kernel/random/boot_id")?
            .trim()
            .into(),
        start_ticks,
    })
}

pub fn process_ended(birth: &ProcessBirth) -> io::Result<bool> {
    if fs::read_to_string("/proc/sys/kernel/random/boot_id")?.trim() != birth.boot_id {
        return Ok(true);
    }
    match process_stat(birth.pid) {
        Ok((start, state)) => Ok(start != birth.start_ticks || state == 'Z' || state == 'X'),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(true),
        Err(error) => Err(error),
    }
}

fn process_stat(pid: u32) -> io::Result<(u64, char)> {
    let record = fs::read_to_string(format!("/proc/{pid}/stat"))?;
    let (_, suffix) = record
        .rsplit_once(") ")
        .ok_or_else(|| io::Error::other("invalid process stat"))?;
    let fields: Vec<_> = suffix.split_whitespace().collect();
    let start = fields
        .get(19)
        .ok_or_else(|| io::Error::other("process stat lacks birth"))?
        .parse()
        .map_err(io::Error::other)?;
    let state = fields
        .first()
        .and_then(|value| value.chars().next())
        .ok_or_else(|| io::Error::other("process stat lacks state"))?;
    Ok((start, state))
}

/// Walk relative components using openat+NOFOLLOW: no symlink or parent escape races.
pub(crate) fn open_artifact(root: &Path, path: &Path) -> io::Result<File> {
    let parts: Vec<_> = path
        .components()
        .map(|component| match component {
            Component::Normal(name) => Ok(name),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "artifact path must contain only relative normal components",
            )),
        })
        .collect::<io::Result<_>>()?;
    if parts.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "empty artifact path",
        ));
    }
    let mut directory = File::open(root)?;
    for (index, name) in parts.iter().enumerate() {
        let name = CString::new(name.as_bytes()).map_err(io::Error::other)?;
        let final_component = index + 1 == parts.len();
        // O_PATH inspects the final inode without opening devices or blocking on FIFOs.
        let flags = libc::O_CLOEXEC
            | libc::O_NOFOLLOW
            | if final_component {
                libc::O_PATH
            } else {
                libc::O_RDONLY | libc::O_DIRECTORY
            };
        // SAFETY: live directory descriptor and NUL-terminated component, no pointer outputs.
        let fd = unsafe { libc::openat(directory.as_raw_fd(), name.as_ptr(), flags) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: openat returned a new owned descriptor.
        let file = unsafe { File::from_raw_fd(fd) };
        if final_component {
            if !file.metadata()?.is_file() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "artifact is not a regular file",
                ));
            }
            // Reopen this exact, retained regular inode through our descriptor, not its name.
            return File::open(format!("/proc/self/fd/{}", file.as_raw_fd()));
        }
        directory = file;
    }
    unreachable!()
}
