//! CPU runner supervision. Scheduling policy remains with the machine owner.
use crate::{
    journal::{
        Artifact, Execution, Invocation, Journal, Outcome, ProcessBirth, ProgressSnapshot,
        ResultRecord, State, SubmissionContext,
    },
    launch_identity::Seal,
    process::Liveness,
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
    process::{Child, Stdio},
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
    /// Measured at install: why a runner cannot start in this environment.
    pub unavailable: Option<String>,
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
    /// Scopes runner JIT caches to this machine run.
    incarnation: String,
    journal: Mutex<Journal>,
    /// Held while a run takes object references, and while unneeded roots are released.
    pub(crate) object_custody: Mutex<()>,
    active: Mutex<HashMap<String, ActiveRun>>,
    owned: Mutex<HashSet<String>>,
    progress: Mutex<HashMap<String, ProgressSnapshot>>,
    activity: Mutex<u64>,
    activity_changed: Condvar,
}

#[derive(Clone)]
enum ActiveRun {
    Protocol(Writer, Arc<crate::process::Watch>),
    Managed(Arc<dyn Fn() -> io::Result<()> + Send + Sync>),
}

impl Engine {
    pub fn open(root: &Path) -> io::Result<Arc<Self>> {
        fs::create_dir_all(root)?;
        fs::set_permissions(root, fs::Permissions::from_mode(0o700))?;
        for name in ["staging", "results"] {
            fs::create_dir_all(root.join(name))?;
        }
        File::open(root)?.sync_all()?;
        let incarnation = uuid::Uuid::new_v4().simple().to_string();
        crate::launch_identity::remove_stale_jit(&root.join("seal"), &incarnation);
        let mut journal = Journal::open(root)?;
        journal.interrupt_preparations()?;
        Ok(Arc::new(Self {
            root: root.to_path_buf(),
            incarnation,
            journal: Mutex::new(journal),
            object_custody: Mutex::new(()),
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
    pub fn bound_generations(&self) -> io::Result<std::collections::HashSet<String>> {
        self.journal.lock().unwrap().bound_generations()
    }
    pub fn installations(&self, actor: &str) -> io::Result<Vec<crate::journal::Installation>> {
        self.journal.lock().unwrap().installations(actor)
    }
    /// Short journal reads and writes that need no execution state transition.
    pub fn with_journal<T>(&self, f: impl FnOnce(&mut Journal) -> io::Result<T>) -> io::Result<T> {
        f(&mut self.journal.lock().unwrap())
    }
    pub fn bind_installation(
        &self,
        record: crate::journal::Installation,
    ) -> io::Result<crate::journal::Installation> {
        self.journal.lock().unwrap().bind_installation(record)
    }

    pub fn preparation(
        &self,
        actor: &str,
        id: &str,
    ) -> io::Result<Option<crate::journal::Preparation>> {
        self.journal.lock().unwrap().preparation(actor, id)
    }

    pub fn recent_preparations(
        &self,
        limit: usize,
    ) -> io::Result<Vec<crate::journal::Preparation>> {
        self.journal.lock().unwrap().recent_preparations(limit)
    }

    pub fn bind_preparation(
        &self,
        record: crate::journal::Preparation,
    ) -> io::Result<crate::journal::Preparation> {
        self.journal.lock().unwrap().bind_preparation(record)
    }

    pub(crate) fn defer_managed(&self, id: &str, reason: String) -> io::Result<()> {
        self.journal.lock().unwrap().defer_unstarted(id, reason)?;
        self.notify_activity();
        Ok(())
    }
    /// The attempt stays with its dispatcher, awaiting a fresh executor; false if a cancel
    /// ended it.
    pub(crate) fn redeliver(&self, id: &str) -> io::Result<bool> {
        let record = self.journal.lock().unwrap().redeliver(id)?;
        self.notify_activity();
        Ok(record.state == State::Starting)
    }
    pub fn public_terminal(&self, id: &str) -> io::Result<Option<crate::journal::PublicTerminal>> {
        self.journal.lock().unwrap().public_terminal(id)
    }
    pub fn acknowledge_collection(&self, id: &str) -> io::Result<Execution> {
        self.journal.lock().unwrap().acknowledge_collection(id)
    }
    pub fn acknowledge_collection_events(
        &self,
        id: &str,
        events: Option<&[u8]>,
    ) -> io::Result<Execution> {
        let record = self
            .journal
            .lock()
            .unwrap()
            .acknowledge_collection_events(id, events)?;
        // Released by the client, and the store holds its products: the custody copy goes.
        if events.is_some() && record.collected && self.public_terminal(id)?.is_some() {
            drop(Spool(self.root.join("results").join(id)));
        }
        Ok(record)
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
    /// Journal one product. `decide` sees the run's earlier products and returns the product
    /// to append, or `None` when it would change nothing (sequence 0).
    pub fn append_product(
        &self,
        id: &str,
        decide: impl FnOnce(&[crate::journal::StoredProduct]) -> io::Result<Option<Vec<u8>>>,
    ) -> io::Result<u64> {
        let progress = self.progress.lock().unwrap();
        let mut journal = self.journal.lock().unwrap();
        let Some(product) = decide(&journal.products(id)?)? else {
            return Ok(0);
        };
        let sequence = journal.append_product(id, progress.get(id), &product)?;
        drop(journal);
        drop(progress);
        self.notify_activity();
        Ok(sequence)
    }
    pub fn products(&self, id: &str) -> io::Result<Vec<crate::journal::StoredProduct>> {
        self.journal.lock().unwrap().products(id)
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

    /// A run accepted before its preparation; true when new. Its preparation's progress is
    /// observed like a supervised attempt's until it is prepared or ends.
    pub fn accept_run(
        &self,
        actor: &str,
        id: &str,
        digest: &str,
        invocation: Invocation,
    ) -> io::Result<(Execution, bool)> {
        self.accept_run_objects(actor, id, digest, invocation, &[])
    }

    pub fn accept_run_objects(
        &self,
        actor: &str,
        id: &str,
        digest: &str,
        invocation: Invocation,
        objects: &[tensorfs_core::ids::ObjectRef],
    ) -> io::Result<(Execution, bool)> {
        let mut owned = self.owned.lock().unwrap();
        let (record, new) = self
            .journal
            .lock()
            .unwrap()
            .accept_run_objects(actor, id, digest, invocation, objects)?;
        if new {
            owned.insert(record.id.clone());
        }
        drop(owned);
        self.notify_activity();
        Ok((record, new))
    }

    /// The preparing run names its code and models and becomes dispatchable. Its observation
    /// ends under the lock a dispatcher claims under: a claim made the moment the run is
    /// queued keeps its ownership (else `reconcile` takes the attempt for an orphan).
    pub fn bind_prepared(
        &self,
        id: &str,
        invocation: Invocation,
        preparation: &str,
    ) -> io::Result<Execution> {
        let mut owned = self.owned.lock().unwrap();
        let mut progress = self.progress.lock().unwrap();
        let record = self
            .journal
            .lock()
            .unwrap()
            .bind_prepared(id, invocation, preparation);
        progress.remove(id);
        owned.remove(id);
        drop((progress, owned));
        self.notify_activity();
        record
    }

    /// The preparing run ends undispatched (warm success or preparation failure).
    pub fn end_preparation(&self, id: &str, outcome: Outcome) -> io::Result<Execution> {
        let record = self.journal.lock().unwrap().end_preparation(id, outcome);
        self.end_observation(id);
        record
    }

    fn end_observation(&self, id: &str) {
        self.progress.lock().unwrap().remove(id);
        self.owned.lock().unwrap().remove(id);
        self.notify_activity();
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

    pub fn ready_after(&self, after: u64, limit: usize) -> io::Result<Vec<Execution>> {
        self.journal.lock().unwrap().ready_after(after, limit)
    }
    pub(crate) fn gpu_births_after(
        &self,
        after: u64,
        limit: usize,
    ) -> io::Result<Vec<(u64, ProcessBirth)>> {
        self.journal.lock().unwrap().gpu_births_after(after, limit)
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
        self.dispatch_managed(id, move |engine, id| engine.run(&id, config))
    }

    /// Trusted adapters share one claim, supervisor reservation and journal. They
    /// register the exact process birth before authorizing any authored code.
    ///
    /// A run rests paused before its stopped attempt has ended (its executor still exits,
    /// its thread still cleans up), and a resume queues it at once. Its next attempt is not
    /// claimed until that thread is done, which wakes the dispatcher: otherwise the old
    /// thread's cleanup would disown the new attempt and end its children.
    pub(crate) fn dispatch_managed<F>(self: &Arc<Self>, id: &str, run: F) -> io::Result<bool>
    where
        F: FnOnce(Arc<Self>, String) -> io::Result<()> + Send + 'static,
    {
        let mut owned = self.owned.lock().unwrap();
        if owned.contains(id) || !self.journal.lock().unwrap().claim(id)? {
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
                if let Err(error) = run(engine.clone(), thread_id.clone()) {
                    eprintln!("execution {thread_id}: {error}");
                }
                if engine
                    .get(&thread_id)
                    .is_ok_and(|record| !matches!(record.state, State::Starting | State::Running))
                {
                    engine.active.lock().unwrap().remove(&thread_id);
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
        if record.state.terminal() {
            return Ok(record);
        }
        let active = self.active.lock().unwrap().get(id).cloned();
        if let Some(active) = active {
            // Delivery failure does not revoke the already committed cancellation authority.
            let _ = match active {
                ActiveRun::Protocol(writer, watch) => {
                    // From now on the runner's frames are the meter: if they stop, it is killed.
                    watch.canceled();
                    write_command(
                        &mut writer.lock().unwrap(),
                        &RunnerCommand::Cancel { execution_id: id },
                    )
                }
                ActiveRun::Managed(cancel) => cancel(),
            };
        }
        Ok(record)
    }

    /// Pause (`Journal::pause`); a started attempt is stopped as a cancel stops it, and the
    /// journaled pause makes its end `paused`.
    pub fn pause(&self, id: &str, actor: &str, unstarted_only: bool) -> io::Result<Execution> {
        let record = {
            let owned = self.owned.lock().unwrap();
            let mut progress = self.progress.lock().unwrap();
            let mut record = self
                .journal
                .lock()
                .unwrap()
                .pause(id, actor, unstarted_only)?;
            overlay_observation(&mut record, &owned, &progress);
            if record.state == State::Paused {
                progress.remove(id);
            }
            record
        };
        self.notify_activity();
        let stopping = matches!(record.state, State::Starting | State::Running)
            && record.pause_actor.is_some()
            && record.cancel_actor.is_none();
        if stopping {
            if let Some(ActiveRun::Managed(stop)) = self.active.lock().unwrap().get(id).cloned() {
                let _ = stop();
            }
        }
        Ok(record)
    }

    pub fn resume(&self, id: &str) -> io::Result<Execution> {
        let record = self.journal.lock().unwrap().resume(id)?;
        self.notify_activity();
        Ok(record)
    }

    /// An attempt that ends unauthorized or stopped: CANCELED, or PAUSED, as journaled.
    pub(crate) fn finish_stopped(&self, id: &str) -> io::Result<Execution> {
        let record = self.get(id)?;
        let outcome = match (&record.cancel_actor, &record.pause_actor) {
            (Some(_), _) => Outcome::Canceled,
            (None, Some(_)) => Outcome::Paused,
            (None, None) => return Err(io::Error::other("the attempt has no stop authority")),
        };
        self.finish(id, outcome)
    }

    pub fn children(&self, parent: &str) -> io::Result<Vec<Execution>> {
        self.journal.lock().unwrap().children(parent)
    }

    pub fn paused(&self, limit: usize) -> io::Result<Vec<Execution>> {
        self.journal.lock().unwrap().paused(limit)
    }

    pub fn declare_checkpoint(
        &self,
        id: &str,
        attempt: u32,
        operation_key: &str,
        logical_key: &str,
        content_digest: &str,
        length: u64,
    ) -> io::Result<(String, bool)> {
        self.journal.lock().unwrap().declare_checkpoint(
            id,
            attempt,
            operation_key,
            logical_key,
            content_digest,
            length,
        )
    }

    pub fn observe_progress(
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
        // Units never regress; a stage or position change without new units still shows.
        if completed_units < record.completed_units
            || completed_units == record.completed_units
                && record.progress.as_deref() == Some(detail.as_str())
        {
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
        drop((progress, journal));
        self.notify_activity(); // wakes event long-polls; nothing durable is written
        Ok(())
    }

    pub(crate) fn finish(&self, id: &str, outcome: Outcome) -> io::Result<Execution> {
        let mut progress = self.progress.lock().unwrap();
        let record = self
            .journal
            .lock()
            .unwrap()
            .finish_observed(id, outcome, progress.get(id))?;
        if record.state.terminal() || record.state == State::Paused {
            progress.remove(id);
        }
        drop(progress);
        self.notify_activity();
        Ok(record)
    }

    pub(crate) fn watch_process(
        self: &Arc<Self>,
        birth: crate::journal::ProcessBirth,
    ) -> io::Result<()> {
        if process_ended(&birth)? {
            return Ok(());
        }
        // Read birth on both sides of pidfd_open so PID reuse cannot watch or
        // reclaim a different process. A kernel exit event only wakes reconciliation.
        let raw = unsafe { libc::syscall(libc::SYS_pidfd_open, birth.pid, 0) } as i32;
        if raw < 0 {
            let error = io::Error::last_os_error();
            return match error.raw_os_error() {
                Some(libc::ESRCH) => {
                    self.notify_activity();
                    Ok(())
                }
                // A leader whose exit is still tearing down (a large GPU context) has no
                // pidfd yet: observe that exit by sampling instead, then wake dispatch.
                Some(libc::EINVAL) => self.watch_by_sampling(birth),
                _ => Err(error),
            };
        }
        let pidfd = unsafe { File::from_raw_fd(raw) };
        if process_ended(&birth)? {
            self.notify_activity();
            return Ok(());
        }
        let engine = self.clone();
        std::thread::Builder::new()
            .name(format!("orphan-{}", birth.pid))
            .spawn(move || {
                let mut item = libc::pollfd {
                    fd: pidfd.as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                };
                loop {
                    let result = unsafe { libc::poll(&mut item, 1, -1) };
                    if result >= 0 {
                        engine.notify_activity();
                        return;
                    }
                    let error = io::Error::last_os_error();
                    if error.kind() != io::ErrorKind::Interrupted {
                        eprintln!("orphan kernel observation: {error}");
                        return;
                    }
                }
            })?;
        Ok(())
    }

    fn watch_by_sampling(self: &Arc<Self>, birth: ProcessBirth) -> io::Result<()> {
        let engine = self.clone();
        std::thread::Builder::new()
            .name(format!("exiting-{}", birth.pid))
            .spawn(move || {
                // The sample period is how often the exit is looked for, never a deadline.
                while !process_ended(&birth).unwrap_or(true) {
                    std::thread::sleep(Duration::from_secs(1));
                }
                engine.notify_activity();
            })?;
        Ok(())
    }

    /// A previous machine's process has no owner channel and is never adopted: kill that
    /// exact birth (and its group) and wake dispatch when its exit is observed.
    pub(crate) fn end_orphan(self: &Arc<Self>, birth: ProcessBirth) -> io::Result<()> {
        if let Some(exact) = crate::process::Exact::open(&birth)? {
            exact.kill()?;
        }
        self.watch_process(birth)
    }

    pub(crate) fn register_managed(
        self: &Arc<Self>,
        id: &str,
        birth: ProcessBirth,
        cancel: Arc<dyn Fn() -> io::Result<()> + Send + Sync>,
    ) -> io::Result<()> {
        self.journal
            .lock()
            .unwrap()
            .register_process(id, birth.clone())?;
        self.watch_process(birth)?;
        self.active
            .lock()
            .unwrap()
            .insert(id.into(), ActiveRun::Managed(cancel.clone()));
        self.notify_activity();
        let record = self.get(id)?;
        if record.cancel_actor.is_some() || record.pause_actor.is_some() {
            let _ = cancel();
        }
        Ok(())
    }

    pub(crate) fn authorize_managed(
        &self,
        id: &str,
        executor: Option<crate::journal::ExecutorFacts>,
    ) -> io::Result<bool> {
        match self.journal.lock().unwrap().running(id, executor) {
            Ok(_) => {
                self.notify_activity();
                Ok(true)
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => Ok(false),
            Err(error) => Err(error),
        }
    }

    /// Write and bind a failed attempt's triage bundle before the run settles. A failure to
    /// keep it is reported, never fatal to the settlement.
    pub fn record_triage(&self, id: &str, facts: &crate::triage::Facts) {
        let result = crate::triage::write(&self.root, facts)
            .and_then(|triage| self.journal.lock().unwrap().bind_triage(id, &triage));
        if let Err(error) = result {
            eprintln!("execution {id}: triage bundle not kept: {error}");
        }
    }
    /// The run's triage bundle bytes and their reference, if it kept one.
    pub fn triage(&self, id: &str) -> io::Result<Option<(crate::triage::TriageRef, Vec<u8>)>> {
        let Some(triage) = self.journal.lock().unwrap().triage(id)? else {
            return Ok(None);
        };
        let bytes = crate::triage::read(&self.root, &triage)?;
        Ok(Some((triage, bytes)))
    }
    pub(crate) fn staging(&self, id: &str) -> io::Result<PathBuf> {
        let root = self.root.join("staging").join(id);
        fs::create_dir_all(&root)?;
        Ok(root)
    }
    pub(crate) fn managed_result(
        &self,
        id: &str,
        source: &Path,
        value: Value,
        bindings: Vec<crate::journal::AssetBinding>,
    ) -> io::Result<Execution> {
        let mut paths: Vec<_> = bindings.iter().map(|b| b.relative_path.clone()).collect();
        paths.sort();
        paths.dedup();
        let result = self.bound_custody(id, source, value, paths, bindings)?;
        self.finish(id, Outcome::Completed(result))
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
            } else if record.invocation.job && record.pause_actor.is_some() {
                // A pausing job's root is replayed by design: it rests paused.
                journal.finish(&record.id, Outcome::Paused)?;
            } else {
                journal.finish(&record.id, Outcome::Failed("owner lost before durable result custody; exact executor birth has ended; started work will not be replayed".into()))?;
            }
            changed = true;
        }
        drop(journal);
        drop(owned);
        if changed {
            self.active.lock().unwrap().retain(|id, _| {
                self.get(id)
                    .is_ok_and(|record| matches!(record.state, State::Starting | State::Running))
            });
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
        if let Some(reason) = &config.unavailable {
            // Known before any launch: fail now rather than start a runner that cannot import.
            self.finish(id, Outcome::Failed(reason.clone()))?;
            return Ok(());
        }
        let output_root = self.root.join("staging").join(id);
        let spool = Spool(output_root.clone());
        let logs = self.root.join("logs");
        let launch = (|| {
            fs::create_dir_all(&output_root)?;
            fs::create_dir_all(&logs)?;
            let seal = Seal::prepare(
                &self.root.join("seal"),
                None,
                &self.incarnation,
                &record.invocation.generation,
                "",
            )?;
            let (parent, runner) = UnixStream::pair()?;
            let stdout = File::create(logs.join(format!("{id}.stdout.log")))?;
            let stderr = File::create(logs.join(format!("{id}.stderr.log")))?;
            let scope = Arc::new(crate::scope::Scope::create(&crate::scope::namespace(
                self.root.parent().unwrap_or(&self.root),
            ))?);
            let child = match spawn_runner(config, &seal, &scope, &runner, stdout, stderr) {
                Ok(child) => child,
                Err(error) => {
                    let _ = scope.end();
                    return Err(error);
                }
            };
            drop(runner);
            let exact = crate::process::Exact::open(&process_birth(child.id())?)?
                .ok_or_else(|| io::Error::other("launched runner has no exact birth"))?
                .with_scope(Some(scope));
            Ok::<_, io::Error>((child, parent, exact))
        })();
        let (mut child, mut reader, exact) = match launch {
            Ok(value) => value,
            Err(error) if crate::process::transient(&error) => {
                self.journal
                    .lock()
                    .unwrap()
                    .defer_unstarted(id, format!("runner launch failed: {error}"))?;
                self.notify_activity();
                return Ok(());
            }
            Err(error) => {
                // Nothing authored ran; a deterministic launch failure is not retried.
                let reason = format!("runner did not start: {error}");
                self.runner_triage(id, 0, &reason, &logs);
                self.finish(id, Outcome::Failed(reason))?;
                return Ok(());
            }
        };
        let supervised = self.supervise(id, &record.invocation, &output_root, &exact, &mut reader);
        // Closing both socket directions is a cooperative EOF signal; the runner is killed
        // only if it then stops making measurable progress without exiting.
        self.active.lock().unwrap().remove(id);
        let _ = reader.shutdown(std::net::Shutdown::Both);
        drop(reader);
        let status = crate::process::reap(&exact, Some(&mut child), Liveness::default())?.status;
        let outcome = match supervised {
            Ok(terminal) => match terminal {
                RunnerEvent::Result {
                    value,
                    artifacts,
                    asset_bindings,
                    ..
                } if status.success() => {
                    match self.bound_custody(id, &output_root, value, artifacts, asset_bindings) {
                        Ok(result) => Outcome::Completed(result),
                        Err(error) => Outcome::Failed(format!("result custody failed: {error}")),
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
                RunnerEvent::Canceled { .. } => {
                    Outcome::Failed("runner canceled without durable cancellation authority".into())
                }
                _ => Outcome::Failed("runner ended without a terminal result".into()),
            },
            Err(error) => {
                let record = self.get(id)?;
                if record.cancel_actor.is_some() {
                    Outcome::Canceled
                } else if record.state == State::Starting {
                    // Ended before authorization: no authored code ran; its exit is the reason.
                    Outcome::Failed(format!(
                        "runner ended before start ({status}): {error}; {}",
                        crate::process::tail(&logs.join(format!("{id}.stderr.log")))
                    ))
                } else {
                    Outcome::Failed(format!("executor ended {status}: {error}"))
                }
            }
        };
        // Custody is complete: a settled run never has a spool.
        drop(spool);
        if let Outcome::Failed(reason) = &outcome {
            self.runner_triage(id, exact.birth.pid, reason, &logs);
        }
        self.finish(id, outcome)?;
        Ok(())
    }

    /// A failed CPU run's bundle: its reason and the end of the runner's stderr.
    fn runner_triage(&self, id: &str, pid: u32, reason: &str, logs: &Path) {
        let Ok(record) = self.get(id) else {
            return;
        };
        let request = record
            .submission
            .as_ref()
            .map_or_else(|| id.to_string(), |s| s.request_id.clone());
        self.record_triage(
            id,
            &crate::triage::Facts {
                request_id: &request,
                attempt: record.attempt,
                terminal: "failed",
                origin: "machine",
                code: "runner_ended",
                message: reason,
                traceback: "",
                executor_pid: pid,
                stderr_tail: &crate::process::tail(&logs.join(format!("{id}.stderr.log"))),
            },
        );
    }

    fn supervise(
        &self,
        id: &str,
        invocation: &Invocation,
        output_root: &Path,
        exact: &crate::process::Exact,
        reader: &mut UnixStream,
    ) -> io::Result<RunnerEvent> {
        // Register identity immediately after spawn, before any package authorization.
        let pid = exact.birth.pid;
        self.journal
            .lock()
            .unwrap()
            .register_process(id, exact.birth.clone())?;
        self.notify_activity();
        let ready =
            read_event(reader)?.ok_or_else(|| io::Error::other("runner EOF before Ready"))?;
        match ready {
            RunnerEvent::Ready {
                pid: ready,
                capabilities,
            } if ready == pid && capabilities.iter().any(|cap| cap == "runtime.author-cpu/1") => {}
            _ => {
                return Err(io::Error::other(
                    "runner did not offer CPU author capability for its actual PID",
                ))
            }
        }
        let writer = Arc::new(Mutex::new(reader.try_clone()?));
        let watching = crate::process::Watching::start(
            exact.try_clone()?,
            crate::process::Meter::Frames,
            Liveness::default(),
            Duration::ZERO,
            None,
            "invocation",
        )?;
        let watch = watching.watch();
        {
            let mut stream = writer.lock().unwrap();
            // Running means authorization may have arrived, including a lost write ack.
            self.journal.lock().unwrap().running(id, None)?;
            self.notify_activity();
            write_command(
                &mut stream,
                &RunnerCommand::Invoke {
                    execution_id: id,
                    invocation,
                    output_root,
                },
            )?;
            self.active.lock().unwrap().insert(
                id.into(),
                ActiveRun::Protocol(writer.clone(), watch.clone()),
            );
        }
        let terminal = self.events(id, reader, &writer, &watch);
        match watching.finish().0 {
            Some(verdict) => Err(io::Error::other(verdict)),
            None => terminal,
        }
    }

    fn events(
        &self,
        id: &str,
        reader: &mut UnixStream,
        writer: &Writer,
        watch: &crate::process::Watch,
    ) -> io::Result<RunnerEvent> {
        if self.get(id)?.cancel_actor.is_some() {
            watch.canceled();
            write_command(
                &mut writer.lock().unwrap(),
                &RunnerCommand::Cancel { execution_id: id },
            )?;
        }
        loop {
            let event = read_event(reader)?
                .ok_or_else(|| io::Error::other("runner EOF before terminal result"))?;
            watch.frame();
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

    fn bound_custody(
        &self,
        id: &str,
        output_root: &Path,
        value: Value,
        paths: Vec<String>,
        bindings: Vec<crate::journal::AssetBinding>,
    ) -> io::Result<ResultRecord> {
        let mut result = self.custody(id, output_root, value, paths)?;
        let mut identities = HashSet::new();
        for binding in &bindings {
            if !identities.insert(&binding.asset_ref)
                || !result
                    .artifacts
                    .iter()
                    .any(|a| a.name == binding.relative_path && a.length == binding.length)
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "output binding does not name one held artifact",
                ));
            }
        }
        result.asset_bindings = bindings;
        Ok(result)
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

/// The CPU runner launches like an executor: through the Runtime trampoline (parent-death,
/// no_new_privs, OOM order, own process group) with the sealed environment and no GPU.
fn spawn_runner(
    config: RunnerConfig,
    seal: &Seal,
    scope: &crate::scope::Scope,
    socket: &UnixStream,
    stdout: File,
    stderr: File,
) -> io::Result<Child> {
    let fd = socket.as_raw_fd();
    let mut configured = std::collections::BTreeMap::new();
    if !config.import_paths.is_empty() {
        // PYTHONPATH is import-path configuration, never an execution-mode switch.
        configured.insert(
            "PYTHONPATH".to_string(),
            std::env::join_paths(config.import_paths)
                .map_err(io::Error::other)?
                .to_string_lossy()
                .into_owned(),
        );
    }
    let mut command = crate::launch_identity::trampoline(&config.python, None, Some(scope))?;
    command
        .arg("-m")
        .arg(config.module)
        .arg("--execution-fd")
        .arg(fd.to_string())
        .env_clear()
        .envs(seal.environment(&configured))
        .envs(scope.environment())
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

pub use crate::process::{process_birth, process_ended};

/// A run's spool: removed once the run is settled, whatever the outcome. Its outputs are in
/// custody by then (results and the store), so the spool would be a third copy.
pub(crate) struct Spool(pub PathBuf);
impl Drop for Spool {
    fn drop(&mut self) {
        // remove_dir_all never follows a symlink the package planted inside.
        if let Err(error) = fs::remove_dir_all(&self.0) {
            if error.kind() != io::ErrorKind::NotFound {
                eprintln!("spool {}: {error}", self.0.display());
            }
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::mpsc, time::Duration};

    #[test]
    fn a_resumed_run_is_not_dispatched_until_its_stopped_attempt_has_ended() {
        let root = std::env::temp_dir().join(format!("cm-resume-{}", uuid::Uuid::new_v4()));
        let engine = Engine::open(&root).unwrap();
        let invocation = Invocation {
            package: "audit/job".into(),
            input: serde_json::json!({}),
            job: true,
            ..Default::default()
        };
        let run = engine.submit("resumed", invocation).unwrap();
        // The first attempt rests paused, then goes on ending until the test lets it.
        let (rested, rests) = mpsc::channel();
        let (end, ends) = mpsc::channel::<()>();
        let first = engine.dispatch_managed(&run.id, move |engine, id| {
            engine.pause(&id, "alice", false)?;
            engine.finish_stopped(&id)?;
            rested.send(()).unwrap();
            ends.recv().unwrap();
            Ok(())
        });
        assert!(first.unwrap());
        rests.recv().unwrap();
        assert_eq!(engine.resume(&run.id).unwrap().state, State::Queued);
        let second = |engine: Arc<Engine>, id: String| {
            engine
                .finish(&id, Outcome::Failed("the second attempt ran".into()))
                .map(drop)
        };
        assert!(
            !engine.dispatch_managed(&run.id, second).unwrap(),
            "the next attempt was claimed while the stopped one was still ending"
        );
        assert_eq!(engine.get(&run.id).unwrap().state, State::Queued);
        // Once it has ended, the run is dispatched, and stays this engine's own.
        end.send(()).unwrap();
        let mut seen = engine.activity_epoch();
        while !engine.dispatch_managed(&run.id, second).unwrap() {
            seen = engine.wait_activity(seen, Some(Duration::from_millis(50)));
        }
        loop {
            engine.reconcile().unwrap();
            let record = engine.get(&run.id).unwrap();
            assert_eq!(record.waiting_reason, None);
            if record.state.terminal() {
                assert_eq!((record.state, record.attempt), (State::Failed, 2));
                break;
            }
            seen = engine.wait_activity(seen, Some(Duration::from_millis(50)));
        }
        let _ = fs::remove_dir_all(root);
    }
}
