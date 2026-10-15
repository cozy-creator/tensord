//! Acceptance, attempt supervision, cancellation, progress coalescing, output custody and
//! reconcile. Scheduling policy remains with the machine owner (`service`).
use crate::journal::{
    Artifact, CallMemo, Execution, Invocation, Journal, Outcome, ProcessBirth, ProgressSnapshot,
    ResultRecord, State, SubmissionContext,
};
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
        },
    },
    path::{Component, Path, PathBuf},
    sync::{Arc, Condvar, Mutex, OnceLock, Weak},
    time::Duration,
};
use tensorfs_core::transport::PullCancellation;

/// The most text one package log line keeps.
const PACKAGE_LOG_LINE: usize = 2 << 10;

/// `text` cut to at most `max` bytes on a character boundary, marked when cut.
fn bounded(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let mut end = max - '…'.len_utf8();
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &text[..end])
}

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

pub struct Engine {
    pub root: PathBuf,
    journal: Mutex<Journal>,
    /// Held while a run takes object references, and while unneeded roots are released.
    pub(crate) object_custody: Mutex<()>,
    /// Each supervised attempt's stop.
    active: Mutex<HashMap<String, Stop>>,
    owned: Mutex<HashSet<String>>,
    progress: Mutex<HashMap<String, ProgressSnapshot>>,
    activity: Mutex<u64>,
    activity_changed: Condvar,
    /// What holds a memoizable run's success for its signer (`call_memo`), once configured.
    memo: OnceLock<Weak<crate::objects::Objects>>,
    /// Each preparing run's stop, and whether it has begun what cannot stop (`Stoppable`).
    preparations: Mutex<HashMap<String, (PullCancellation, bool)>>,
}

/// One run's preparation, registered while it lasts: its cancel stops `stop`, which ends the
/// run's downloads, provider fetches and package installs.
pub struct Stoppable {
    engine: Arc<Engine>,
    id: String,
    pub stop: PullCancellation,
}
impl Stoppable {
    /// From here on the run cannot stop (an upload finalizing at its Hub): it refuses
    /// cancellation. False when it was canceled first.
    pub fn commit(&self) -> bool {
        let mut preparations = self.engine.preparations.lock().unwrap();
        let Some((stop, committed)) = preparations.get_mut(&self.id) else {
            return false;
        };
        *committed = !stop.is_cancelled();
        *committed
    }
}
impl Drop for Stoppable {
    fn drop(&mut self) {
        self.engine.preparations.lock().unwrap().remove(&self.id);
    }
}

/// A cancel of a preparing run past its point of no return is refused; any other stops it.
fn stop_preparation(
    preparations: &HashMap<String, (PullCancellation, bool)>,
    id: &str,
    journal: impl FnOnce() -> io::Result<Execution>,
) -> io::Result<Execution> {
    if preparations.get(id).is_some_and(|(_, committed)| *committed) {
        return Err(io::Error::new(
            io::ErrorKind::ResourceBusy,
            "this run's upload is finalizing at its Hub and ends on its own",
        ));
    }
    let record = journal()?;
    if let Some((stop, _)) = preparations.get(id) {
        stop.cancel();
    }
    Ok(record)
}

type Stop = Arc<dyn Fn() -> io::Result<()> + Send + Sync>;

impl Engine {
    pub fn open(root: &Path) -> io::Result<Arc<Self>> {
        fs::create_dir_all(root)?;
        fs::set_permissions(root, fs::Permissions::from_mode(0o700))?;
        for name in ["staging", "results"] {
            fs::create_dir_all(root.join(name))?;
        }
        File::open(root)?.sync_all()?;
        let mut journal = Journal::open(root)?;
        journal.interrupt_preparations()?;
        Ok(Arc::new(Self {
            root: root.to_path_buf(),
            journal: Mutex::new(journal),
            object_custody: Mutex::new(()),
            active: Mutex::new(HashMap::new()),
            owned: Mutex::new(HashSet::new()),
            progress: Mutex::new(HashMap::new()),
            activity: Mutex::new(0),
            activity_changed: Condvar::new(),
            memo: OnceLock::new(),
            preparations: Mutex::new(HashMap::new()),
        }))
    }

    /// Registers run `id`'s preparation until the answer is dropped. A cancel journaled before
    /// it registered stops it at once.
    pub fn stoppable(self: &Arc<Self>, id: &str) -> Stoppable {
        let stop = PullCancellation::default();
        self.preparations.lock().unwrap().insert(id.into(), (stop.clone(), false));
        if self.get(id).is_ok_and(|record| record.cancel_actor.is_some()) {
            stop.cancel();
        }
        Stoppable { engine: self.clone(), id: id.into(), stop }
    }

    /// A memoizable run's success is kept by `objects` from now on (`Objects::keep`).
    pub fn hold_memos(&self, objects: &Arc<crate::objects::Objects>) {
        let _ = self.memo.set(Arc::downgrade(objects));
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

    pub fn bind_preparation(
        &self,
        record: crate::journal::Preparation,
    ) -> io::Result<crate::journal::Preparation> {
        self.journal.lock().unwrap().bind_preparation(record)
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
    /// A line the machine writes to the run's log (`{"level", "text"}`).
    pub fn append_log(&self, id: &str, level: &str, text: &str) -> io::Result<u64> {
        let line = serde_json::to_vec(&serde_json::json!({"level": level, "text": text}))?;
        let progress = self.progress.lock().unwrap();
        let sequence = self
            .journal
            .lock()
            .unwrap()
            .append_log(id, progress.get(id), &line)?;
        drop(progress);
        self.notify_activity();
        Ok(sequence)
    }
    /// One line a run's package logged (`Telemetry.log`), on its root run's log, the one
    /// `run show` displays: a call's lines lead with its function. A line keeps at most
    /// `PACKAGE_LOG_LINE` bytes of text; past the root log's bound the first line becomes one
    /// notice and later ones are dropped (`Journal::append_package_log`).
    pub fn append_package_log(&self, id: &str, level: &str, text: &str) -> io::Result<()> {
        let progress = self.progress.lock().unwrap();
        let mut journal = self.journal.lock().unwrap();
        let record = journal.get(id)?;
        let (root, text) = match record.invocation.root.as_str() {
            "" => (id, text.to_string()),
            root => (root, format!("{}: {text}", record.invocation.entrypoint)),
        };
        let text = bounded(&text, PACKAGE_LOG_LINE);
        let line = serde_json::to_vec(&serde_json::json!({"level": level, "text": text}))?;
        let appended = journal.append_package_log(root, progress.get(root), &line)?;
        drop(journal);
        drop(progress);
        if appended {
            self.notify_activity();
        }
        Ok(())
    }
    /// A root's GPU allocation changed: a `gpu` entry in its log (`gpu_service::note_gpus`).
    pub fn append_gpu(&self, id: &str, entry: &serde_json::Value) -> io::Result<u64> {
        let line = serde_json::to_vec(entry)?;
        let progress = self.progress.lock().unwrap();
        let sequence = self.journal.lock().unwrap().append_log(id, progress.get(id), &line)?;
        drop(progress);
        self.notify_activity();
        Ok(sequence)
    }
    /// A call of the run applied its attention pin (its executor said so).
    pub fn apply_attention(&self, id: &str) -> io::Result<()> {
        self.journal.lock().unwrap().apply_attention(id)
    }
    pub fn logs(&self, id: &str) -> io::Result<Vec<crate::journal::StoredProduct>> {
        self.journal.lock().unwrap().logs(id)
    }
    /// A settled call of the run (`call` names it), recorded in the run's log once.
    pub fn append_call(&self, id: &str, call: &str, body: &[u8]) -> io::Result<Option<u64>> {
        let progress = self.progress.lock().unwrap();
        let sequence = self
            .journal
            .lock()
            .unwrap()
            .append_call(id, progress.get(id), call, body)?;
        drop(progress);
        if sequence.is_some() {
            self.notify_activity();
        }
        Ok(sequence)
    }
    pub fn calls(&self, id: &str) -> io::Result<Vec<crate::journal::StoredProduct>> {
        self.journal.lock().unwrap().calls(id)
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

    /// The preparing run names its code and models and becomes dispatchable, or completes with
    /// the memoized result that answered it (`answer`). Its observation ends under the lock a
    /// dispatcher claims under: a claim made the moment the run is queued keeps its ownership
    /// (else `reconcile` takes the attempt for an orphan).
    pub fn bind_prepared(
        &self,
        id: &str,
        invocation: Invocation,
        preparation: &str,
        memo: CallMemo,
        answer: Option<ResultRecord>,
    ) -> io::Result<Execution> {
        let mut owned = self.owned.lock().unwrap();
        let mut progress = self.progress.lock().unwrap();
        let record = self
            .journal
            .lock()
            .unwrap()
            .bind_prepared(id, invocation, preparation, memo, answer);
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
    pub fn list(&self) -> io::Result<Vec<Execution>> {
        let owned = self.owned.lock().unwrap();
        let progress = self.progress.lock().unwrap();
        let mut records = self.journal.lock().unwrap().list()?;
        for record in &mut records {
            overlay_observation(record, &owned, &progress);
        }
        Ok(records)
    }
    /// See `Journal::jobs`.
    pub fn jobs(&self, since: i64) -> io::Result<(bool, i64)> {
        self.journal.lock().unwrap().jobs(since)
    }
    pub fn supervising(&self) -> usize {
        self.owned.lock().unwrap().len()
    }
    /// Whether this machine run supervises `id`.
    pub fn owns(&self, id: &str) -> bool {
        self.owned.lock().unwrap().contains(id)
    }

    /// See `Journal::fail_unstarted`.
    pub fn fail_unstarted(&self, id: &str, code: &str, reason: impl Into<String>) -> io::Result<Execution> {
        let failure = crate::journal::Failure::machine(code, reason);
        let record = self.journal.lock().unwrap().fail_unstarted(id, failure)?;
        self.notify_activity();
        Ok(record)
    }

    /// Every run ready to dispatch, by its root's acceptance, then its own.
    pub fn ready(&self) -> io::Result<Vec<Execution>> {
        self.journal.lock().unwrap().ready()
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

    /// Trusted adapters share one claim, supervisor reservation and journal. They
    /// register the exact process birth before authorizing any authored code.
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
            drop(owned);
            self.fail_unstarted(&id, "executor_launch_failed", format!("supervisor launch failed: {error}"))?;
            return Err(error);
        }
        Ok(true)
    }

    /// The durable actor record precedes delivery. Transport loss never calls this.
    pub fn cancel(&self, id: &str, actor: &str) -> io::Result<Execution> {
        let record = {
            let preparations = self.preparations.lock().unwrap();
            let owned = self.owned.lock().unwrap();
            let mut progress = self.progress.lock().unwrap();
            let mut record = stop_preparation(&preparations, id, || {
                self.journal.lock().unwrap().cancel_observed(id, actor, progress.get(id))
            })?;
            progress.remove(id);
            overlay_observation(&mut record, &owned, &progress);
            record
        };
        self.deliver_cancellation(&record);
        Ok(record)
    }

    /// Explicit native cancellation also fences an actor/ID not yet accepted by Run.
    pub fn cancel_run(&self, actor: &str, id: &str) -> io::Result<Execution> {
        let record = {
            let preparations = self.preparations.lock().unwrap();
            let owned = self.owned.lock().unwrap();
            let mut progress = self.progress.lock().unwrap();
            let mut journal = self.journal.lock().unwrap();
            let prior = journal.reserve_run_cancellation(actor, id)?;
            let mut record = stop_preparation(&preparations, &prior.id, || {
                journal.cancel_observed(&prior.id, actor, progress.get(&prior.id))
            })?;
            progress.remove(&prior.id);
            overlay_observation(&mut record, &owned, &progress);
            record
        };
        self.deliver_cancellation(&record);
        Ok(record)
    }

    fn deliver_cancellation(&self, record: &Execution) {
        self.notify_activity();
        if record.state.terminal() {
            return;
        }
        let id = &record.id;
        let stop = self.active.lock().unwrap().get(id).cloned();
        if let Some(stop) = stop {
            // Delivery failure does not revoke the already committed cancellation authority.
            let _ = stop();
        }
    }

    /// An attempt that ends unauthorized or stopped: CANCELED, as journaled.
    pub(crate) fn finish_stopped(&self, id: &str) -> io::Result<Execution> {
        if self.get(id)?.cancel_actor.is_none() {
            return Err(io::Error::other("the attempt has no stop authority"));
        }
        self.finish(id, Outcome::Canceled)
    }

    pub fn children(&self, parent: &str) -> io::Result<Vec<Execution>> {
        self.journal.lock().unwrap().children(parent)
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
        let held = self.held_memo(id, &outcome);
        let mut progress = self.progress.lock().unwrap();
        let mut journal = self.journal.lock().unwrap();
        if matches!(outcome, Outcome::Completed(_)) {
            // A warning never stands between a run and its result.
            if let Err(error) = journal.warn_unapplied_attention(id, progress.get(id)) {
                eprintln!("run {id}: attention pin warning not logged: {error}");
            }
        }
        let record = journal.finish_observed(id, outcome, progress.get(id), held)?;
        drop(journal);
        if record.state.terminal() {
            progress.remove(id);
        }
        drop(progress);
        self.notify_activity();
        Ok(record)
    }

    /// A memoizable run's success, its files kept for its signer, to hold as it completes: a
    /// later call of the same computation is answered from it the moment this one is done.
    fn held_memo(&self, id: &str, outcome: &Outcome) -> Option<crate::journal::HeldMemo> {
        let (Outcome::Completed(result), Some(objects)) = (outcome, self.memo.get().and_then(Weak::upgrade)) else {
            return None;
        };
        let record = self.get(id).ok().filter(|r| !r.memo.computation.is_empty() && !r.memo.memoized)?;
        objects.keep(&record, result).unwrap_or_else(|error| {
            eprintln!("run {id}: its success is not held for the memo: {error}");
            None
        })
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
        cancel: Stop,
    ) -> io::Result<()> {
        self.journal
            .lock()
            .unwrap()
            .register_process(id, birth.clone())?;
        self.watch_process(birth)?;
        self.active
            .lock()
            .unwrap()
            .insert(id.into(), cancel.clone());
        self.notify_activity();
        let record = self.get(id)?;
        if record.cancel_actor.is_some() {
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
        let executor = self.get(id).ok().and_then(|record| record.executor);
        let result = crate::triage::write(&self.root, facts, executor.as_ref())
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
        // A tree's member files are held beside its manifest, by digest (`tree_binding`).
        for tree in bindings.iter().filter(|b| b.media_type == crate::gpu_service::TREE_MEDIA) {
            let mut manifest = Vec::new();
            open_artifact(source, Path::new(&tree.relative_path))?.read_to_end(&mut manifest)?;
            paths.extend(tree_members(&manifest)?.into_iter().map(|(sha, _)| format!("sha256-{sha}")));
        }
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
                journal.requeue_unstarted(&record.id)?;
            } else if record.cancel_actor.is_some() {
                // A journaled cancel ends it CANCELED, however it stopped.
                journal.finish(&record.id, Outcome::Canceled)?;
            } else {
                journal.finish(&record.id, Outcome::Failed(crate::journal::Failure::machine("machine_stopped", "owner lost before durable result custody; exact executor birth has ended; started work will not be replayed")))?;
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

    /// A held result file. Custody hashed it once as it took it (`custody`); a read serves it.
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
        let file = open_artifact(&self.root, Path::new(&artifact.path))?;
        if file.metadata()?.len() != artifact.length {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "durable result content changed",
            ));
        }
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
/// A tree manifest's members: each file's (sha256 hex, length).
pub(crate) fn tree_members(manifest: &[u8]) -> io::Result<Vec<(String, u64)>> {
    let invalid = || io::Error::new(io::ErrorKind::InvalidData, "tree manifest has an invalid entry");
    let document: Value = serde_json::from_slice(manifest).map_err(|_| invalid())?;
    document["entries"]
        .as_array()
        .ok_or_else(invalid)?
        .iter()
        .map(|entry| {
            let sha = entry["blob"]["sha256"].as_str().filter(|h| h.len() == 64).ok_or_else(invalid)?;
            Ok((sha.to_string(), entry["blob"]["length"].as_u64().ok_or_else(invalid)?))
        })
        .collect()
}

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

    /// Slurm FIFO: an older root's step queued later dispatches before a younger root's
    /// step queued earlier; each child carries its root and GPU family.
    #[test]
    fn ready_runs_queue_by_their_roots_acceptance() {
        let root = std::env::temp_dir().join(format!("cm-ready-{}", uuid::Uuid::new_v4()));
        let engine = Engine::open(&root).unwrap();
        let submit = |key: &str, parent: &str, gpu: bool| engine.submit(key, Invocation {
            degree: gpu as u32, parent: parent.into(), ..Default::default()
        }).unwrap();
        let (older, younger) = (submit("older", "", false), submit("younger", "", false));
        for root in [&older, &younger] {
            assert!(engine.dispatch_managed(&root.id, |_, _| Ok(())).unwrap());
        }
        let younger_step = submit("younger/1", &younger.id, false);
        let job = submit("older/1", &older.id, true);
        let nested = submit("older/1/1", &job.id, false);
        let order: Vec<_> = engine.ready().unwrap().into_iter().map(|r| r.id).collect();
        assert_eq!(order, [job.id.clone(), nested.id.clone(), younger_step.id.clone()]);
        assert_eq!((nested.root_seq(), nested.family()), (older.root_seq(), Some(job.id.as_str())));
        assert_eq!((younger_step.root_seq(), younger_step.family()), (younger.root_seq(), None));
        drop(engine);
        std::fs::remove_dir_all(root).unwrap();
    }

    /// One managed attempt of `id`, as the pool runs it: a real process registered and
    /// authorized, then `settle` with the engine; the process ends with the attempt.
    fn attempt(engine: &Arc<Engine>, id: &str, settle: impl FnOnce(&Engine, &str) + Send + 'static) {
        let (done, finished) = std::sync::mpsc::channel();
        assert!(engine.dispatch_managed(id, move |engine, id| {
            let mut process = std::process::Command::new("/bin/cat")
                .stdin(std::process::Stdio::piped())
                .spawn()?;
            engine.register_managed(&id, process_birth(process.id())?, Arc::new(|| Ok(())))?;
            assert!(engine.authorize_managed(&id, None)?);
            settle(&engine, &id);
            drop(process.stdin.take());
            process.wait()?;
            done.send(()).unwrap();
            Ok(())
        })
        .unwrap());
        finished.recv().unwrap();
    }

    /// A run's attention pin is its calls': a root completing with a pin no call applied says
    /// so once in its log and completes all the same; one a call applied (a child's counts for
    /// its root) completes without it, and a run with no pin logs nothing.
    #[test]
    fn a_pin_no_call_applied_is_a_warning_in_the_runs_log() {
        let root = std::env::temp_dir().join(format!("cm-pin-{}", uuid::Uuid::new_v4()));
        let engine = Engine::open(&root).unwrap();
        let run = |key: &str, pin: &str, applied: bool| {
            let invocation = Invocation { attention_kernel: pin.into(), ..Default::default() };
            let id = engine.submit(key, invocation.clone()).unwrap().id;
            let child = Invocation { parent: id.clone(), ..invocation };
            let child = engine.submit(&format!("{key}-call"), child).unwrap().id;
            attempt(&engine, &id, move |engine, id| {
                if applied {
                    engine.apply_attention(&child).unwrap();
                }
                let value = ResultRecord { value: serde_json::json!({}), artifacts: vec![], asset_bindings: vec![] };
                engine.finish(id, Outcome::Completed(value)).unwrap();
            });
            assert_eq!(engine.get(&id).unwrap().state, State::Completed);
            let logs = engine.logs(&id).unwrap();
            logs.iter().map(|log| serde_json::from_slice::<Value>(&log.product).unwrap()).collect::<Vec<_>>()
        };
        assert_eq!(run("unapplied", "sdpa", false),
            [serde_json::json!({"level": "warning", "text": "attention pin sdpa was never applied"})]);
        assert_eq!(run("applied", "sdpa", true), Vec::<Value>::new());
        assert_eq!(run("unpinned", "", false), Vec::<Value>::new());
        drop(engine);
        std::fs::remove_dir_all(root).unwrap();
    }

    /// A result's files are taken into custody once, from the attempt's spool: a symlink or a
    /// FIFO in its place fails the run, the spool is gone once it settles, and a held file
    /// changed afterwards is refused, never served.
    #[test]
    fn custody_takes_one_verified_copy_and_refuses_escapes_and_mutation() {
        let root = std::env::temp_dir().join(format!("cm-custody-{}", uuid::Uuid::new_v4()));
        let engine = Engine::open(&root).unwrap();
        let binding = |length: u64| crate::journal::AssetBinding {
            relative_path: "prediction".into(),
            asset_ref: "attempt:run/file/0".into(),
            media_type: "text/plain".into(),
            length,
        };
        let run = |key: &str, place: fn(&Path)| {
            let id = engine.submit(key, Invocation::default()).unwrap().id;
            attempt(&engine, &id, move |engine, id| {
                let spool = engine.staging(id).unwrap();
                place(&spool.join("prediction"));
                let _spool = Spool(spool.clone());
                let length = fs::symlink_metadata(spool.join("prediction")).map_or(0, |m| m.len());
                if let Err(error) = engine.managed_result(id, &spool, serde_json::json!({"rows": 2}), vec![binding(length)]) {
                    engine.finish(id, Outcome::Failed(crate::journal::Failure::custody(&error.to_string()))).unwrap();
                }
            });
            engine.get(&id).unwrap()
        };
        let escaped = run("symlink", |path| std::os::unix::fs::symlink("/etc/passwd", path).unwrap());
        assert_eq!(escaped.state, State::Failed);
        let fifo = run("fifo", |path| {
            let name = CString::new(path.as_os_str().as_bytes()).unwrap();
            assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
        });
        assert_eq!(fifo.state, State::Failed);
        let held = run("held", |path| fs::write(path, b"17,39\n").unwrap());
        assert_eq!(held.state, State::Completed, "{held:?}");
        assert!(!root.join("staging").join(&held.id).exists(), "the spool goes once settled");
        let mut bytes = String::new();
        engine.open_result(&held.id, 0).unwrap().read_to_string(&mut bytes).unwrap();
        assert_eq!(bytes, "17,39\n");
        let path = root.join(&held.result.unwrap().artifacts[0].path);
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        fs::write(&path, b"corrupted").unwrap();
        assert_eq!(engine.open_result(&held.id, 0).unwrap_err().kind(), io::ErrorKind::InvalidData);
        drop(engine);
        std::fs::remove_dir_all(root).unwrap();
    }
}
