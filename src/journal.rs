//! Durable acceptance and transitions. Observation has no lifecycle side effects.
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    fs, io,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

// Status observation cursor allocation, not a duration or liveness policy.
const REVISION_WINDOW: u64 = 1 << 32;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Invocation {
    pub package: String,
    pub generation: String,
    pub module: String,
    pub entrypoint: String,
    pub input: Value,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
pub struct SubmissionContext {
    pub actor: String,
    pub request_id: String,
    pub submission_id: String,
    pub expected_workspace_id: String,
    pub capture_digest: String,
    pub invocation_digest: String,
    pub payload_digest: String,
    pub publication_authorization_id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Installation {
    pub actor: String,
    pub alias: String,
    pub generation: String,
    pub package: String,
    pub release: String,
    pub interface: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct PublicTerminal {
    pub outcome: Vec<u8>,
    pub events: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdmissionError {
    WorkspaceMismatch,
    BindingConflict,
    SubmissionClosed,
}
impl std::fmt::Display for AdmissionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::WorkspaceMismatch => {
                "expected workspace differs from this durable machine workspace"
            }
            Self::BindingConflict => {
                "request or submission is already bound to different authored semantics"
            }
            Self::SubmissionClosed => "submission was durably closed before acceptance",
        })
    }
}
impl std::error::Error for AdmissionError {}

fn admission(error: AdmissionError) -> io::Error {
    let kind = match error {
        AdmissionError::WorkspaceMismatch => io::ErrorKind::InvalidInput,
        AdmissionError::BindingConflict => io::ErrorKind::AlreadyExists,
        AdmissionError::SubmissionClosed => io::ErrorKind::PermissionDenied,
    };
    io::Error::new(kind, error)
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum State {
    Queued,
    Starting,
    Running,
    Completed,
    Failed,
    Canceled,
}

impl State {
    pub fn terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Canceled)
    }
    fn name(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Starting => "starting",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Canceled => "canceled",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ProcessBirth {
    pub pid: u32,
    pub boot_id: String,
    pub start_ticks: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Artifact {
    pub name: String,
    pub path: String,
    pub sha256: String,
    pub length: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ResultRecord {
    pub value: Value,
    pub artifacts: Vec<Artifact>,
    #[serde(default)]
    pub asset_bindings: Vec<AssetBinding>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct AssetBinding {
    pub relative_path: String,
    pub asset_ref: String,
    pub media_type: String,
    pub checksum: OutputChecksum,
    pub length: u64,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct OutputChecksum {
    pub algorithm: String,
    pub value: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Execution {
    pub id: String,
    pub idempotency_key: String,
    pub invocation: Invocation,
    #[serde(default)]
    pub submission: Option<SubmissionContext>,
    pub state: State,
    pub revision: u64,
    #[serde(default)]
    pub accepted_at_ms: u64,
    #[serde(default)]
    pub finished_at_ms: u64,
    #[serde(default)]
    pub acceptance_boot_id: String,
    /// Reserved observation cursor ceiling. Older stored records default to no reservation.
    #[serde(default)]
    pub revision_ceiling: u64,
    #[serde(default)]
    pub attempt: u32,
    #[serde(default)]
    pub waiting_reason: Option<String>,
    pub process: Option<ProcessBirth>,
    pub cancel_actor: Option<String>,
    pub completed_units: u64,
    pub progress: Option<String>,
    pub result: Option<ResultRecord>,
    pub failure: Option<String>,
}

/// Coalesced observation; persisted only as part of an authoritative transition.
#[derive(Clone, Debug)]
pub struct ProgressSnapshot {
    pub completed_units: u64,
    pub detail: String,
    pub revision: u64,
}

impl ProgressSnapshot {
    pub fn overlay(&self, record: &mut Execution) -> bool {
        if self.completed_units <= record.completed_units {
            return false;
        }
        record.completed_units = self.completed_units;
        record.progress = Some(self.detail.clone());
        record.revision = record.revision.max(self.revision);
        true
    }
}

pub struct Journal {
    connection: Connection,
    workspace_id: String,
}

fn db_error(error: impl std::fmt::Display) -> io::Error {
    io::Error::other(error.to_string())
}
fn encoded(value: &impl Serialize) -> io::Result<String> {
    serde_json::to_string(value).map_err(db_error)
}
fn timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(i64::MAX as u128) as i64
}

impl Journal {
    pub fn open(root: &Path) -> io::Result<Self> {
        fs::create_dir_all(root)?;
        let connection = Connection::open(root.join("executions.sqlite3")).map_err(db_error)?;
        connection
            .execute_batch(
                "PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS executions (
              id INTEGER PRIMARY KEY AUTOINCREMENT, idempotency_key TEXT NOT NULL UNIQUE,
              invocation TEXT NOT NULL, record TEXT NOT NULL, state TEXT NOT NULL,
              updated_ms INTEGER NOT NULL);
            CREATE INDEX IF NOT EXISTS executions_nonterminal ON executions(id)
              WHERE state IN ('queued','starting','running');
            CREATE INDEX IF NOT EXISTS executions_active ON executions(id)
              WHERE state IN ('starting','running');
            CREATE INDEX IF NOT EXISTS executions_ready ON executions(id)
              WHERE state='queued' AND json_extract(record,'$.waiting_reason') IS NULL;",
            )
            .map_err(db_error)?;
        // Add consumed index columns without rewriting any older private records.
        let columns: Vec<String> = {
            let mut statement = connection
                .prepare("PRAGMA table_info(executions)")
                .map_err(db_error)?;
            let rows = statement
                .query_map([], |row| row.get(1))
                .map_err(db_error)?;
            rows.collect::<Result<_, _>>().map_err(db_error)?
        };
        for column in ["actor", "request_id", "submission_id"] {
            if !columns.iter().any(|existing| existing == column) {
                connection
                    .execute(
                        &format!("ALTER TABLE executions ADD COLUMN {column} TEXT"),
                        [],
                    )
                    .map_err(db_error)?;
            }
        }
        connection.execute_batch("CREATE TABLE IF NOT EXISTS machine_metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS submission_closures(actor TEXT NOT NULL,submission_id TEXT NOT NULL,request_id TEXT NOT NULL,workspace_id TEXT NOT NULL,closed_ms INTEGER NOT NULL,PRIMARY KEY(actor,submission_id));
            CREATE TABLE IF NOT EXISTS installations(actor TEXT NOT NULL,alias TEXT NOT NULL,record TEXT NOT NULL,PRIMARY KEY(actor,alias));
            CREATE TABLE IF NOT EXISTS public_terminals(execution INTEGER PRIMARY KEY REFERENCES executions(id),outcome BLOB NOT NULL,events BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS native_outputs(actor TEXT NOT NULL,owner TEXT NOT NULL,source BLOB NOT NULL,PRIMARY KEY(actor,owner));
            CREATE UNIQUE INDEX IF NOT EXISTS executions_actor_request ON executions(actor,request_id) WHERE actor IS NOT NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS executions_actor_submission ON executions(actor,submission_id) WHERE actor IS NOT NULL;
            CREATE INDEX IF NOT EXISTS executions_actor_order ON executions(actor,id) WHERE actor IS NOT NULL;").map_err(db_error)?;
        let existing: Option<String> = connection
            .query_row(
                "SELECT value FROM machine_metadata WHERE key='workspace_id'",
                [],
                |row| row.get(0),
            )
            .optional()
            .map_err(db_error)?;
        if existing.is_none() {
            connection
                .execute(
                    "INSERT OR IGNORE INTO machine_metadata(key,value) VALUES('workspace_id',?1)",
                    [uuid::Uuid::new_v4().to_string()],
                )
                .map_err(db_error)?;
        }
        let workspace_id = connection
            .query_row(
                "SELECT value FROM machine_metadata WHERE key='workspace_id'",
                [],
                |row| row.get(0),
            )
            .map_err(db_error)?;
        fs::File::open(root)?.sync_all()?;
        Ok(Self {
            connection,
            workspace_id,
        })
    }

    pub fn workspace_id(&self) -> &str {
        &self.workspace_id
    }

    pub fn installation(&self, actor: &str, alias: &str) -> io::Result<Option<Installation>> {
        let record: Option<String> = self
            .connection
            .query_row(
                "SELECT record FROM installations WHERE actor=?1 AND alias=?2",
                params![actor, alias],
                |r| r.get(0),
            )
            .optional()
            .map_err(db_error)?;
        record
            .map(|record| serde_json::from_str(&record).map_err(db_error))
            .transpose()
    }
    pub fn bind_installation(&mut self, record: Installation) -> io::Result<Installation> {
        validate_scope(&record.actor, &record.alias, &record.generation)?;
        if let Some(prior) = self.installation(&record.actor, &record.alias)? {
            if prior != record {
                return Err(admission(AdmissionError::BindingConflict));
            }
            return Ok(prior);
        }
        self.connection
            .execute(
                "INSERT INTO installations(actor,alias,record) VALUES(?1,?2,?3)",
                params![record.actor, record.alias, encoded(&record)?],
            )
            .map_err(db_error)?;
        Ok(record)
    }
    pub fn public_terminal(&self, id: &str) -> io::Result<Option<PublicTerminal>> {
        self.connection
            .query_row(
                "SELECT outcome,events FROM public_terminals WHERE execution=?1",
                [id],
                |row| {
                    Ok(PublicTerminal {
                        outcome: row.get(0)?,
                        events: row.get(1)?,
                    })
                },
            )
            .optional()
            .map_err(db_error)
    }
    pub fn native_output(&self, actor: &str, owner: &str) -> io::Result<Option<Vec<u8>>> {
        self.connection
            .query_row(
                "SELECT source FROM native_outputs WHERE actor=?1 AND owner=?2",
                params![actor, owner],
                |row| row.get(0),
            )
            .optional()
            .map_err(db_error)
    }
    pub fn bind_native_output(
        &mut self,
        actor: &str,
        owner: &str,
        source: &[u8],
    ) -> io::Result<()> {
        if let Some(prior) = self.native_output(actor, owner)? {
            if prior != source {
                return Err(admission(AdmissionError::BindingConflict));
            }
            return Ok(());
        }
        self.connection
            .execute(
                "INSERT INTO native_outputs(actor,owner,source) VALUES(?1,?2,?3)",
                params![actor, owner, source],
            )
            .map_err(db_error)?;
        Ok(())
    }
    pub fn commit_public_terminal(
        &mut self,
        id: &str,
        projection: PublicTerminal,
    ) -> io::Result<PublicTerminal> {
        if !self.get(id)?.state.terminal() {
            return Err(db_error(
                "terminal projection requires durable terminal custody",
            ));
        }
        if let Some(prior) = self.public_terminal(id)? {
            return Ok(prior);
        }
        self.connection
            .execute(
                "INSERT INTO public_terminals(execution,outcome,events) VALUES(?1,?2,?3)",
                params![id, projection.outcome, projection.events],
            )
            .map_err(db_error)?;
        Ok(projection)
    }

    pub fn accept_public(
        &mut self,
        context: SubmissionContext,
        invocation: Invocation,
    ) -> io::Result<Execution> {
        self.accept_public_on_boot(context, invocation, "")
    }
    pub fn accept_public_on_boot(
        &mut self,
        context: SubmissionContext,
        invocation: Invocation,
        boot: &str,
    ) -> io::Result<Execution> {
        self.validate_workspace(&context.expected_workspace_id)?;
        validate_scope(&context.actor, &context.request_id, &context.submission_id)?;
        // An escaped tuple is an unambiguous selector encoding, not a version/fingerprint gate.
        let key = format!(
            "public:{}",
            encoded(&(context.actor.as_str(), context.submission_id.as_str()))?
        );
        self.accept_bound(&key, invocation, Some(context), boot)
    }

    /// A successful return is an acceptance receipt: FULL WAL commit precedes it.
    /// Reusing a key with different semantics fails rather than silently replacing work.
    pub fn accept(&mut self, key: &str, invocation: Invocation) -> io::Result<Execution> {
        if key.is_empty() || key.len() > 512 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "idempotency key must contain 1..512 bytes",
            ));
        }
        self.accept_bound(key, invocation, None, "")
    }

    fn accept_bound(
        &mut self,
        key: &str,
        invocation: Invocation,
        context: Option<SubmissionContext>,
        boot: &str,
    ) -> io::Result<Execution> {
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        let mut prior: Option<String> = tx
            .query_row(
                "SELECT record FROM executions WHERE idempotency_key=?1",
                [key],
                |row| row.get(0),
            )
            .optional()
            .map_err(db_error)?;
        if let Some(context) = &context {
            if prior.is_none() {
                prior = public_prior(
                    &tx,
                    &context.actor,
                    &context.request_id,
                    &context.submission_id,
                )?;
            }
        }
        if let Some(prior) = prior {
            let execution: Execution = serde_json::from_str(&prior).map_err(db_error)?;
            if execution.invocation != invocation || execution.submission != context {
                if context.is_some() {
                    return Err(admission(AdmissionError::BindingConflict));
                }
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "idempotency key already identifies a different invocation",
                ));
            }
            return Ok(execution);
        }
        if let Some(context) = &context {
            let closed: bool = tx.query_row("SELECT EXISTS(SELECT 1 FROM submission_closures WHERE actor=?1 AND submission_id=?2)", params![context.actor,context.submission_id], |row| row.get(0)).map_err(db_error)?;
            if closed {
                return Err(admission(AdmissionError::SubmissionClosed));
            }
        }
        tx.execute("INSERT INTO executions(idempotency_key,invocation,record,state,updated_ms,actor,request_id,submission_id) VALUES(?1,?2,'{}','queued',?3,?4,?5,?6)", params![key, encoded(&invocation)?, timestamp(),context.as_ref().map(|context|context.actor.as_str()),context.as_ref().map(|context|context.request_id.as_str()),context.as_ref().map(|context|context.submission_id.as_str())]).map_err(db_error)?;
        let execution = Execution {
            id: tx.last_insert_rowid().to_string(),
            idempotency_key: key.into(),
            invocation,
            submission: context,
            state: State::Queued,
            revision: 1,
            accepted_at_ms: timestamp().max(0) as u64,
            finished_at_ms: 0,
            acceptance_boot_id: boot.into(),
            revision_ceiling: 1,
            attempt: 0,
            waiting_reason: None,
            process: None,
            cancel_actor: None,
            completed_units: 0,
            progress: None,
            result: None,
            failure: None,
        };
        tx.execute(
            "UPDATE executions SET record=?1 WHERE id=?2",
            params![encoded(&execution)?, execution.id],
        )
        .map_err(db_error)?;
        tx.commit().map_err(db_error)?;
        Ok(execution)
    }

    pub fn get(&self, id: &str) -> io::Result<Execution> {
        let value: Option<String> = self
            .connection
            .query_row("SELECT record FROM executions WHERE id=?1", [id], |row| {
                row.get(0)
            })
            .optional()
            .map_err(db_error)?;
        let value = value
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "execution does not exist"))?;
        serde_json::from_str(&value).map_err(db_error)
    }

    pub fn get_public(&self, actor: &str, request_id: &str) -> io::Result<Execution> {
        let value: Option<String> = self
            .connection
            .query_row(
                "SELECT record FROM executions WHERE actor=?1 AND request_id=?2",
                params![actor, request_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(db_error)?;
        serde_json::from_str(&value.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "public execution does not exist for this actor",
            )
        })?)
        .map_err(db_error)
    }

    pub fn list_actor(&self, actor: &str, limit: usize) -> io::Result<Vec<Execution>> {
        let mut statement = self
            .connection
            .prepare("SELECT record FROM executions WHERE actor=?1 ORDER BY id LIMIT ?2")
            .map_err(db_error)?;
        let records = statement
            .query_map(params![actor, limit.min(i64::MAX as usize) as i64], |row| {
                row.get::<_, String>(0)
            })
            .map_err(db_error)?;
        records
            .map(|record| serde_json::from_str(&record.map_err(db_error)?).map_err(db_error))
            .collect()
    }

    /// Closure is a durable acceptance tombstone, never implicit execution cancellation.
    pub fn close_submission(
        &mut self,
        actor: &str,
        submission_id: &str,
        request_id: &str,
        expected_workspace_id: &str,
    ) -> io::Result<Option<Execution>> {
        self.validate_workspace(expected_workspace_id)?;
        validate_scope(actor, request_id, submission_id)?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        let existing = public_prior(&tx, actor, request_id, submission_id)?;
        let existing: Option<Execution> = existing
            .map(|record| serde_json::from_str(&record).map_err(db_error))
            .transpose()?;
        if let Some(record) = &existing {
            let context = record
                .submission
                .as_ref()
                .ok_or_else(|| admission(AdmissionError::BindingConflict))?;
            if context.request_id != request_id || context.submission_id != submission_id {
                return Err(admission(AdmissionError::BindingConflict));
            }
        }
        let prior: Option<String> = tx
            .query_row(
                "SELECT request_id FROM submission_closures WHERE actor=?1 AND submission_id=?2",
                params![actor, submission_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(db_error)?;
        if prior.is_some_and(|prior| prior != request_id) {
            return Err(admission(AdmissionError::BindingConflict));
        }
        tx.execute("INSERT OR IGNORE INTO submission_closures(actor,submission_id,request_id,workspace_id,closed_ms) VALUES(?1,?2,?3,?4,?5)", params![actor,submission_id,request_id,expected_workspace_id,timestamp()]).map_err(db_error)?;
        tx.commit().map_err(db_error)?;
        Ok(existing)
    }

    fn validate_workspace(&self, expected: &str) -> io::Result<()> {
        if expected != self.workspace_id {
            return Err(admission(AdmissionError::WorkspaceMismatch));
        }
        Ok(())
    }

    pub fn list(&self) -> io::Result<Vec<Execution>> {
        let mut statement = self
            .connection
            .prepare("SELECT record FROM executions ORDER BY id")
            .map_err(db_error)?;
        let records = statement
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(db_error)?;
        records
            .map(|record| serde_json::from_str(&record.map_err(db_error)?).map_err(db_error))
            .collect()
    }

    pub fn nonterminal(&self, limit: usize) -> io::Result<Vec<Execution>> {
        self.selected("SELECT record FROM executions WHERE state IN ('queued','starting','running') ORDER BY id LIMIT ?1", limit)
    }

    pub fn active(&self, limit: usize) -> io::Result<Vec<Execution>> {
        self.selected("SELECT record FROM executions WHERE state IN ('starting','running') ORDER BY id LIMIT ?1", limit)
    }

    pub fn ready(&self, limit: usize) -> io::Result<Vec<Execution>> {
        self.selected("SELECT record FROM executions WHERE state='queued' AND json_extract(record,'$.waiting_reason') IS NULL ORDER BY id LIMIT ?1", limit)
    }

    fn selected(&self, query: &str, limit: usize) -> io::Result<Vec<Execution>> {
        let mut statement = self.connection.prepare(query).map_err(db_error)?;
        let records = statement
            .query_map([limit.min(i64::MAX as usize) as i64], |row| {
                row.get::<_, String>(0)
            })
            .map_err(db_error)?;
        records
            .map(|record| serde_json::from_str(&record.map_err(db_error)?).map_err(db_error))
            .collect()
    }

    fn update(
        &mut self,
        id: &str,
        change: impl FnOnce(&mut Execution) -> io::Result<bool>,
    ) -> io::Result<Execution> {
        self.update_observed(id, None, change)
    }

    fn update_observed(
        &mut self,
        id: &str,
        progress: Option<&ProgressSnapshot>,
        change: impl FnOnce(&mut Execution) -> io::Result<bool>,
    ) -> io::Result<Execution> {
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        let record: Option<String> = tx
            .query_row("SELECT record FROM executions WHERE id=?1", [id], |row| {
                row.get(0)
            })
            .optional()
            .map_err(db_error)?;
        let mut record: Execution =
            serde_json::from_str(&record.ok_or_else(|| {
                io::Error::new(io::ErrorKind::NotFound, "execution does not exist")
            })?)
            .map_err(db_error)?;
        let was_terminal = record.state.terminal();
        let changed = change(&mut record)?;
        let observed =
            !was_terminal && progress.is_some_and(|progress| progress.overlay(&mut record));
        if changed || observed {
            // Every durable mutation jumps past all volatile cursors the former owner
            // could have published, even when its progress snapshot was lost.
            record.revision = record
                .revision
                .max(record.revision_ceiling)
                .checked_add(1)
                .ok_or_else(|| db_error("execution observation cursor exhausted"))?;
            record.revision_ceiling = if record.state == State::Running {
                record
                    .revision
                    .checked_add(REVISION_WINDOW)
                    .ok_or_else(|| db_error("execution observation cursor exhausted"))?
            } else {
                record.revision
            };
            tx.execute(
                "UPDATE executions SET record=?1,state=?2,updated_ms=?3 WHERE id=?4",
                params![encoded(&record)?, record.state.name(), timestamp(), id],
            )
            .map_err(db_error)?;
        }
        tx.commit().map_err(db_error)?;
        Ok(record)
    }

    /// Only one dispatcher can claim a never-started attempt.
    pub fn wait_for_environment(
        &mut self,
        id: &str,
        reason: Option<String>,
    ) -> io::Result<Execution> {
        self.update(id, |record| {
            if record.state != State::Queued || record.waiting_reason == reason {
                return Ok(false);
            }
            record.waiting_reason = reason;
            Ok(true)
        })
    }

    /// Only one dispatcher can claim a never-started attempt.
    pub fn claim(&mut self, id: &str) -> io::Result<bool> {
        let mut claimed = false;
        self.update(id, |record| {
            if record.state != State::Queued {
                return Ok(false);
            }
            record.state = State::Starting;
            record.attempt += 1;
            record.waiting_reason = None;
            claimed = true;
            Ok(true)
        })?;
        Ok(claimed)
    }

    /// Retry is legal only when no start authorization could have reached package code.
    /// Dispatcher calls this after exact termination (or before any spawn).
    pub fn defer_unstarted(&mut self, id: &str, reason: String) -> io::Result<Execution> {
        self.update(id, |record| {
            if record.state != State::Starting {
                return Err(db_error(
                    "only a never-authorized attempt may return to queued",
                ));
            }
            record.process = None;
            record.waiting_reason = Some(reason);
            record.state = if record.cancel_actor.is_some() {
                State::Canceled
            } else {
                State::Queued
            };
            Ok(true)
        })
    }

    /// Called while the runner is blocked awaiting Invoke, before package execution.
    pub fn register_process(&mut self, id: &str, process: ProcessBirth) -> io::Result<Execution> {
        self.update(id, |record| {
            if record.state != State::Starting || record.process.is_some() {
                return Err(db_error("attempt is not awaiting its first executor"));
            }
            record.process = Some(process);
            Ok(true)
        })
    }

    pub fn running(&mut self, id: &str) -> io::Result<Execution> {
        self.update(id, |record| {
            if record.state != State::Starting || record.process.is_none() {
                return Err(db_error("attempt has no registered executor"));
            }
            if record.cancel_actor.is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "canceled before package start authorization",
                ));
            }
            record.state = State::Running;
            Ok(true)
        })
    }

    /// Reserve another cursor window only when the previous one is exhausted.
    /// This rare allocation commit is not a per-event telemetry write.
    pub fn reserve_observations(
        &mut self,
        id: &str,
        progress: Option<&ProgressSnapshot>,
    ) -> io::Result<Execution> {
        self.update_observed(id, progress, |record| {
            if record.state != State::Running {
                return Err(db_error(
                    "only running executions reserve observation cursors",
                ));
            }
            Ok(true)
        })
    }

    pub fn cancel(&mut self, id: &str, actor: &str) -> io::Result<Execution> {
        self.cancel_observed(id, actor, None)
    }

    pub fn cancel_observed(
        &mut self,
        id: &str,
        actor: &str,
        progress: Option<&ProgressSnapshot>,
    ) -> io::Result<Execution> {
        if actor.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cancellation requires an actor",
            ));
        }
        self.update_observed(id, progress, |record| {
            if record.state.terminal() || record.cancel_actor.is_some() {
                return Ok(false);
            }
            record.cancel_actor = Some(actor.into());
            if record.state == State::Queued {
                record.state = State::Canceled;
                record.finished_at_ms = timestamp().max(0) as u64;
            }
            Ok(true)
        })
    }

    /// Caller must prove process termination and durable artifact custody first.
    pub fn finish(&mut self, id: &str, outcome: Outcome) -> io::Result<Execution> {
        self.finish_observed(id, outcome, None)
    }

    pub fn finish_observed(
        &mut self,
        id: &str,
        outcome: Outcome,
        progress: Option<&ProgressSnapshot>,
    ) -> io::Result<Execution> {
        self.update_observed(id, progress, |record| {
            if record.state.terminal() {
                return Ok(false);
            }
            if record.state == State::Queued {
                return Err(db_error("unstarted attempt cannot finish"));
            }
            match outcome {
                Outcome::Completed(result) => {
                    record.state = State::Completed;
                    record.result = Some(result);
                }
                Outcome::Failed(reason) => {
                    record.state = State::Failed;
                    record.failure = Some(reason);
                }
                Outcome::Canceled => {
                    if record.cancel_actor.is_none() {
                        return Err(db_error("runner cannot cancel without durable authority"));
                    }
                    record.state = State::Canceled;
                }
            }
            record.finished_at_ms = timestamp().max(0) as u64;
            Ok(true)
        })
    }
}

pub enum Outcome {
    Completed(ResultRecord),
    Failed(String),
    Canceled,
}

fn validate_scope(actor: &str, request_id: &str, submission_id: &str) -> io::Result<()> {
    if [actor, request_id, submission_id]
        .iter()
        .any(|value| value.is_empty() || value.len() > 512)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "actor, request and submission identities must contain 1..512 bytes",
        ));
    }
    Ok(())
}

fn public_prior(
    tx: &rusqlite::Transaction<'_>,
    actor: &str,
    request_id: &str,
    submission_id: &str,
) -> io::Result<Option<String>> {
    let request: Option<String> = tx
        .query_row(
            "SELECT record FROM executions WHERE actor=?1 AND request_id=?2",
            params![actor, request_id],
            |row| row.get(0),
        )
        .optional()
        .map_err(db_error)?;
    if request.is_some() {
        return Ok(request);
    }
    tx.query_row(
        "SELECT record FROM executions WHERE actor=?1 AND submission_id=?2",
        params![actor, submission_id],
        |row| row.get(0),
    )
    .optional()
    .map_err(db_error)
}
