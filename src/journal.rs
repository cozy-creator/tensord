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

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct Invocation {
    pub package: String,
    pub generation: String,
    pub module: String,
    pub entrypoint: String,
    pub input: Value,
    /// The caller's attention-kernel pin, passed to the executor verbatim.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub attention_kernel: String,
    /// File inputs the caller imported, verified at acceptance.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<InputFile>,
    /// An `@app.job`: it runs in a deviceless executor and calls children through its seam.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub job: bool,
    /// A child run's parent (its execution id): children end with it.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub parent: String,
}

/// One file input: the declared field path, its exact bytes (sha256 and length, held in the
/// store by the caller's retention) and media type, in the caller's order.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct InputFile {
    pub input_id: String,
    pub digest: String,
    pub length: u64,
    pub media_type: String,
    pub order: u32,
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
    pub preparation_id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Preparation {
    pub actor: String,
    pub id: String,
    pub installation: String,
    pub document: Vec<u8>,
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
        })
    }
}
impl std::error::Error for AdmissionError {}

fn admission(error: AdmissionError) -> io::Error {
    let kind = match error {
        AdmissionError::WorkspaceMismatch => io::ErrorKind::InvalidInput,
        AdmissionError::BindingConflict => io::ErrorKind::AlreadyExists,
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
    /// A job at rest between attempts: neither terminal nor dispatched; `resume` queues it.
    Paused,
    /// Written by a newer machine: listed, never dispatched, settled or overwritten.
    #[serde(other)]
    Unknown,
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
            Self::Paused => "paused",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ExecutorFacts {
    pub pid: u32,
    pub runtime_version: String,
    pub tensorfs_version: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ProcessBirth {
    pub pid: u32,
    pub boot_id: String,
    pub start_ticks: u64,
}

/// Memoized results each signer keeps, its most recently used first.
pub const MAX_HELD_MEMOS: i64 = 4096;

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
    #[serde(default)]
    pub collected: bool,
    /// Reserved observation cursor ceiling. Older stored records default to no reservation.
    #[serde(default)]
    pub revision_ceiling: u64,
    #[serde(default)]
    pub attempt: u32,
    #[serde(default)]
    pub waiting_reason: Option<String>,
    pub process: Option<ProcessBirth>,
    pub cancel_actor: Option<String>,
    /// Explicit cancellation reserved this actor/ID before any Run spec was accepted.
    #[serde(default)]
    pub canceled_before_acceptance: bool,
    /// Who paused it: the run is paused, or pauses once its started attempt stops (or, while
    /// preparing, once prepared). `resume` clears it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pause_actor: Option<String>,
    pub completed_units: u64,
    pub progress: Option<String>,
    /// The cursor at which the attempt started running: its `running` event.
    #[serde(default)]
    pub running_revision: u64,
    /// When the attempt started running (wall clock), and on which executor.
    #[serde(default)]
    pub started_at_ms: u64,
    /// The run's own time running its callable, every ended attempt summed.
    #[serde(default)]
    pub executed_ms: u64,
    #[serde(default)]
    pub executor: Option<ExecutorFacts>,
    pub result: Option<ResultRecord>,
    pub failure: Option<String>,
}

impl Execution {
    /// Its progress is observed: an attempt running, or a run preparing inside itself.
    pub fn observed(&self) -> bool {
        self.state == State::Running
            || (self.state == State::Queued && self.waiting_reason.as_deref() == Some(PREPARING))
    }
}

/// One journaled product of an execution's output log (`api::domain::RunProduct` bytes).
#[derive(Clone, Debug)]
pub struct StoredProduct {
    pub sequence: u64,
    pub at_ms: u64,
    pub product: Vec<u8>,
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
        if self.revision <= record.revision || self.completed_units < record.completed_units {
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

/// The record layout this machine writes. A newer journal still opens: unknown fields are
/// ignored and rows it cannot read are skipped, never a refusal of the whole journal.
pub const JOURNAL_FORMAT: u32 = 1;

/// One unreadable row (written by a newer machine, or damaged) must not hide every other run.
fn readable(rows: Vec<String>) -> Vec<Execution> {
    rows.into_iter()
        .filter_map(|row| match serde_json::from_str(&row) {
            Ok(record) => Some(record),
            Err(error) => {
                eprintln!("journal: skipping an unreadable execution row: {error}");
                None
            }
        })
        .collect()
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
        connection.execute_batch("BEGIN; CREATE TABLE IF NOT EXISTS machine_metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS update_run_ids(actor TEXT NOT NULL,id TEXT NOT NULL,PRIMARY KEY(actor,id));
            CREATE TABLE IF NOT EXISTS installations(actor TEXT NOT NULL,alias TEXT NOT NULL,record TEXT NOT NULL,PRIMARY KEY(actor,alias));
            CREATE TABLE IF NOT EXISTS preparations(actor TEXT NOT NULL,id TEXT NOT NULL,record TEXT NOT NULL,PRIMARY KEY(actor,id));
            CREATE TABLE IF NOT EXISTS warm_sets(actor TEXT NOT NULL,position INTEGER NOT NULL,record TEXT NOT NULL,PRIMARY KEY(actor,position));
            CREATE TABLE IF NOT EXISTS public_terminals(execution INTEGER PRIMARY KEY REFERENCES executions(id),outcome BLOB NOT NULL,events BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS run_products(execution INTEGER NOT NULL REFERENCES executions(id),sequence INTEGER NOT NULL,at_ms INTEGER NOT NULL,product BLOB NOT NULL,PRIMARY KEY(execution,sequence));
            CREATE TABLE IF NOT EXISTS run_measurements(execution INTEGER PRIMARY KEY REFERENCES executions(id),measurements BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS run_logs(execution INTEGER NOT NULL REFERENCES executions(id),sequence INTEGER NOT NULL,at_ms INTEGER NOT NULL,record BLOB NOT NULL,PRIMARY KEY(execution,sequence));
            CREATE TABLE IF NOT EXISTS attention_applied(execution INTEGER PRIMARY KEY REFERENCES executions(id));
            CREATE TABLE IF NOT EXISTS run_calls(execution INTEGER NOT NULL REFERENCES executions(id),sequence INTEGER NOT NULL,at_ms INTEGER NOT NULL,call TEXT NOT NULL,record BLOB NOT NULL,PRIMARY KEY(execution,sequence),UNIQUE(execution,call));
            CREATE TABLE IF NOT EXISTS native_outputs(actor TEXT NOT NULL,owner TEXT NOT NULL,source BLOB NOT NULL,PRIMARY KEY(actor,owner));
            CREATE TABLE IF NOT EXISTS input_intakes(actor TEXT NOT NULL,retention TEXT NOT NULL,record TEXT NOT NULL,PRIMARY KEY(actor,retention));
            CREATE TABLE IF NOT EXISTS hub_access(actor TEXT NOT NULL,origin TEXT NOT NULL,record TEXT NOT NULL,PRIMARY KEY(actor,origin));
            CREATE TABLE IF NOT EXISTS triage(execution INTEGER PRIMARY KEY REFERENCES executions(id),record TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS resolutions(actor TEXT NOT NULL,key TEXT NOT NULL,package TEXT NOT NULL,preparation TEXT NOT NULL,PRIMARY KEY(actor,key));
            CREATE TABLE IF NOT EXISTS job_inputs(actor TEXT NOT NULL,key TEXT NOT NULL,repository TEXT NOT NULL,manifest TEXT NOT NULL,PRIMARY KEY(actor,key));
            CREATE TABLE IF NOT EXISTS objects(actor TEXT NOT NULL,sha256 TEXT NOT NULL,length INTEGER NOT NULL,PRIMARY KEY(actor,sha256));
            CREATE TABLE IF NOT EXISTS object_uses(sha256 TEXT PRIMARY KEY,length INTEGER NOT NULL,used_ms INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS run_objects(execution INTEGER NOT NULL REFERENCES executions(id),sha256 TEXT NOT NULL,length INTEGER NOT NULL,PRIMARY KEY(execution,sha256));
            CREATE INDEX IF NOT EXISTS run_objects_digest ON run_objects(sha256);
            CREATE TABLE IF NOT EXISTS job_contexts(execution INTEGER PRIMARY KEY REFERENCES executions(id),record BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS held_memos(actor TEXT NOT NULL,computation TEXT NOT NULL,record BLOB NOT NULL,used_ms INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(actor,computation));
            CREATE TABLE IF NOT EXISTS held_memo_objects(actor TEXT NOT NULL,computation TEXT NOT NULL,sha256 TEXT NOT NULL,PRIMARY KEY(actor,computation,sha256));
            CREATE INDEX IF NOT EXISTS held_memo_objects_digest ON held_memo_objects(sha256);
            CREATE TABLE IF NOT EXISTS checkpoints(execution INTEGER NOT NULL REFERENCES executions(id),operation_key TEXT NOT NULL,logical_key TEXT NOT NULL,content_digest TEXT NOT NULL,length INTEGER NOT NULL,attempt INTEGER NOT NULL,receipt TEXT NOT NULL,at_ms INTEGER NOT NULL,PRIMARY KEY(execution,operation_key,logical_key));
            CREATE UNIQUE INDEX IF NOT EXISTS executions_actor_request ON executions(actor,request_id) WHERE actor IS NOT NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS executions_actor_submission ON executions(actor,submission_id) WHERE actor IS NOT NULL;
            CREATE INDEX IF NOT EXISTS executions_actor_order ON executions(actor,id) WHERE actor IS NOT NULL; COMMIT;").map_err(db_error)?;
        // A journal from before held memos kept their use (#148's builds) gains the column.
        let used: bool = connection
            .query_row("SELECT COUNT(*) FROM pragma_table_info('held_memos') WHERE name='used_ms'", [], |r| r.get(0))
            .map_err(db_error)?;
        if !used {
            connection
                .execute("ALTER TABLE held_memos ADD COLUMN used_ms INTEGER NOT NULL DEFAULT 0", [])
                .map_err(db_error)?;
        }
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
        connection
            .execute(
                "INSERT OR IGNORE INTO machine_metadata(key,value) VALUES('journal_format',?1)",
                [JOURNAL_FORMAT.to_string()],
            )
            .map_err(db_error)?;
        let format: String = connection
            .query_row(
                "SELECT value FROM machine_metadata WHERE key='journal_format'",
                [],
                |row| row.get(0),
            )
            .map_err(db_error)?;
        if format
            .parse::<u32>()
            .map_or(true, |format| format > JOURNAL_FORMAT)
        {
            eprintln!(
                "journal: format {format} is newer than {JOURNAL_FORMAT}; reading what it can"
            );
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
    pub fn begin_intake(
        &mut self,
        spec: crate::native_inputs::IntakeSpec,
    ) -> io::Result<crate::native_inputs::IntakeState> {
        use crate::native_inputs::IntakeState;
        if self
            .native_owner(&spec.retention_id)?
            .is_some_and(|held| held != spec.actor)
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "intake retention belongs to another actor",
            ));
        }
        let prior: Option<String> = self
            .connection
            .query_row(
                "SELECT record FROM input_intakes WHERE actor=?1 AND retention=?2",
                params![spec.actor, spec.retention_id],
                |r| r.get(0),
            )
            .optional()
            .map_err(db_error)?;
        if let Some(prior) = prior {
            let state: IntakeState = serde_json::from_str(&prior).map_err(db_error)?;
            if encoded(&state.spec)? != encoded(&spec)? {
                return Err(admission(AdmissionError::BindingConflict));
            }
            return Ok(state);
        }
        let state = IntakeState {
            spec,
            released: false,
            receipt: None,
        };
        self.connection
            .execute(
                "INSERT INTO input_intakes(actor,retention,record) VALUES(?1,?2,?3)",
                params![state.spec.actor, state.spec.retention_id, encoded(&state)?],
            )
            .map_err(db_error)?;
        Ok(state)
    }
    pub fn intake(
        &self,
        actor: &str,
        retention: &str,
    ) -> io::Result<Option<crate::native_inputs::IntakeState>> {
        self.connection
            .query_row(
                "SELECT record FROM input_intakes WHERE actor=?1 AND retention=?2",
                params![actor, retention],
                |r| r.get::<_, String>(0),
            )
            .optional()
            .map_err(db_error)?
            .map(|record| serde_json::from_str(&record).map_err(db_error))
            .transpose()
    }
    pub fn settle_intake(
        &mut self,
        actor: &str,
        retention: &str,
        receipt: Option<Vec<u8>>,
        abort: bool,
    ) -> io::Result<crate::native_inputs::IntakeState> {
        use crate::{api::domain, native_inputs::IntakeState};
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        let raw: Option<String> = tx
            .query_row(
                "SELECT record FROM input_intakes WHERE actor=?1 AND retention=?2",
                params![actor, retention],
                |r| r.get(0),
            )
            .optional()
            .map_err(db_error)?;
        let mut state: IntakeState =
            serde_json::from_str(&raw.ok_or_else(|| {
                io::Error::new(io::ErrorKind::NotFound, "input intake not bound")
            })?)
            .map_err(db_error)?;
        if let Some(receipt) = receipt {
            let result =
                crate::archive::decode_native_byte_retention_result(receipt.as_slice()).map_err(db_error)?;
            if result.retention_id != retention || result.source.is_none() {
                return Err(db_error("native input receipt differs from bound intake"));
            }
            if let Some(prior) = &state.receipt {
                if prior != &receipt {
                    return Err(admission(AdmissionError::BindingConflict));
                }
            }
            state.receipt = Some(receipt);
            if !state.released && !abort {
                let source = crate::archive::encode_native_byte_retention_request(&domain::NativeByteRetentionRequest {
                    source: result.source,
                    retention_id: retention.into(),
                });
                let owner: Option<String> = tx
                    .query_row(
                        "SELECT actor FROM native_outputs WHERE owner=?1 LIMIT 1",
                        [retention],
                        |r| r.get(0),
                    )
                    .optional()
                    .map_err(db_error)?;
                if owner.is_some_and(|held| held != actor) {
                    return Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "native intake owner belongs to another actor",
                    ));
                }
                let prior: Option<Vec<u8>> = tx
                    .query_row(
                        "SELECT source FROM native_outputs WHERE actor=?1 AND owner=?2",
                        params![actor, retention],
                        |r| r.get(0),
                    )
                    .optional()
                    .map_err(db_error)?;
                if prior.as_ref().is_some_and(|held| held != &source) {
                    return Err(admission(AdmissionError::BindingConflict));
                }
                tx.execute(
                    "INSERT OR IGNORE INTO native_outputs(actor,owner,source) VALUES(?1,?2,?3)",
                    params![actor, retention, source],
                )
                .map_err(db_error)?;
            }
        }
        state.released |= abort;
        tx.execute(
            "UPDATE input_intakes SET record=?1 WHERE actor=?2 AND retention=?3",
            params![encoded(&state)?, actor, retention],
        )
        .map_err(db_error)?;
        tx.commit().map_err(db_error)?;
        Ok(state)
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
    pub fn installation_for_generation(
        &self,
        actor: &str,
        generation: &str,
    ) -> io::Result<Option<Installation>> {
        let record: Option<String> = self.connection.query_row("SELECT record FROM installations WHERE actor=?1 AND json_extract(record,'$.generation')=?2 LIMIT 1", params![actor,generation], |r| r.get(0)).optional().map_err(db_error)?;
        record
            .map(|record| serde_json::from_str(&record).map_err(db_error))
            .transpose()
    }
    /// Every generation an installation or an unfinished run names: never evicted meanwhile.
    pub fn bound_generations(&self) -> io::Result<std::collections::HashSet<String>> {
        let mut statement = self
            .connection
            .prepare(
                "SELECT json_extract(record,'$.generation') FROM installations
                 UNION SELECT json_extract(invocation,'$.generation') FROM executions
                 WHERE state NOT IN ('completed','failed','canceled')",
            )
            .map_err(db_error)?;
        let rows = statement
            .query_map([], |row| row.get::<_, Option<String>>(0))
            .map_err(db_error)?;
        rows.filter_map(|row| row.map_err(db_error).transpose())
            .collect()
    }

    pub fn installations(&self, actor: &str) -> io::Result<Vec<Installation>> {
        let mut statement = self
            .connection
            .prepare("SELECT record FROM installations WHERE actor=?1 ORDER BY alias")
            .map_err(db_error)?;
        let rows = statement
            .query_map([actor], |r| r.get::<_, String>(0))
            .map_err(db_error)?;
        rows.map(|r| serde_json::from_str(&r.map_err(db_error)?).map_err(db_error))
            .collect()
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

    pub fn hub_grant(&self, actor: &str, origin: &str) -> io::Result<Option<crate::hub::Grant>> {
        self.connection
            .query_row(
                "SELECT record FROM hub_access WHERE actor=?1 AND origin=?2",
                params![actor, origin],
                |r| r.get::<_, String>(0),
            )
            .optional()
            .map_err(db_error)?
            .map(|record| serde_json::from_str(&record).map_err(db_error))
            .transpose()
    }
    /// Retains a grant unless this owner already holds another account at the origin.
    pub fn put_hub_grant(
        &mut self,
        actor: &str,
        origin: &str,
        grant: &crate::hub::Grant,
    ) -> io::Result<bool> {
        if self
            .hub_grant(actor, origin)?
            .is_some_and(|prior| prior.principal != grant.principal)
        {
            return Ok(false);
        }
        self.connection
            .execute(
                "INSERT OR REPLACE INTO hub_access(actor,origin,record) VALUES(?1,?2,?3)",
                params![actor, origin, encoded(grant)?],
            )
            .map_err(db_error)?;
        Ok(true)
    }
    pub fn forget_hub_grant(&mut self, actor: &str, origin: &str) -> io::Result<()> {
        self.connection
            .execute(
                "DELETE FROM hub_access WHERE actor=?1 AND origin=?2",
                params![actor, origin],
            )
            .map_err(db_error)?;
        Ok(())
    }
    pub fn resolution(&self, actor: &str, key: &str) -> io::Result<Option<String>> {
        self.connection
            .query_row(
                "SELECT preparation FROM resolutions WHERE actor=?1 AND key=?2",
                params![actor, key],
                |r| r.get(0),
            )
            .optional()
            .map_err(db_error)
    }
    pub fn bind_resolution(
        &mut self,
        actor: &str,
        key: &str,
        package: &str,
        preparation: &str,
    ) -> io::Result<()> {
        self.connection
            .execute("INSERT OR REPLACE INTO resolutions(actor,key,package,preparation) VALUES(?1,?2,?3,?4)", params![actor, key, package, preparation])
            .map_err(db_error)?;
        Ok(())
    }
    /// A job model input this signer resolved at its Hub under `key` (its installation, slot,
    /// choice, owner, binding revision and GPU): its repository and manifest, or None.
    pub fn job_input(&self, actor: &str, key: &str) -> io::Result<Option<(String, String)>> {
        self.connection
            .query_row(
                "SELECT repository,manifest FROM job_inputs WHERE actor=?1 AND key=?2",
                params![actor, key],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()
            .map_err(db_error)
    }
    pub fn bind_job_input(&mut self, actor: &str, key: &str, repository: &str, manifest: &str) -> io::Result<()> {
        self.connection
            .execute(
                "INSERT OR REPLACE INTO job_inputs(actor,key,repository,manifest) VALUES(?1,?2,?3,?4)",
                params![actor, key, repository, manifest],
            )
            .map_err(db_error)?;
        Ok(())
    }
    /// The length of an object this signer wrote (`objects`), or None.
    /// A memoized call's result, reusable while this machine holds every file it names: the
    /// child's result record, keyed by its signer and computation. Each signer keeps its
    /// `MAX_HELD_MEMOS` most recently used.
    pub fn hold_memo(&mut self, actor: &str, computation: &str, record: &[u8], objects: &[String]) -> io::Result<()> {
        let tx = self.connection.transaction_with_behavior(TransactionBehavior::Immediate).map_err(db_error)?;
        tx.execute("INSERT OR REPLACE INTO held_memos(actor,computation,record,used_ms) VALUES(?1,?2,?3,?4)",
            params![actor, computation, record, timestamp()]).map_err(db_error)?;
        tx.execute("DELETE FROM held_memos WHERE actor=?1 AND computation NOT IN
            (SELECT computation FROM held_memos WHERE actor=?1 ORDER BY used_ms DESC LIMIT ?2)",
            params![actor, MAX_HELD_MEMOS]).map_err(db_error)?;
        tx.execute("DELETE FROM held_memo_objects WHERE actor=?1 AND computation NOT IN
            (SELECT computation FROM held_memos WHERE actor=?1)", params![actor]).map_err(db_error)?;
        tx.execute("DELETE FROM held_memo_objects WHERE actor=?1 AND computation=?2", params![actor, computation])
            .map_err(db_error)?;
        for sha256 in objects {
            tx.execute("INSERT OR IGNORE INTO held_memo_objects(actor,computation,sha256) VALUES(?1,?2,?3)",
                params![actor, computation, sha256]).map_err(db_error)?;
        }
        tx.commit().map_err(db_error)
    }
    pub fn held_memo(&self, actor: &str, computation: &str) -> io::Result<Option<Vec<u8>>> {
        let record = self.connection
            .query_row("SELECT record FROM held_memos WHERE actor=?1 AND computation=?2",
                params![actor, computation], |r| r.get(0))
            .optional()
            .map_err(db_error)?;
        if record.is_some() {
            self.connection.execute("UPDATE held_memos SET used_ms=?3 WHERE actor=?1 AND computation=?2",
                params![actor, computation, timestamp()]).map_err(db_error)?;
        }
        Ok(record)
    }
    /// Every held memo naming `sha256` goes with it (its file is no longer this machine's).
    pub fn drop_held_memos(&mut self, sha256: &str) -> io::Result<usize> {
        let tx = self.connection.transaction_with_behavior(TransactionBehavior::Immediate).map_err(db_error)?;
        let dropped = tx.execute("DELETE FROM held_memos WHERE (actor,computation) IN
            (SELECT actor,computation FROM held_memo_objects WHERE sha256=?1)", params![sha256]).map_err(db_error)?;
        tx.execute("DELETE FROM held_memo_objects WHERE (actor,computation) NOT IN
            (SELECT actor,computation FROM held_memos)", []).map_err(db_error)?;
        tx.commit().map_err(db_error)?;
        Ok(dropped)
    }
    pub fn object(&self, actor: &str, sha256: &str) -> io::Result<Option<u64>> {
        self.connection
            .query_row(
                "SELECT length FROM objects WHERE actor=?1 AND sha256=?2",
                params![actor, sha256],
                |r| r.get(0),
            )
            .optional()
            .map_err(db_error)
    }
    pub fn bind_object(
        &mut self,
        actor: &str,
        object: &tensorfs_core::ids::ObjectRef,
    ) -> io::Result<()> {
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        tx.execute(
            "INSERT OR REPLACE INTO objects(actor,sha256,length) VALUES(?1,?2,?3)",
            params![actor, object.sha256, object.length],
        )
        .map_err(db_error)?;
        touch_object(&tx, object)?;
        tx.commit().map_err(db_error)?;
        Ok(())
    }

    /// A child's result becomes its parent's dependency before the parent is handed it.
    pub fn adopt_object(
        &mut self,
        actor: &str,
        parent: &str,
        object: &tensorfs_core::ids::ObjectRef,
    ) -> io::Result<()> {
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        tx.execute(
            "INSERT OR REPLACE INTO objects(actor,sha256,length) VALUES(?1,?2,?3)",
            params![actor, object.sha256, object.length],
        )
        .map_err(db_error)?;
        reference_objects(&tx, parent, std::slice::from_ref(object))?;
        tx.commit().map_err(db_error)
    }

    /// Whether nothing needs this object's root: no use (a Write, a new run) since
    /// `before_ms`, and no run naming it that is unfinished (any state but the three
    /// terminal ones) or ended since then.
    pub fn object_releasable(&self, sha256: &str, before_ms: i64) -> io::Result<bool> {
        self.connection
            .query_row(
                "SELECT NOT EXISTS(SELECT 1 FROM object_uses WHERE sha256=?1 AND used_ms>=?2)
                 AND NOT EXISTS(SELECT 1 FROM run_objects r JOIN executions e ON e.id=r.execution
                     WHERE r.sha256=?1 AND (e.updated_ms>=?2
                         OR e.state NOT IN ('completed','failed','canceled')))",
                params![sha256, before_ms],
                |row| row.get(0),
            )
            .map_err(db_error)
    }
    /// The failed attempt's triage bundle reference; written before the run settles.
    pub fn bind_triage(&mut self, id: &str, triage: &crate::triage::TriageRef) -> io::Result<()> {
        self.connection
            .execute(
                "INSERT OR REPLACE INTO triage(execution,record) VALUES(?1,?2)",
                params![id, encoded(triage)?],
            )
            .map_err(db_error)?;
        Ok(())
    }
    pub fn triage(&self, id: &str) -> io::Result<Option<crate::triage::TriageRef>> {
        let record: Option<String> = self
            .connection
            .query_row("SELECT record FROM triage WHERE execution=?1", [id], |r| {
                r.get(0)
            })
            .optional()
            .map_err(db_error)?;
        record
            .map(|record| serde_json::from_str(&record).map_err(db_error))
            .transpose()
    }
    /// Drops cached model resolutions of a package: the next run reads its bindings again.
    pub fn forget_resolutions(&mut self, actor: &str, package: &str) -> io::Result<()> {
        self.connection
            .execute(
                "DELETE FROM resolutions WHERE actor=?1 AND package=?2",
                params![actor, package],
            )
            .map_err(db_error)?;
        Ok(())
    }
    /// The accepted public execution a submission or request already names, if any.
    pub fn accepted_public(
        &self,
        actor: &str,
        request_id: &str,
        submission_id: &str,
    ) -> io::Result<Option<Execution>> {
        let record: Option<String> = self
            .connection
            .query_row(
                "SELECT record FROM executions WHERE actor=?1 AND (request_id=?2 OR submission_id=?3) LIMIT 1",
                params![actor, request_id, submission_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(db_error)?;
        record
            .map(|r| serde_json::from_str(&r).map_err(db_error))
            .transpose()
    }
    pub fn preparation(&self, actor: &str, id: &str) -> io::Result<Option<Preparation>> {
        self.connection
            .query_row(
                "SELECT record FROM preparations WHERE actor=?1 AND id=?2",
                params![actor, id],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(db_error)?
            .map(|record| serde_json::from_str(&record).map_err(db_error))
            .transpose()
    }

    /// Replace `actor`'s warm set with `members` (each one's record, in its order).
    pub fn replace_warm_set(&mut self, actor: &str, members: &[String]) -> io::Result<()> {
        let transaction = self.connection.transaction().map_err(db_error)?;
        transaction
            .execute("DELETE FROM warm_sets WHERE actor=?1", [actor])
            .map_err(db_error)?;
        for (position, record) in members.iter().enumerate() {
            transaction
                .execute(
                    "INSERT INTO warm_sets(actor,position,record) VALUES(?1,?2,?3)",
                    params![actor, position as i64, record],
                )
                .map_err(db_error)?;
        }
        transaction.commit().map_err(db_error)
    }
    /// Every warm set member's record as (actor, record), each actor's in its order.
    pub fn warm_sets(&self) -> io::Result<Vec<(String, String)>> {
        let mut statement = self
            .connection
            .prepare("SELECT actor,record FROM warm_sets ORDER BY actor,position")
            .map_err(db_error)?;
        let rows = statement
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .map_err(db_error)?;
        rows.map(|row| row.map_err(db_error)).collect()
    }
    /// Every preparation `actor` bound to `installation`.
    pub fn preparations_of(&self, actor: &str, installation: &str) -> io::Result<Vec<Preparation>> {
        let mut statement = self
            .connection
            .prepare("SELECT record FROM preparations WHERE actor=?1 AND json_extract(record,'$.installation')=?2")
            .map_err(db_error)?;
        let rows = statement
            .query_map(params![actor, installation], |row| row.get::<_, String>(0))
            .map_err(db_error)?;
        rows.map(|row| serde_json::from_str(&row.map_err(db_error)?).map_err(db_error))
            .collect()
    }

    pub fn bind_preparation(&mut self, record: Preparation) -> io::Result<Preparation> {
        validate_scope(&record.actor, &record.id, &record.installation)?;
        if self
            .installation(&record.actor, &record.installation)?
            .is_none()
        {
            return Err(admission(AdmissionError::BindingConflict));
        }
        if let Some(prior) = self.preparation(&record.actor, &record.id)? {
            if prior != record {
                return Err(admission(AdmissionError::BindingConflict));
            }
            return Ok(prior);
        }
        self.connection
            .execute(
                "INSERT INTO preparations(actor,id,record) VALUES(?1,?2,?3)",
                params![record.actor, record.id, encoded(&record)?],
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
    pub fn acknowledge_collection(&mut self, id: &str) -> io::Result<Execution> {
        self.acknowledge_collection_events(id, None)
    }
    pub fn acknowledge_collection_events(
        &mut self,
        id: &str,
        events: Option<&[u8]>,
    ) -> io::Result<Execution> {
        let mut record = self.get(id)?;
        if !record.state.terminal() {
            return Err(db_error(
                "collection acknowledgement requires a terminal result",
            ));
        }
        let transaction = self.connection.transaction().map_err(db_error)?;
        if let Some(events) = events {
            if transaction
                .execute(
                    "UPDATE public_terminals SET events=?1 WHERE execution=?2",
                    params![events, id],
                )
                .map_err(db_error)?
                != 1
            {
                return Err(db_error("terminal projection absent during collection"));
            }
        }
        if !record.collected {
            record.collected = true;
            transaction
                .execute(
                    "UPDATE executions SET record=?1 WHERE id=?2",
                    params![encoded(&record)?, id],
                )
                .map_err(db_error)?;
        }
        transaction.commit().map_err(db_error)?;
        Ok(record)
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
    pub fn native_owner(&self, owner: &str) -> io::Result<Option<String>> {
        self.connection
            .query_row(
                "SELECT actor FROM native_outputs WHERE owner=?1 LIMIT 1",
                [owner],
                |r| r.get(0),
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
        if self.native_owner(owner)?.is_some_and(|held| held != actor) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "native retention is already owned by another actor",
            ));
        }
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
        if !context.preparation_id.is_empty() {
            let preparation = self
                .preparation(&context.actor, &context.preparation_id)?
                .ok_or_else(|| admission(AdmissionError::BindingConflict))?;
            let installation = self
                .installation(&context.actor, &preparation.installation)?
                .ok_or_else(|| admission(AdmissionError::BindingConflict))?;
            if installation.generation != invocation.generation {
                return Err(admission(AdmissionError::BindingConflict));
            }
        }
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
        let execution = insert(&tx, key, invocation, context, boot, None)?;
        tx.commit().map_err(db_error)?;
        Ok(execution)
    }

    /// A run (`cozy.machine.v1` Run) accepted before its preparation: queued and waiting on
    /// `PREPARING`, idempotent on (actor, id) by its spec digest. True when newly accepted.
    pub fn accept_run(
        &mut self,
        actor: &str,
        id: &str,
        digest: &str,
        invocation: Invocation,
    ) -> io::Result<(Execution, bool)> {
        self.accept_run_objects(actor, id, digest, invocation, &[])
    }

    /// `accept_run`, also referencing `objects` (tree members, a local source) the run needs.
    pub fn accept_run_objects(
        &mut self,
        actor: &str,
        id: &str,
        digest: &str,
        invocation: Invocation,
        objects: &[tensorfs_core::ids::ObjectRef],
    ) -> io::Result<(Execution, bool)> {
        validate_scope(actor, id, id)?;
        let key = format!("run:{}", encoded(&(actor, id))?);
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        if update_run_reserved(&tx, actor, id)? {
            return Err(io::Error::new(io::ErrorKind::AlreadyExists, "this run id names a software update"));
        }
        let prior: Option<String> = tx
            .query_row(
                "SELECT record FROM executions WHERE idempotency_key=?1",
                [&key],
                |row| row.get(0),
            )
            .optional()
            .map_err(db_error)?;
        if let Some(prior) = prior {
            let execution: Execution = serde_json::from_str(&prior).map_err(db_error)?;
            if execution.canceled_before_acceptance && execution.state == State::Canceled {
                return Ok((execution, false));
            }
            if execution
                .submission
                .as_ref()
                .map(|s| s.invocation_digest.as_str())
                != Some(digest)
            {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "this run id already names another run spec",
                ));
            }
            return Ok((execution, false));
        }
        let context = SubmissionContext {
            actor: actor.into(),
            request_id: id.into(),
            submission_id: id.into(),
            expected_workspace_id: String::new(),
            capture_digest: String::new(),
            invocation_digest: digest.into(),
            payload_digest: String::new(),
            publication_authorization_id: String::new(),
            preparation_id: String::new(),
        };
        let execution = insert(&tx, &key, invocation, Some(context), "", Some(PREPARING))?;
        reference_objects(&tx, &execution.id, objects)?;
        tx.commit().map_err(db_error)?;
        Ok((execution, true))
    }

    /// Reserves an absent run as canceled in the same transaction namespace as acceptance.
    /// An accepted run is returned for normal cancellation; its spec/outcome is never replaced.
    pub fn reserve_run_cancellation(&mut self, actor: &str, id: &str) -> io::Result<Execution> {
        validate_scope(actor, id, id)?;
        let key = format!("run:{}", encoded(&(actor, id))?);
        let tx = self.connection.transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        if update_run_reserved(&tx, actor, id)? {
            return Err(io::Error::new(io::ErrorKind::Unsupported, "software update runs do not support Control"));
        }
        if let Some(prior) = public_prior(&tx, actor, id, id)? {
            return serde_json::from_str(&prior).map_err(db_error);
        }
        let context = SubmissionContext {
            actor: actor.into(), request_id: id.into(), submission_id: id.into(),
            ..Default::default()
        };
        let mut record = insert(&tx, &key, Invocation::default(), Some(context), "", None)?;
        record.state = State::Canceled;
        record.cancel_actor = Some(actor.into());
        record.canceled_before_acceptance = true;
        record.finished_at_ms = timestamp().max(0) as u64;
        tx.execute("UPDATE executions SET record=?1,state='canceled',updated_ms=?2 WHERE id=?3",
            params![encoded(&record)?, timestamp(), record.id]).map_err(db_error)?;
        tx.commit().map_err(db_error)?;
        Ok(record)
    }

    /// Updates keep their payload journal, but share actor/ID ownership with Run and CANCEL.
    pub fn reserve_update_run(&mut self, actor: &str, id: &str) -> io::Result<()> {
        validate_scope(actor, id, id)?;
        let tx = self.connection.transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        if public_prior(&tx, actor, id, id)?.is_some() {
            return Err(io::Error::new(io::ErrorKind::AlreadyExists, "this run id already names an ordinary run"));
        }
        tx.execute("INSERT OR IGNORE INTO update_run_ids(actor,id) VALUES(?1,?2)", params![actor, id])
            .map_err(db_error)?;
        tx.commit().map_err(db_error)
    }

    pub fn update_run_reserved(&self, actor: &str, id: &str) -> io::Result<bool> {
        update_run_reserved(&self.connection, actor, id)
    }

    /// A preparing run's code and models are ready: it names them and becomes dispatchable.
    pub fn bind_prepared(
        &mut self,
        id: &str,
        invocation: Invocation,
        preparation: &str,
    ) -> io::Result<Execution> {
        let encoded_invocation = encoded(&invocation)?;
        self.update_with(
            id,
            None,
            |record| {
                if record.state != State::Queued
                    || record.waiting_reason.as_deref() != Some(PREPARING)
                {
                    return Ok(false);
                }
                record.invocation = invocation;
                if let Some(submission) = record.submission.as_mut() {
                    submission.preparation_id = preparation.into();
                }
                record.waiting_reason = None;
                if record.pause_actor.is_some() {
                    record.state = State::Paused;
                }
                Ok(true)
            },
            |tx, record| {
                tx.execute(
                    "UPDATE executions SET invocation=?1 WHERE id=?2",
                    params![encoded_invocation, record.id],
                )
                .map_err(db_error)?;
                Ok(())
            },
        )
    }

    /// A preparing run ends without dispatch: a warm run's success or a preparation failure.
    pub fn end_preparation(&mut self, id: &str, outcome: Outcome) -> io::Result<Execution> {
        self.update(id, |record| {
            if record.state != State::Queued || record.waiting_reason.as_deref() != Some(PREPARING)
            {
                return Ok(false);
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
                Outcome::Canceled => record.state = State::Canceled,
                Outcome::Paused => return Err(db_error("a preparation pauses once prepared")),
            }
            record.waiting_reason = None;
            Ok(true)
        })
    }

    /// A restarted machine holds no preparation and no run's Hub token: each preparing run
    /// ends FAILED (nothing of it was started).
    pub(crate) fn interrupt_preparations(&mut self) -> io::Result<()> {
        let preparing: Vec<String> = {
            let mut statement = self
                .connection
                .prepare("SELECT id FROM executions WHERE state='queued' AND json_extract(record,'$.waiting_reason')=?1")
                .map_err(db_error)?;
            let rows = statement
                .query_map([PREPARING], |row| row.get::<_, i64>(0))
                .map_err(db_error)?;
            rows.map(|id| id.map(|id| id.to_string()))
                .collect::<Result<_, _>>()
                .map_err(db_error)?
        };
        for id in preparing {
            let failure = Failure {
                status: 3,
                cause: 7,
                origin: 3,
                message: "preparation_interrupted: the machine restarted while this run was preparing; run it again".into(),
            };
            self.end_preparation(&id, Outcome::Failed(failure.encode()))?;
        }
        Ok(())
    }
}

fn touch_object(
    tx: &rusqlite::Transaction<'_>,
    object: &tensorfs_core::ids::ObjectRef,
) -> io::Result<()> {
    tensorfs_core::ids::hex64("run object", &object.sha256).map_err(db_error)?;
    tx.execute(
        "INSERT INTO object_uses(sha256,length,used_ms) VALUES(?1,?2,?3)
        ON CONFLICT(sha256) DO UPDATE SET used_ms=excluded.used_ms",
        params![object.sha256, object.length, timestamp()],
    )
    .map_err(db_error)?;
    Ok(())
}
fn reference_objects(
    tx: &rusqlite::Transaction<'_>,
    execution: &str,
    objects: &[tensorfs_core::ids::ObjectRef],
) -> io::Result<()> {
    for object in objects {
        touch_object(tx, object)?;
        tx.execute(
            "INSERT OR IGNORE INTO run_objects(execution,sha256,length) VALUES(?1,?2,?3)",
            params![execution, object.sha256, object.length],
        )
        .map_err(db_error)?;
    }
    Ok(())
}

/// The waiting reason of a run accepted before its preparation completes.
pub const PREPARING: &str = "preparing";

fn insert(
    tx: &rusqlite::Transaction<'_>,
    key: &str,
    invocation: Invocation,
    context: Option<SubmissionContext>,
    boot: &str,
    waiting: Option<&str>,
) -> io::Result<Execution> {
    let objects: Vec<_> = invocation
        .inputs
        .iter()
        .filter_map(|input| {
            Some(tensorfs_core::ids::ObjectRef {
                sha256: input.digest.strip_prefix("sha256:")?.into(),
                length: input.length,
            })
        })
        .collect();
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
        collected: false,
        revision_ceiling: 1,
        attempt: 0,
        waiting_reason: waiting.map(String::from),
        process: None,
        cancel_actor: None,
        canceled_before_acceptance: false,
        pause_actor: None,
        completed_units: 0,
        progress: None,
        running_revision: 0,
        started_at_ms: 0,
        executed_ms: 0,
        executor: None,
        result: None,
        failure: None,
    };
    tx.execute(
        "UPDATE executions SET record=?1 WHERE id=?2",
        params![encoded(&execution)?, execution.id],
    )
    .map_err(db_error)?;
    reference_objects(tx, &execution.id, &objects)?;
    Ok(execution)
}

impl Journal {
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
            .map(|record| record.map_err(db_error))
            .collect::<io::Result<Vec<_>>>()
            .map(readable)
    }

    /// Closure is a durable acceptance tombstone, never implicit execution cancellation.
    pub fn actor_page(
        &self,
        actor: &str,
        after: u64,
        before: u64,
        newest: bool,
        states: &[String],
        limit: usize,
    ) -> io::Result<Vec<Execution>> {
        let states: Vec<&str> = states
            .iter()
            .map(|s| {
                if s == "succeeded" {
                    "completed"
                } else {
                    s.as_str()
                }
            })
            .collect();
        let sql=format!("SELECT record FROM executions WHERE actor=?1 AND id>?2 AND (?3=0 OR id<?3) AND (?4='[]' OR state IN (SELECT value FROM json_each(?4))) ORDER BY id {} LIMIT ?5",if newest{"DESC"}else{"ASC"});
        let mut statement = self.connection.prepare(&sql).map_err(db_error)?;
        let records = statement
            .query_map(
                params![
                    actor,
                    after.min(i64::MAX as u64) as i64,
                    before.min(i64::MAX as u64) as i64,
                    encoded(&states)?,
                    limit.min(256) as i64
                ],
                |row| row.get::<_, String>(0),
            )
            .map_err(db_error)?;
        records
            .map(|record| record.map_err(db_error))
            .collect::<io::Result<Vec<_>>>()
            .map(readable)
    }
    pub fn actor_head(&self, actor: &str) -> io::Result<u64> {
        self.connection
            .query_row(
                "SELECT COALESCE(MAX(id),0) FROM executions WHERE actor=?1",
                [actor],
                |r| r.get(0),
            )
            .map_err(db_error)
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
            .map(|record| record.map_err(db_error))
            .collect::<io::Result<Vec<_>>>()
            .map(readable)
    }

    /// Every run not completed, failed or canceled: paused and unknown states too. A row
    /// that does not decode is an error here, where a missed run is a model evicted.
    pub fn unfinished(&self) -> io::Result<Vec<Execution>> {
        let mut statement = self
            .connection
            .prepare("SELECT record FROM executions WHERE state NOT IN ('completed','failed','canceled') ORDER BY id")
            .map_err(db_error)?;
        let rows = statement
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(db_error)?;
        rows.map(|row| serde_json::from_str(&row.map_err(db_error)?).map_err(db_error))
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

    pub fn ready_after(&self, after: u64, limit: usize) -> io::Result<Vec<Execution>> {
        let mut statement = self.connection.prepare("SELECT record FROM executions WHERE id>?1 AND state='queued' AND json_extract(record,'$.waiting_reason') IS NULL ORDER BY id LIMIT ?2").map_err(db_error)?;
        let records = statement
            .query_map(
                params![
                    after.min(i64::MAX as u64) as i64,
                    limit.min(i64::MAX as usize) as i64
                ],
                |row| row.get::<_, String>(0),
            )
            .map_err(db_error)?;
        records
            .map(|record| record.map_err(db_error))
            .collect::<io::Result<Vec<_>>>()
            .map(readable)
    }

    /// Includes terminal requests: their retained executor may still own a context.
    /// Read only birth metadata in bounded pages, never materialize old results.
    pub fn gpu_births_after(
        &self,
        after: u64,
        limit: usize,
    ) -> io::Result<Vec<(u64, ProcessBirth)>> {
        let mut statement = self.connection.prepare("SELECT id,json_extract(record,'$.process') FROM executions WHERE id>?1 AND json_extract(record,'$.submission.preparation_id') IS NOT NULL AND json_extract(record,'$.submission.preparation_id')!='' AND json_extract(record,'$.process') IS NOT NULL ORDER BY id LIMIT ?2").map_err(db_error)?;
        let rows = statement
            .query_map(
                params![
                    after.min(i64::MAX as u64) as i64,
                    limit.min(i64::MAX as usize) as i64
                ],
                |row| Ok((row.get::<_, u64>(0)?, row.get::<_, String>(1)?)),
            )
            .map_err(db_error)?;
        rows.map(|row| {
            let (id, birth) = row.map_err(db_error)?;
            Ok((id, serde_json::from_str(&birth).map_err(db_error)?))
        })
        .collect()
    }

    fn selected(&self, query: &str, limit: usize) -> io::Result<Vec<Execution>> {
        let mut statement = self.connection.prepare(query).map_err(db_error)?;
        let records = statement
            .query_map([limit.min(i64::MAX as usize) as i64], |row| {
                row.get::<_, String>(0)
            })
            .map_err(db_error)?;
        records
            .map(|record| record.map_err(db_error))
            .collect::<io::Result<Vec<_>>>()
            .map(readable)
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
        self.update_with(id, progress, change, |_, _| Ok(()))
    }

    /// One durable transition; `also` writes in the same transaction, after the record has
    /// its new revision, only when the transition changed it.
    fn update_with(
        &mut self,
        id: &str,
        progress: Option<&ProgressSnapshot>,
        change: impl FnOnce(&mut Execution) -> io::Result<bool>,
        also: impl FnOnce(&rusqlite::Transaction<'_>, &Execution) -> io::Result<()>,
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
        if record.state == State::Unknown {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "execution state was written by a newer machine",
            ));
        }
        let was_terminal = record.state.terminal();
        let running_since = (record.state == State::Running).then_some(record.started_at_ms);
        let changed = change(&mut record)?;
        if let Some(since) = running_since.filter(|&s| s > 0 && record.state != State::Running) {
            let now = timestamp().max(0) as u64;
            record.executed_ms += now.saturating_sub(since);
        }
        let observed =
            !was_terminal && progress.is_some_and(|progress| progress.overlay(&mut record));
        if changed || observed {
            if record.state.terminal() && record.finished_at_ms == 0 {
                record.finished_at_ms = timestamp().max(0) as u64;
            }
            // Every durable mutation jumps past all volatile cursors the former owner
            // could have published, even when its progress snapshot was lost.
            record.revision = record
                .revision
                .max(record.revision_ceiling)
                .checked_add(1)
                .ok_or_else(|| db_error("execution observation cursor exhausted"))?;
            record.revision_ceiling = if record.observed() {
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
            also(&tx, &record)?;
        }
        tx.commit().map_err(db_error)?;
        Ok(record)
    }

    /// Journal one product on a started execution's output log. Its event sequence is the
    /// transition's revision, so products interleave with every other observed change.
    pub fn append_product(
        &mut self,
        id: &str,
        progress: Option<&ProgressSnapshot>,
        product: &[u8],
    ) -> io::Result<u64> {
        let record = self.update_with(
            id,
            progress,
            |record| {
                if record.state.terminal() || record.state == State::Queued {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "only a running execution publishes products",
                    ));
                }
                Ok(true)
            },
            |tx, record| {
                tx.execute(
                    "INSERT INTO run_products(execution,sequence,at_ms,product) VALUES(?1,?2,?3,?4)",
                    params![id, record.revision as i64, timestamp(), product],
                )
                .map_err(db_error)?;
                Ok(())
            },
        )?;
        Ok(record.revision)
    }

    /// One line of the run's log the machine itself writes (a `log` event: a warning about
    /// its request), at the next sequence.
    pub fn append_log(
        &mut self,
        id: &str,
        progress: Option<&ProgressSnapshot>,
        line: &[u8],
    ) -> io::Result<u64> {
        self.append_entry("INSERT INTO run_logs(execution,sequence,at_ms,record) VALUES(?1,?2,?3,?4)", id, progress, line)
    }
    /// A call of the run applied its attention pin: the pin is the root run's, whichever call
    /// holds the attention sites.
    pub fn apply_attention(&mut self, id: &str) -> io::Result<()> {
        let mut root = self.get(id)?;
        while !root.invocation.parent.is_empty() {
            root = self.get(&root.invocation.parent)?;
        }
        self.connection
            .execute(
                "INSERT OR IGNORE INTO attention_applied(execution) VALUES(?1)",
                [&root.id],
            )
            .map_err(db_error)?;
        Ok(())
    }
    /// A root run completing with a pin none of its calls applied says so in its log: an
    /// unused request field warns, never fails.
    pub fn warn_unapplied_attention(
        &mut self,
        id: &str,
        progress: Option<&ProgressSnapshot>,
    ) -> io::Result<()> {
        let record = self.get(id)?;
        let pin = &record.invocation.attention_kernel;
        if pin.is_empty() || !record.invocation.parent.is_empty() || record.state.terminal() {
            return Ok(());
        }
        let applied: bool = self
            .connection
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM attention_applied WHERE execution=?1)",
                [id],
                |row| row.get(0),
            )
            .map_err(db_error)?;
        if applied {
            return Ok(());
        }
        let text = format!("attention pin {pin} was never applied");
        let line = serde_json::to_vec(&serde_json::json!({"level": "warning", "text": text}))?;
        self.append_log(id, progress, &line).map(drop)
    }
    fn append_entry(
        &mut self,
        insert: &str,
        id: &str,
        progress: Option<&ProgressSnapshot>,
        entry: &[u8],
    ) -> io::Result<u64> {
        let record = self.update_with(
            id,
            progress,
            |record| Ok(!record.state.terminal()),
            |tx, record| {
                tx.execute(insert, params![id, record.revision as i64, timestamp(), entry])
                    .map_err(db_error)?;
                Ok(())
            },
        )?;
        Ok(record.revision)
    }
    /// One settled call of the run (a `call` event: a child run, or an effect such as its
    /// weights publication) at the next sequence, once per `call`; None when it is already
    /// in the log or the run has ended.
    pub fn append_call(
        &mut self,
        id: &str,
        progress: Option<&ProgressSnapshot>,
        call: &str,
        body: &[u8],
    ) -> io::Result<Option<u64>> {
        let known: bool = self
            .connection
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM run_calls WHERE execution=?1 AND call=?2)",
                params![id, call],
                |row| row.get(0),
            )
            .map_err(db_error)?;
        if known {
            return Ok(None);
        }
        let mut appended = false;
        let record = self.update_with(
            id,
            progress,
            |record| Ok(!record.state.terminal()),
            |tx, record| {
                appended = true;
                tx.execute(
                    "INSERT INTO run_calls(execution,sequence,at_ms,call,record) VALUES(?1,?2,?3,?4,?5)",
                    params![id, record.revision as i64, timestamp(), call, body],
                )
                .map_err(db_error)?;
                Ok(())
            },
        )?;
        Ok(appended.then_some(record.revision))
    }
    pub fn calls(&self, id: &str) -> io::Result<Vec<StoredProduct>> {
        self.stored(
            "SELECT sequence,at_ms,record FROM run_calls WHERE execution=?1 ORDER BY sequence",
            id,
        )
    }
    pub fn logs(&self, id: &str) -> io::Result<Vec<StoredProduct>> {
        self.stored(
            "SELECT sequence,at_ms,record FROM run_logs WHERE execution=?1 ORDER BY sequence",
            id,
        )
    }

    /// What the execution's executor measured of its latest attempt (`run show`).
    pub fn record_measurements(&mut self, id: &str, measurements: &[u8]) -> io::Result<()> {
        self.connection
            .execute(
                "INSERT INTO run_measurements(execution,measurements) VALUES(?1,?2)
                 ON CONFLICT(execution) DO UPDATE SET measurements=excluded.measurements",
                params![id, measurements],
            )
            .map(drop)
            .map_err(db_error)
    }
    pub fn measurements(&self, id: &str) -> io::Result<Option<Vec<u8>>> {
        self.connection
            .query_row(
                "SELECT measurements FROM run_measurements WHERE execution=?1",
                [id],
                |row| row.get(0),
            )
            .optional()
            .map_err(db_error)
    }

    /// The execution's output log, oldest first.
    pub fn products(&self, id: &str) -> io::Result<Vec<StoredProduct>> {
        self.stored(
            "SELECT sequence,at_ms,product FROM run_products WHERE execution=?1 ORDER BY sequence",
            id,
        )
    }
    fn stored(&self, query: &str, id: &str) -> io::Result<Vec<StoredProduct>> {
        let mut statement = self.connection.prepare(query).map_err(db_error)?;
        let rows = statement
            .query_map([id], |row| {
                Ok(StoredProduct {
                    sequence: row.get::<_, i64>(0)? as u64,
                    at_ms: row.get::<_, i64>(1)? as u64,
                    product: row.get(2)?,
                })
            })
            .map_err(db_error)?;
        rows.collect::<Result<_, _>>().map_err(db_error)
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
            } else if record.pause_actor.is_some() {
                record.waiting_reason = None;
                State::Paused
            } else {
                State::Queued
            };
            Ok(true)
        })
    }

    /// Authorized, but the request never reached a handler: a retained executor's channel was
    /// gone before Invoke (PrepareRequest enters none). After its exact termination the attempt
    /// stays claimed, back to awaiting its first executor.
    pub fn redeliver(&mut self, id: &str) -> io::Result<Execution> {
        self.update(id, |record| {
            if !matches!(record.state, State::Starting | State::Running) {
                return Err(db_error("only an active attempt is redelivered"));
            }
            record.process = None;
            record.executor = None;
            record.started_at_ms = 0;
            record.state = if record.cancel_actor.is_some() {
                State::Canceled
            } else if record.pause_actor.is_some() {
                State::Paused
            } else {
                State::Starting
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

    pub fn running(&mut self, id: &str, executor: Option<ExecutorFacts>) -> io::Result<Execution> {
        self.update(id, |record| {
            if record.state != State::Starting || record.process.is_none() {
                return Err(db_error("attempt has no registered executor"));
            }
            if record.cancel_actor.is_some() || record.pause_actor.is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "stopped before package start authorization",
                ));
            }
            record.state = State::Running;
            record.running_revision = record.revision.max(record.revision_ceiling) + 1;
            record.started_at_ms = timestamp().max(0) as u64;
            record.executor = executor;
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
            if !record.observed() {
                return Err(db_error(
                    "only running or preparing executions reserve observation cursors",
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
            if matches!(record.state, State::Queued | State::Paused) {
                record.state = State::Canceled;
                record.finished_at_ms = timestamp().max(0) as u64;
            }
            Ok(true)
        })
    }

    /// Pause a run: an unstarted attempt rests at once, a preparing one once prepared, a
    /// started one when it stops (`Outcome::Paused`). `unstarted_only` holds only an attempt
    /// whose package code has not started: queued, or claimed and not yet authorized (`running`
    /// then refuses it). A paused job's children: started work runs to its end. Finished,
    /// canceled and already pausing runs are unchanged.
    pub fn pause(&mut self, id: &str, actor: &str, unstarted_only: bool) -> io::Result<Execution> {
        if actor.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "a pause requires an actor",
            ));
        }
        self.update(id, |record| {
            if record.state.terminal()
                || record.state == State::Paused
                || record.cancel_actor.is_some()
                || record.pause_actor.is_some()
                || (unstarted_only && !matches!(record.state, State::Queued | State::Starting))
            {
                return Ok(false);
            }
            record.pause_actor = Some(actor.into());
            if record.state == State::Queued && record.waiting_reason.as_deref() != Some(PREPARING)
            {
                record.state = State::Paused;
                record.waiting_reason = None;
            }
            Ok(true)
        })
    }

    /// A paused run queues for a fresh attempt; one paused while preparing only drops the
    /// pause. A run still stopping, running or finished is unchanged.
    pub fn resume(&mut self, id: &str) -> io::Result<Execution> {
        self.update(id, |record| {
            match record.state {
                State::Paused => {
                    record.state = State::Queued;
                    record.waiting_reason = None;
                    // A new attempt reports its own progress from zero.
                    record.completed_units = 0;
                    record.progress = None;
                }
                State::Queued if record.pause_actor.is_some() => (),
                _ => return Ok(false),
            }
            record.pause_actor = None;
            Ok(true)
        })
    }

    /// A job's children that have not finished, paused ones included.
    pub fn children(&self, parent: &str) -> io::Result<Vec<Execution>> {
        let mut statement = self
            .connection
            .prepare("SELECT record FROM executions WHERE state IN ('queued','starting','running','paused') AND json_extract(invocation,'$.parent')=?1 ORDER BY id")
            .map_err(db_error)?;
        let records = statement
            .query_map([parent], |row| row.get::<_, String>(0))
            .map_err(db_error)?;
        records
            .map(|record| record.map_err(db_error))
            .collect::<io::Result<Vec<_>>>()
            .map(readable)
    }

    pub fn paused(&self, limit: usize) -> io::Result<Vec<Execution>> {
        self.selected(
            "SELECT record FROM executions WHERE state='paused' ORDER BY id LIMIT ?1",
            limit,
        )
    }

    /// A job's preparation context without its tokens (`runs::JobContext`): what its children
    /// prepare with after a restart.
    pub fn bind_job_context(&mut self, id: &str, record: &[u8]) -> io::Result<()> {
        self.connection
            .execute(
                "INSERT OR REPLACE INTO job_contexts(execution,record) VALUES(?1,?2)",
                params![id, record],
            )
            .map(drop)
            .map_err(db_error)
    }
    pub fn job_context(&self, id: &str) -> io::Result<Option<Vec<u8>>> {
        self.connection
            .query_row(
                "SELECT record FROM job_contexts WHERE execution=?1",
                [id],
                |row| row.get(0),
            )
            .optional()
            .map_err(db_error)
    }
    pub fn forget_job_context(&mut self, id: &str) -> io::Result<()> {
        self.connection
            .execute("DELETE FROM job_contexts WHERE execution=?1", [id])
            .map(drop)
            .map_err(db_error)
    }

    /// Record that a job's attempt declared a checkpoint (`Checkpoints.declare`): the same
    /// keys and content replay its receipt; the same keys with other content are a conflict.
    /// The run's scratch holds the bytes; only the declaration is journaled.
    pub fn declare_checkpoint(
        &mut self,
        id: &str,
        attempt: u32,
        operation_key: &str,
        logical_key: &str,
        content_digest: &str,
        length: u64,
    ) -> io::Result<(String, bool)> {
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        let held: Option<(String, String)> = tx
            .query_row(
                "SELECT content_digest,receipt FROM checkpoints WHERE execution=?1 AND operation_key=?2 AND logical_key=?3",
                params![id, operation_key, logical_key],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(db_error)?;
        if let Some((digest, receipt)) = held {
            if digest != content_digest {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    format!(
                        "{logical_key}: operation key {operation_key:?} is declared at {digest}; \
                         a checkpoint is never replaced"
                    ),
                ));
            }
            return Ok((receipt, true));
        }
        let receipt = format!("ckpt-{}", &uuid::Uuid::new_v4().simple().to_string()[..24]);
        tx.execute(
            "INSERT INTO checkpoints(execution,operation_key,logical_key,content_digest,length,attempt,receipt,at_ms) VALUES(?1,?2,?3,?4,?5,?6,?7,?8)",
            params![id, operation_key, logical_key, content_digest, length.min(i64::MAX as u64) as i64, attempt, receipt, timestamp()],
        )
        .map_err(db_error)?;
        tx.commit().map_err(db_error)?;
        Ok((receipt, false))
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
            // A paused run's attempt has ended: a late outcome of it (its executor's shutdown
            // failing) changes nothing.
            if record.state.terminal() || record.state == State::Paused {
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
                // The attempt stopped: the run rests, awaiting `resume` and a fresh attempt.
                Outcome::Paused => {
                    if record.pause_actor.is_none() {
                        return Err(db_error("an attempt pauses only with durable authority"));
                    }
                    record.state = State::Paused;
                    record.process = None;
                    return Ok(true);
                }
            }
            record.finished_at_ms = timestamp().max(0) as u64;
            Ok(true)
        })
    }
}

/// Why a run did not complete, in the retained outcome body's terms (numeric codes:
/// status FAILED=3/REFUSED=2/ABANDONED=5; cause codes and origins as numbered there).
/// Stored as JSON in `Execution::failure`; older plain text reads as an executor fault.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Failure {
    pub status: u8,
    pub cause: u8,
    pub origin: u8,
    pub message: String,
}
impl Failure {
    /// An executor's own terminal (`refused`/`failed`, origin `request`/`runtime`/`author`).
    pub fn executor(terminal: &str, origin: &str, code: &str, message: &str) -> Self {
        let coded = format!("{code}: {message}");
        let (status, cause, origin, message) = match (terminal, origin, code) {
            ("refused", ..) | (_, "request", _) => (2, 1, 6, coded),
            (_, _, "device_out_of_memory") => {
                (3, 7, 2, format!("accepted_envelope_breach: {coded}"))
            }
            (_, "runtime", _) => (3, 7, 2, coded),
            (_, "author", _) => (3, 6, 1, coded),
            _ => (3, 7, 3, coded),
        };
        Self {
            status,
            cause,
            origin,
            message,
        }
    }
    /// The executor process ended under the attempt.
    pub fn abandoned(why: &str) -> Self {
        Self {
            status: 5,
            cause: 16,
            origin: 3,
            message: format!("executor invalidated: {why}"),
        }
    }
    /// Results could not be taken into custody (encoding, checksums, storage).
    pub fn custody(why: &str) -> Self {
        Self {
            status: 3,
            cause: 10,
            origin: 4,
            message: format!("result custody failed: {why}"),
        }
    }
    pub fn encode(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
    pub fn decode(text: &str) -> Self {
        serde_json::from_str(text).unwrap_or_else(|_| Self {
            status: 3,
            cause: 7,
            origin: 3,
            message: text.into(),
        })
    }
}

pub enum Outcome {
    Completed(ResultRecord),
    Failed(String),
    Canceled,
    Paused,
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

fn update_run_reserved(connection: &Connection, actor: &str, id: &str) -> io::Result<bool> {
    connection.query_row("SELECT EXISTS(SELECT 1 FROM update_run_ids WHERE actor=?1 AND id=?2)",
        params![actor, id], |row| row.get(0)).map_err(db_error)
}

#[cfg(test)]
mod pause_tests {
    use super::*;

    /// A paused job's child that is claimed but not yet authorized has run no package code:
    /// the pause holds it, and its authorization is refused. One already running is not held.
    #[test]
    fn a_held_pause_takes_a_claimed_attempt_and_leaves_a_running_one() {
        let root = std::env::temp_dir().join(format!("cm-pause-{}", uuid::Uuid::new_v4()));
        let mut journal = Journal::open(&root).unwrap();
        let birth = |pid| ProcessBirth {
            pid,
            boot_id: "boot".into(),
            start_ticks: 1,
        };
        let mut claimed = |key: &str| {
            let invocation = Invocation {
                package: "audit/package".into(),
                input: serde_json::json!({}),
                ..Default::default()
            };
            let id = journal.accept(key, invocation).unwrap().id;
            assert!(journal.claim(&id).unwrap());
            journal
                .register_process(&id, birth(id.parse().unwrap()))
                .unwrap();
            id
        };
        let (starting, started) = (claimed("starting"), claimed("started"));
        journal.running(&started, None).unwrap();

        let held = journal.pause(&starting, "alice", true).unwrap();
        assert_eq!(
            (held.state, held.pause_actor.as_deref()),
            (State::Starting, Some("alice"))
        );
        let refused = journal.running(&starting, None).unwrap_err();
        assert_eq!(refused.kind(), io::ErrorKind::Interrupted);
        assert_eq!(
            journal.finish(&starting, Outcome::Paused).unwrap().state,
            State::Paused
        );

        // The held attempt has ended: its executor failing to shut down changes nothing.
        let late = Outcome::Failed("device executor ended".into());
        assert_eq!(
            journal.finish(&starting, late).unwrap().state,
            State::Paused
        );

        let running = journal.pause(&started, "alice", true).unwrap();
        assert_eq!((running.state, running.pause_actor), (State::Running, None));
        let _ = std::fs::remove_dir_all(root);
    }
}
