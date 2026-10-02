//! Durable acceptance and transitions. Observation has no lifecycle side effects.
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    fs, io,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Invocation {
    pub package: String,
    pub generation: String,
    pub module: String,
    pub entrypoint: String,
    pub input: Value,
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
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Execution {
    pub id: String,
    pub idempotency_key: String,
    pub invocation: Invocation,
    pub state: State,
    pub revision: u64,
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

pub struct Journal {
    connection: Connection,
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
              id INTEGER PRIMARY KEY, idempotency_key TEXT NOT NULL UNIQUE,
              invocation TEXT NOT NULL, record TEXT NOT NULL, state TEXT NOT NULL,
              updated_ms INTEGER NOT NULL);",
            )
            .map_err(db_error)?;
        Ok(Self { connection })
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
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        let prior: Option<String> = tx
            .query_row(
                "SELECT record FROM executions WHERE idempotency_key=?1",
                [key],
                |row| row.get(0),
            )
            .optional()
            .map_err(db_error)?;
        if let Some(prior) = prior {
            let execution: Execution = serde_json::from_str(&prior).map_err(db_error)?;
            if execution.invocation != invocation {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "idempotency key already identifies a different invocation",
                ));
            }
            return Ok(execution);
        }
        tx.execute("INSERT INTO executions(idempotency_key,invocation,record,state,updated_ms) VALUES(?1,?2,'','queued',?3)", params![key, encoded(&invocation)?, timestamp()]).map_err(db_error)?;
        let execution = Execution {
            id: tx.last_insert_rowid().to_string(),
            idempotency_key: key.into(),
            invocation,
            state: State::Queued,
            revision: 1,
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

    fn update(
        &mut self,
        id: &str,
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
        if change(&mut record)? {
            record.revision += 1;
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

    pub fn cancel(&mut self, id: &str, actor: &str) -> io::Result<Execution> {
        if actor.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cancellation requires an actor",
            ));
        }
        self.update(id, |record| {
            if record.state.terminal() || record.cancel_actor.is_some() {
                return Ok(false);
            }
            record.cancel_actor = Some(actor.into());
            if record.state == State::Queued {
                record.state = State::Canceled;
            }
            Ok(true)
        })
    }

    /// Counts actual completed work, never mere enqueues or repeated retry churn.
    pub fn progress(&mut self, id: &str, completed_units: u64, detail: String) -> io::Result<()> {
        self.update(id, |record| {
            if record.state.terminal() || completed_units <= record.completed_units {
                return Ok(false);
            }
            record.completed_units = completed_units;
            record.progress = Some(detail);
            Ok(true)
        })?;
        Ok(())
    }

    /// Caller must prove process termination and durable artifact custody first.
    pub fn finish(&mut self, id: &str, outcome: Outcome) -> io::Result<Execution> {
        self.update(id, |record| {
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
            Ok(true)
        })
    }
}

pub enum Outcome {
    Completed(ResultRecord),
    Failed(String),
    Canceled,
}
