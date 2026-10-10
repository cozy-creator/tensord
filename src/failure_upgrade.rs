//! One-time conversion to typed failure codes. Serving paths never parse message prefixes.
//! Original journal and cached terminal bytes are retained in the same transaction.
use crate::archive;
use prost::Message;
use rusqlite::{params, Connection, OptionalExtension, Transaction};
use serde_json::{json, Value};
use std::io;
use tensorfs_core::sha256;

const MARKER: &str = "structured_failure_codes_v1";

fn error(value: impl std::fmt::Display) -> io::Error {
    io::Error::other(value.to_string())
}

// Only this upgrader interprets the old message format. Distinct inner codes and
// code-like text elsewhere in the message are preserved.
fn old_reason(message: &str, fallback: &str) -> (String, String) {
    let code = message
        .split_once(": ")
        .map(|(code, _)| code)
        .filter(|code| {
            !code.is_empty()
                && code
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b == b'_' || b == b'.')
        })
        .unwrap_or(fallback);
    let prefix = format!("{code}: ");
    let mut detail = message;
    while let Some(rest) = detail.strip_prefix(&prefix) {
        detail = rest;
    }
    (code.to_owned(), detail.to_owned())
}

fn failure(text: &str) -> io::Result<Option<String>> {
    if crate::journal::Failure::decode(text).is_ok() {
        return Ok(None); // Current typed messages are literal, even when they contain a prefix.
    }
    let mut value: Value = serde_json::from_str(text).unwrap_or(Value::Null);
    // The old decoder treated anything outside its four-field shape as literal text.
    if !["status", "cause", "origin"]
        .iter()
        .all(|field| value[*field].as_u64().is_some_and(|n| n <= u8::MAX as u64))
        || !value["message"].is_string()
    {
        value = json!({"status":3,"cause":7,"origin":3,"message":text});
    }
    let message = value["message"]
        .as_str()
        .ok_or_else(|| error("stored failure has no message"))?;
    let (code, detail) = old_reason(message, "failed");
    value["code"] = json!(code);
    value["message"] = json!(detail);
    serde_json::to_string(&value).map(Some).map_err(error)
}

fn call(bytes: &[u8]) -> io::Result<Option<Vec<u8>>> {
    let mut value: Value = serde_json::from_slice(bytes).map_err(error)?;
    if value.get("error_code").is_some() {
        return Ok(None);
    }
    let Some(message) = value["error"].as_str().filter(|s| !s.is_empty()) else {
        return Ok(None);
    };
    let (code, detail) = old_reason(message, "failed");
    value["error_code"] = json!(code);
    value["error"] = json!(detail);
    serde_json::to_vec(&value).map(Some).map_err(error)
}

fn outcome(value: &mut archive::AttemptOutcome) -> io::Result<bool> {
    let mut body: Value = serde_json::from_slice(&value.outcome_canonical_bytes).map_err(error)?;
    if body["status"].as_u64() == Some(1) || body.get("error_code").is_some() {
        return Ok(false);
    }
    let message = body["safe_message"]
        .as_str()
        .ok_or_else(|| error("cached failed outcome has no safe_message"))?;
    let fallback = if body["status"].as_u64() == Some(4) {
        "canceled"
    } else {
        "failed"
    };
    let (code, detail) = old_reason(message, fallback);
    body["error_code"] = json!(code);
    body["safe_message"] = json!(detail);
    if let Some(cause) = body.get_mut("cause").and_then(Value::as_object_mut) {
        // The numeric cause and origin retain their original meanings.
        cause.insert("detail".into(), json!(detail));
    }
    value.outcome_canonical_bytes = serde_json_canonicalizer::to_vec(&body).map_err(error)?;
    value.outcome_digest = sha256::digest(&value.outcome_canonical_bytes).to_vec();
    value.outcome_id = format!("out-{}", sha256::hex(&value.outcome_digest));
    Ok(true)
}

fn terminal(raw_outcome: &[u8], raw_events: &[u8]) -> io::Result<Option<(Vec<u8>, Vec<u8>)>> {
    // Use the archive structs directly to retain their custody metadata fields.
    let mut result = archive::AttemptOutcome::decode(raw_outcome).map_err(error)?;
    let old_id = result.outcome_id.clone();
    let old_digest = format!("sha256:{}", sha256::hex(&result.outcome_digest));
    let changed = outcome(&mut result)?;
    let mut events = archive::MachineExecutionEventPage::decode(raw_events).map_err(error)?;
    let mut events_changed = false;
    for event in &mut events.events {
        if let Some(embedded) = &mut event.outcome {
            events_changed |= outcome(embedded)?;
        }
        if changed && event.kind == "outcome" {
            let mut body: Value =
                serde_json::from_slice(&event.body_canonical_bytes).map_err(error)?;
            if body["outcome_id"].as_str() == Some(old_id.as_str())
                && body["outcome_digest"].as_str() == Some(old_digest.as_str())
            {
                body["outcome_id"] = json!(result.outcome_id);
                body["outcome_digest"] =
                    json!(format!("sha256:{}", sha256::hex(&result.outcome_digest)));
                event.body_canonical_bytes =
                    serde_json_canonicalizer::to_vec(&body).map_err(error)?;
                events_changed = true;
            } else {
                return Err(error(
                    "cached terminal reference disagrees with its outcome",
                ));
            }
        }
        if event.kind == "call" || event.kind == "machine.call" {
            if let Some(updated) = call(&event.body_canonical_bytes)? {
                event.body_canonical_bytes = updated;
                events_changed = true;
            }
        }
    }
    Ok((changed || events_changed).then(|| (result.encode_to_vec(), events.encode_to_vec())))
}

fn backup(
    tx: &Transaction<'_>,
    kind: &str,
    id: i64,
    sequence: i64,
    raw: &[u8],
    auxiliary: Option<&[u8]>,
) -> io::Result<()> {
    tx.execute(
        "INSERT INTO failure_upgrade_originals(kind,execution,sequence,record,auxiliary) VALUES(?1,?2,?3,?4,?5)",
        params![kind, id, sequence, raw, auxiliary],
    ).map_err(error)?;
    Ok(())
}

pub(crate) fn upgrade(connection: &mut Connection) -> io::Result<()> {
    let tx = connection.transaction().map_err(error)?;
    let done: Option<String> = tx
        .query_row(
            "SELECT value FROM machine_metadata WHERE key=?1",
            [MARKER],
            |r| r.get(0),
        )
        .optional()
        .map_err(error)?;
    if done.is_some() {
        return Ok(());
    }
    tx.execute_batch(
        "CREATE TABLE IF NOT EXISTS failure_upgrade_originals(
        kind TEXT NOT NULL,execution INTEGER NOT NULL,sequence INTEGER NOT NULL,
        record BLOB NOT NULL,auxiliary BLOB,PRIMARY KEY(kind,execution,sequence));",
    )
    .map_err(error)?;
    let rows: Vec<(i64, String)> = {
        let mut query = tx
            .prepare(
                "SELECT id,record FROM executions WHERE json_type(record,'$.failure')='text'
                OR (json_extract(record,'$.state')='failed'
                    AND COALESCE(json_type(record,'$.failure'),'null')='null')",
            )
            .map_err(error)?;
        let rows = query
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .map_err(error)?;
        rows.collect::<Result<_, _>>().map_err(error)?
    };
    for (id, raw) in rows {
        let mut record: Value = serde_json::from_str(&raw).map_err(error)?;
        if let Some(updated) = failure(
            record["failure"]
                .as_str()
                // Before the typed contract, failed rows without a reason used this
                // public detail. Supply it once here, never in a serving fallback.
                .unwrap_or("the run failed without a recorded reason"),
        )? {
            record["failure"] = json!(updated);
            backup(&tx, "execution", id, 0, raw.as_bytes(), None)?;
            tx.execute(
                "UPDATE executions SET record=?1 WHERE id=?2",
                params![serde_json::to_string(&record).map_err(error)?, id],
            )
            .map_err(error)?;
        }
    }
    let calls: Vec<(i64, i64, Vec<u8>)> = {
        let mut query = tx
            .prepare("SELECT execution,sequence,record FROM run_calls")
            .map_err(error)?;
        let rows = query
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .map_err(error)?;
        rows.collect::<Result<_, _>>().map_err(error)?
    };
    for (id, sequence, raw) in calls {
        if let Some(updated) = call(&raw)? {
            backup(&tx, "call", id, sequence, &raw, None)?;
            tx.execute(
                "UPDATE run_calls SET record=?1 WHERE execution=?2 AND sequence=?3",
                params![updated, id, sequence],
            )
            .map_err(error)?;
        }
    }
    let terminals: Vec<(i64, Vec<u8>, Vec<u8>)> = {
        let mut query = tx
            .prepare("SELECT execution,outcome,events FROM public_terminals")
            .map_err(error)?;
        let rows = query
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .map_err(error)?;
        rows.collect::<Result<_, _>>().map_err(error)?
    };
    for (id, raw, events) in terminals {
        if let Some((updated, updated_events)) = terminal(&raw, &events)? {
            backup(&tx, "terminal", id, 0, &raw, Some(&events))?;
            tx.execute(
                "UPDATE public_terminals SET outcome=?1,events=?2 WHERE execution=?3",
                params![updated, updated_events, id],
            )
            .map_err(error)?;
        }
    }
    tx.execute(
        "INSERT INTO machine_metadata(key,value) VALUES(?1,'1')",
        [MARKER],
    )
    .map_err(error)?;
    tx.commit().map_err(error)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn database() -> Connection {
        let c = Connection::open_in_memory().unwrap();
        c.execute_batch(
            "CREATE TABLE machine_metadata(key TEXT PRIMARY KEY,value TEXT);
            CREATE TABLE executions(id INTEGER PRIMARY KEY,record TEXT);
            CREATE TABLE run_calls(execution INTEGER,sequence INTEGER,record BLOB);
            CREATE TABLE public_terminals(execution INTEGER,outcome BLOB,events BLOB);",
        )
        .unwrap();
        c
    }

    fn cached() -> (Vec<u8>, Vec<u8>) {
        let body = json!({"status":3,"cause":{"code":7,"origin":2,"detail":"model_choice_absent: model_choice_absent: missing"},
            "safe_message":"model_choice_absent: model_choice_absent: missing","unknown":{"kept":true}});
        let canonical = serde_json_canonicalizer::to_vec(&body).unwrap();
        let digest = sha256::digest(&canonical).to_vec();
        let outcome = archive::AttemptOutcome {
            outcome_id: format!("out-{}", sha256::hex(&digest)),
            outcome_digest: digest,
            outcome_canonical_bytes: canonical,
            record_owner_epoch: 42,
            placement_id: "keep-placement".into(),
            ..Default::default()
        };
        let event = archive::MachineExecutionEvent {
            sequence: 17,
            kind: "outcome".into(),
            outcome: Some(outcome.clone()),
            body_canonical_bytes: serde_json::to_vec(
                &json!({"state":"failed","outcome_id":outcome.outcome_id,
                "outcome_digest":format!("sha256:{}",sha256::hex(&outcome.outcome_digest))}),
            )
            .unwrap(),
            ..Default::default()
        };
        (
            outcome.encode_to_vec(),
            archive::MachineExecutionEventPage {
                events: vec![event],
                next_after: 17,
                head_sequence: 17,
                ..Default::default()
            }
            .encode_to_vec(),
        )
    }

    #[test]
    fn migrates_all_projections_atomically_and_keeps_original_bytes() {
        let mut c = database();
        let failure = json!({"status":3,"cause":6,"origin":1,"message":"model_choice_absent: model_choice_absent: missing","unknown":17}).to_string();
        let record = json!({"failure":failure,"untouched":true}).to_string();
        let original_record = record.as_bytes().to_vec();
        let call = br#"{"error":"model_choice_absent: model_choice_absent: missing","status":"failed","unknown":17}"#;
        let (outcome, events) = cached();
        c.execute("INSERT INTO executions VALUES(1,?1)", [&record])
            .unwrap();
        c.execute("INSERT INTO run_calls VALUES(1,7,?1)", [call.as_slice()])
            .unwrap();
        c.execute(
            "INSERT INTO public_terminals VALUES(1,?1,?2)",
            params![outcome, events],
        )
        .unwrap();
        upgrade(&mut c).unwrap();
        let current: String = c
            .query_row("SELECT record FROM executions", [], |r| r.get(0))
            .unwrap();
        let record: Value = serde_json::from_str(&current).unwrap();
        let failure: Value = serde_json::from_str(record["failure"].as_str().unwrap()).unwrap();
        let typed = crate::journal::Failure::decode(record["failure"].as_str().unwrap()).unwrap();
        assert_eq!(typed.code, "model_choice_absent");
        assert_eq!(typed.message, "missing");
        assert_eq!(failure["code"], "model_choice_absent");
        assert_eq!(failure["message"], "missing");
        assert_eq!(failure["unknown"], 17);
        assert_eq!(record["untouched"], true);
        let raw: Vec<u8> = c
            .query_row("SELECT record FROM run_calls", [], |r| r.get(0))
            .unwrap();
        let call: Value = serde_json::from_slice(&raw).unwrap();
        assert_eq!(call["error_code"], "model_choice_absent");
        assert_eq!(call["error"], "missing");
        let (raw, page): (Vec<u8>, Vec<u8>) = c
            .query_row("SELECT outcome,events FROM public_terminals", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap();
        let result = archive::AttemptOutcome::decode(raw.as_slice()).unwrap();
        let body: Value = serde_json::from_slice(&result.outcome_canonical_bytes).unwrap();
        assert_eq!(body["error_code"], "model_choice_absent");
        assert_eq!(body["safe_message"], "missing");
        assert_eq!(
            body["cause"],
            json!({"code":7,"origin":2,"detail":"missing"})
        );
        assert_eq!(result.record_owner_epoch, 42);
        assert_eq!(result.placement_id, "keep-placement");
        assert_eq!(
            result.outcome_digest,
            sha256::digest(&result.outcome_canonical_bytes)
        );
        let page = archive::MachineExecutionEventPage::decode(page.as_slice()).unwrap();
        assert_eq!(page.events[0].outcome.as_ref(), Some(&result));
        let reference: Value =
            serde_json::from_slice(&page.events[0].body_canonical_bytes).unwrap();
        assert_eq!(reference["outcome_id"], result.outcome_id);
        assert_eq!(
            reference["outcome_digest"],
            format!("sha256:{}", sha256::hex(&result.outcome_digest))
        );
        let original: (Vec<u8>, Vec<u8>) = c
            .query_row(
                "SELECT record,auxiliary FROM failure_upgrade_originals WHERE kind='terminal'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(original, (outcome, events));
        assert_eq!(
            c.query_row(
                "SELECT record FROM failure_upgrade_originals WHERE kind='execution'",
                [],
                |r| r.get::<_, Vec<u8>>(0)
            )
            .unwrap(),
            original_record
        );
        assert_eq!(
            c.query_row("SELECT record FROM failure_upgrade_originals WHERE kind='call'", [], |r| r.get::<_, Vec<u8>>(0)).unwrap(),
            br#"{"error":"model_choice_absent: model_choice_absent: missing","status":"failed","unknown":17}"#
        );
        upgrade(&mut c).unwrap();
        let repeat: Vec<u8> = c
            .query_row("SELECT outcome FROM public_terminals", [], |r| r.get(0))
            .unwrap();
        assert_eq!(repeat, raw);
        assert_eq!(
            c.query_row("SELECT count(*) FROM failure_upgrade_originals", [], |r| {
                r.get::<_, i64>(0)
            })
            .unwrap(),
            3
        );
    }

    #[test]
    fn malformed_cached_terminal_rolls_back_every_projection_and_marker() {
        let mut c = database();
        let record = json!({"failure":"plain old failure"}).to_string();
        c.execute("INSERT INTO executions VALUES(1,?1)", [&record])
            .unwrap();
        c.execute("INSERT INTO public_terminals VALUES(1,x'ff',x'ff')", [])
            .unwrap();
        assert!(upgrade(&mut c).is_err());
        assert_eq!(
            c.query_row("SELECT record FROM executions", [], |r| r
                .get::<_, String>(0))
                .unwrap(),
            record
        );
        assert_eq!(
            c.query_row("SELECT count(*) FROM machine_metadata", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            0
        );
    }

    #[test]
    fn failed_rows_without_a_reason_gain_the_historical_detail_only_once() {
        let mut c = database();
        let current = json!({"state":"failed","failure":crate::journal::Failure::machine(
            "typed", "typed: literal body").encode(),"untouched":17})
        .to_string();
        let rows = [
            r#"{"state":"failed","failure":null,"untouched":1}"#.to_owned(),
            r#"{"state":"failed","untouched":2}"#.to_owned(),
            r#"{"state":"completed","failure":null,"untouched":3}"#.to_owned(),
            r#"{"state":"running","failure":null,"untouched":4}"#.to_owned(),
            current,
        ];
        for (index, raw) in rows.iter().enumerate() {
            c.execute(
                "INSERT INTO executions VALUES(?1,?2)",
                params![index + 1, raw],
            )
            .unwrap();
        }
        upgrade(&mut c).unwrap();
        for (index, original) in rows.iter().enumerate() {
            let raw: String = c
                .query_row(
                    "SELECT record FROM executions WHERE id=?1",
                    [index + 1],
                    |r| r.get(0),
                )
                .unwrap();
            if index >= 2 {
                assert_eq!(
                    &raw, original,
                    "current and nonfailed rows are byte-identical"
                );
                continue;
            }
            let record: Value = serde_json::from_str(&raw).unwrap();
            let failure =
                crate::journal::Failure::decode(record["failure"].as_str().unwrap()).unwrap();
            assert_eq!(failure.code, "failed");
            assert_eq!(failure.message, "the run failed without a recorded reason");
            assert_eq!((failure.status, failure.cause, failure.origin), (3, 7, 3));
            assert_eq!(record["untouched"], index + 1);
            let backup: Vec<u8> = c.query_row("SELECT record FROM failure_upgrade_originals WHERE kind='execution' AND execution=?1",[index + 1],|r|r.get(0)).unwrap();
            assert_eq!(backup, original.as_bytes());
        }
        upgrade(&mut c).unwrap();
        assert_eq!(
            c.query_row("SELECT count(*) FROM failure_upgrade_originals", [], |r| {
                r.get::<_, i64>(0)
            })
            .unwrap(),
            2
        );
    }

    #[test]
    fn legacy_json_detail_with_a_code_key_is_not_mistaken_for_current_failure() {
        let mut c = database();
        let literal = r#"{"code":"remote_error","message":"plain old detail"}"#;
        let record = json!({"state":"failed","failure":literal}).to_string();
        c.execute("INSERT INTO executions VALUES(1,?1)", [&record])
            .unwrap();
        upgrade(&mut c).unwrap();
        let raw: String = c
            .query_row("SELECT record FROM executions", [], |r| r.get(0))
            .unwrap();
        let upgraded: Value = serde_json::from_str(&raw).unwrap();
        let failure =
            crate::journal::Failure::decode(upgraded["failure"].as_str().unwrap()).unwrap();
        assert_eq!((failure.status, failure.cause, failure.origin), (3, 7, 3));
        assert_eq!(failure.code, "failed");
        assert_eq!(failure.message, literal);
        let original: Vec<u8> = c
            .query_row(
                "SELECT record FROM failure_upgrade_originals WHERE kind='execution'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(original, record.as_bytes());
        upgrade(&mut c).unwrap();
        assert_eq!(
            c.query_row("SELECT record FROM executions", [], |r| r
                .get::<_, String>(0))
                .unwrap(),
            raw
        );
    }

    #[test]
    fn current_messages_and_distinct_inner_causes_are_literal() {
        let current =
            json!({"code":"typed","status":3,"cause":7,"origin":2,"message":"typed: literal"})
                .to_string();
        assert_eq!(failure(&current).unwrap(), None);
        assert_eq!(
            old_reason("outer: outer: inner: detail", "failed"),
            ("outer".into(), "inner: detail".into())
        );
        assert_eq!(
            old_reason("https://example.test/path: detail", "failed"),
            ("failed".into(), "https://example.test/path: detail".into())
        );
        assert_eq!(
            old_reason("literal contains outer: detail", "failed"),
            ("failed".into(), "literal contains outer: detail".into())
        );
        let plain: Value = serde_json::from_str(&failure("legacy text").unwrap().unwrap()).unwrap();
        assert_eq!(
            plain,
            json!({"status":3,"cause":7,"origin":3,"code":"failed","message":"legacy text"})
        );
    }
}
