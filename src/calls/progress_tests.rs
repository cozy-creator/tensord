//! A caller's relayed child stages, read from real wire frames.
use super::*;
use crate::{device_executor, journal::Invocation};

/// The Runtime relays a child call's stage without its bytes; the caller's progress carries the
/// bytes of the download that child is in, and nothing when it is in none or names no child.
#[test]
fn a_relayed_child_stage_carries_its_download_bytes() {
    let root = std::env::temp_dir().join(format!("cm-relay-{}", uuid::Uuid::new_v4()));
    let service = Service::open(&root.join("state"), &root.join("generations"), 1).unwrap();
    let store = Arc::new(Store::ensure(&root.join("store")).unwrap());
    let broker = Calls::configure(&service, store, None).unwrap();
    let engine = &service.engine;
    let accept = |id: &str| engine.accept_run("alice", id, id, Invocation::default()).unwrap().0.id;
    let (job, child) = (accept("job"), accept("job/0"));
    let call = ChildCall {
        child: child.clone(),
        module: "callee".into(),
        function: "generate".into(),
        label: "Reference".into(),
        recorded: true,
        request: "job/0".into(),
        settled: None,
        progress: (0, None),
    };
    let parent = Arc::new(Parent {
        id: job.clone(),
        request: "job".into(),
        actor: "alice".into(),
        spool: root.clone(),
        callables: Default::default(),
        pure: false,
        pure_callables: Default::default(),
        inputs: vec![],
        calls: Mutex::new(CallTable { by_index: [(0, call)].into(), ..Default::default() }),
    });
    let caller = Caller { calls: broker.clone(), parent };
    let mut completed = 0;
    let mut relay = |stage: &str, call: Option<&str>| -> Value {
        let mut event = json!({"event": "progress", "request_id": job, "kind": "progress", "stage": stage,
            "advance": 0, "step_ms": 0.0, "call_attempt": 1});
        // The Runtime always sends the key: null when the frame names no child (run 5257).
        event["call_request"] = call.map_or(Value::Null, Value::from);
        let body = serde_json::to_vec(&event).unwrap();
        let (mut runtime, mut machine) = UnixStream::pair().unwrap();
        runtime.write_all(&(body.len() as u32).to_be_bytes()).unwrap();
        runtime.write_all(&body).unwrap();
        observe(engine, &job, &mut completed, &device_executor::read_frame(&mut machine).unwrap().unwrap(), Some(&caller));
        serde_json::from_str(&engine.get(&job).unwrap().progress.unwrap()).unwrap()
    };
    let own = |detail: Value| engine.observe_progress(&child, 0, detail.to_string()).unwrap();

    own(json!({"stage": "downloading acme/model@1.0.0 bf16", "bytes_done": 5u64 << 30, "bytes_total": 20u64 << 30}));
    let seen = relay("Reference / downloading acme/model@1.0.0 bf16", Some("job/0"));
    assert_eq!((&seen["bytes_done"], &seen["bytes_total"]), (&json!(5u64 << 30), &json!(20u64 << 30)));
    // An older Runtime names no child: the stage alone.
    let seen = relay("Reference / downloading acme/model@1.0.0 bf16", None);
    assert!(seen.get("bytes_done").is_none(), "{seen}");
    // Past its download, the child's stage carries no bytes.
    own(json!({"stage": "preparing ", "bytes_done": 0, "bytes_total": 0}));
    let seen = relay("Reference / preparing ", Some("job/0"));
    assert!(seen.get("bytes_total").is_none(), "{seen}");
    drop(caller);
    drop(broker);
    drop(service);
    let _ = std::fs::remove_dir_all(root);
}

/// A package's `Telemetry.log` lines, read from real wire frames, reach the root run's log
/// (the one `run show` displays): the root's as written, a call's led by its function, each
/// line bounded, and past the log's bound one notice and nothing more.
#[test]
fn package_log_lines_reach_the_root_log_and_stop_at_its_bound() {
    let root = std::env::temp_dir().join(format!("cm-log-{}", uuid::Uuid::new_v4()));
    let service = Service::open(&root.join("state"), &root.join("generations"), 1).unwrap();
    let engine = &service.engine;
    let accept = |id: &str, entrypoint: &str, parent: &str| {
        let invocation = Invocation { entrypoint: entrypoint.into(), parent: parent.into(), ..Default::default() };
        engine.accept_run("alice", id, id, invocation).unwrap().0.id
    };
    let job = accept("job", "long_form", "");
    let child = accept("job/0", "caption_images", &job);
    let send = |seam: &str, request: &str, message: &str, level: &str, fields: Value| {
        let event = json!({"event": "progress", "request_id": request, "kind": "log", "name": message,
            "value": level, "fields": fields, "seq": 1, "at_unix_ms": 1});
        let body = serde_json::to_vec(&event).unwrap();
        let (mut runtime, mut machine) = UnixStream::pair().unwrap();
        runtime.write_all(&(body.len() as u32).to_be_bytes()).unwrap();
        runtime.write_all(&body).unwrap();
        observe(engine, seam, &mut 0, &device_executor::read_frame(&mut machine).unwrap().unwrap(), None);
    };
    let lines = || -> Vec<Value> {
        engine.logs(&job).unwrap().iter().map(|l| serde_json::from_slice(&l.product).unwrap()).collect()
    };
    let (own, call) = (job.as_str(), child.as_str());

    // `Telemetry.log(message, /, *, level, **fields)`: a field may be called `name` or `text`.
    send(own, &job, "description text used", "info",
        json!({"name": "Andrea", "text": "A woman with long dark hair", "index": 0}));
    send(call, &child, "caption slow", "warn", json!({"seconds": 3.5}));
    // Another run's frame on this seam is not this run's line; other observations are not logs.
    send(own, &child, "not mine", "info", json!({}));
    let body = serde_json::to_vec(&json!({"event": "progress", "request_id": job, "kind": "metric",
        "name": "steps", "value": 3, "fields": {}})).unwrap();
    let (mut runtime, mut machine) = UnixStream::pair().unwrap();
    runtime.write_all(&(body.len() as u32).to_be_bytes()).unwrap();
    runtime.write_all(&body).unwrap();
    observe(engine, own, &mut 0, &device_executor::read_frame(&mut machine).unwrap().unwrap(), None);
    assert_eq!(lines(), vec![
        json!({"level": "info", "text": "description text used index=0 name=Andrea text=\"A woman with long dark hair\""}),
        json!({"level": "warning", "text": "caption_images: caption slow seconds=3.5"}),
    ]);
    assert!(engine.logs(&child).unwrap().is_empty(), "a call's lines are its root's");

    // One line keeps at most 2 KiB of text, cut on a character boundary.
    send(own, &job, &"é".repeat(3000), "error", json!({}));
    let long = lines().pop().unwrap();
    let text = long["text"].as_str().unwrap();
    assert_eq!((long["level"].as_str(), text.len() <= 2048, text.ends_with('…')), (Some("error"), true, true));

    // The log stops at its bound: one notice, then nothing.
    for n in 0..crate::journal::PACKAGE_LOG_LINES {
        send(call, &child, "step", "info", json!({"n": n}));
    }
    let kept = lines();
    assert_eq!(kept.len() as u64, crate::journal::PACKAGE_LOG_LINES + 1);
    assert_eq!(kept.last().unwrap()["level"], "warning");
    assert!(kept.last().unwrap()["text"].as_str().unwrap().contains("dropped"), "{}", kept.last().unwrap());
    assert_eq!(kept[crate::journal::PACKAGE_LOG_LINES as usize - 1]["text"], json!(format!("caption_images: step n={}", crate::journal::PACKAGE_LOG_LINES - 4)));
    drop(service);
    let _ = std::fs::remove_dir_all(root);
}
