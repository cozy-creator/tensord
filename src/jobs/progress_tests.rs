//! A job's relayed child stages, read from real wire frames.
use super::*;
use crate::journal::Invocation;

/// The Runtime relays a child call's stage without its bytes; the job's progress carries the
/// bytes of the download that child is in, and nothing when it is in none or names no child.
#[test]
fn a_relayed_child_stage_carries_its_download_bytes() {
    let root = std::env::temp_dir().join(format!("cm-relay-{}", uuid::Uuid::new_v4()));
    let service = Service::open(&root.join("state"), &root.join("generations"), 1).unwrap();
    let store = Arc::new(Store::ensure(&root.join("store")).unwrap());
    let jobs = Jobs::configure(&service, store.clone(), None).unwrap();
    let engine = &service.engine;
    let accept = |id: &str| engine.accept_run("alice", id, id, Invocation::default()).unwrap().0.id;
    let (job, child) = (accept("job"), accept("job/0"));
    let call = ChildCall {
        child: child.clone(),
        module: "callee".into(),
        function: "generate".into(),
        label: "Reference".into(),
        recorded: true,
        computation: None,
        memo: None,
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
        memoized: Default::default(),
        pure: false,
        pure_callables: Default::default(),
        serving: Default::default(),
        inputs: vec![],
        calls: Mutex::new(Calls { by_index: [(0, call)].into(), ..Default::default() }),
    });
    let mut seam = Seam {
        engine,
        id: &job,
        store: &store,
        spool: &root,
        completed: 0,
        publish: true,
        job: Some((&jobs, &parent)),
        appended: HashMap::new(),
        weights: None,
    };
    let mut relay = |stage: &str, call: Option<&str>| -> Value {
        let mut event = json!({"event": "progress", "request_id": job, "kind": "progress", "stage": stage,
            "advance": 0, "step_ms": 0.0, "call_attempt": 1});
        // The Runtime always sends the key: null when the frame names no child (run 5257).
        event["call_request"] = call.map_or(Value::Null, Value::from);
        let body = serde_json::to_vec(&event).unwrap();
        let (mut runtime, mut machine) = UnixStream::pair().unwrap();
        runtime.write_all(&(body.len() as u32).to_be_bytes()).unwrap();
        runtime.write_all(&body).unwrap();
        seam.progress(&device_executor::read_frame(&mut machine).unwrap().unwrap());
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
    drop(seam);
    drop(jobs);
    drop(service);
    let _ = std::fs::remove_dir_all(root);
}
