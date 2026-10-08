//! Real control frames, native handler and durable child bindings; no GPU is configured.
use super::*;
use crate::{
    journal::{Installation, Invocation},
    objects::Objects,
};

fn artifact() -> Value {
    json!({
        "producer_request_id": "retained-output", "output_slot": "model",
        "manifest": {"digest": format!("sha256:{}", "ab".repeat(32)), "length": 123},
        "tensorfs_receipt_digest": format!("sha256:{}", "cd".repeat(32)),
        "future_observation": true,
    })
}

#[test]
fn prefetch_choices_cross_the_native_handler_and_keep_callee_authority() {
    let root = std::env::temp_dir().join(format!("cm-prefetch-{}", uuid::Uuid::new_v4()));
    let service = Service::open(&root.join("state"), &root.join("generations"), 1).unwrap();
    let store = Arc::new(Store::ensure(&root.join("store")).unwrap());
    let objects = Arc::new(
        Objects::new(&root.join("writes"), store.clone(), service.engine.clone()).unwrap(),
    );
    let runs = Arc::new(Runs {
        service: service.clone(),
        objects,
        publisher: None,
        local: None,
        own_hub: None,
        jobs: Default::default(),
    });
    let jobs = Jobs::configure(&service, store, Some(&runs)).unwrap();
    let interface = json!({"entrypoints": [{"name": "render", "models": [
        {"path": "render.models.network", "class": "fixture.Network"}
    ]}]});
    service
        .engine
        .bind_installation(Installation {
            actor: "alice".into(),
            alias: "installed".into(),
            generation: "generation".into(),
            package: "local/fixture".into(),
            release: "1".into(),
            interface: serde_json::to_vec(&interface).unwrap(),
        })
        .unwrap();
    let (record, _) = service
        .engine
        .accept_run(
            "alice",
            "parent",
            "parent",
            Invocation {
                job: true,
                ..Default::default()
            },
        )
        .unwrap();
    let context = json!({
        "installation": "installed", "owner": "alice", "binding_revision": "choices-1",
        "attention_kernel": "", "models": [],
    });
    service
        .engine
        .with_journal(|journal| {
            journal.bind_job_context(&record.id, &serde_json::to_vec(&context).unwrap())
        })
        .unwrap();
    let parent = Parent {
        id: record.id.clone(),
        request: "parent".into(),
        actor: "alice".into(),
        spool: root.clone(),
        callables: [(
            ("fixture".into(), "render".into()),
            Some((String::new(), "render".into())),
        )]
        .into(),
        memoized: Default::default(),
        pure: false,
        pure_callables: Default::default(),
        serving: Default::default(),
        inputs: vec![],
        calls: Default::default(),
    };
    let ask = |export: &str, payload: &str| {
        let body = serde_json::to_vec(&json!({
            "event": "request", "kind": "model_prefetch", "seq": 7,
            "module": "fixture", "export": export, "payload": payload, "future_field": 1,
        }))
        .unwrap();
        let (mut client, mut server) = UnixStream::pair().unwrap();
        client
            .write_all(&(body.len() as u32).to_be_bytes())
            .unwrap();
        client.write_all(&body).unwrap();
        let frame = device_executor::read_frame(&mut server).unwrap().unwrap();
        jobs.seam(&parent, &frame).unwrap().0
    };
    // Empty payload is an older peer. Null/omitted choices keep inherited/default selection.
    for payload in [
        String::new(),
        "{}".into(),
        json!({"network": null}).to_string(),
        json!({"network": artifact()}).to_string(),
    ] {
        assert!(ask("render", &payload).ok, "{payload}");
    }
    assert_eq!(ask("undeclared", "{}").code, "child_undeclared");
    assert_eq!(
        ask("render", &json!({"another_slot": artifact()}).to_string()).code,
        "invalid_request"
    );
    for payload in [
        "[]".to_string(), "{bad".to_string(), "{\"network\":null,\"network\":null}".to_string(),
        json!({"network": "not-an-artifact"}).to_string(),
        json!({"network": {"manifest": {"digest": "sha256:bad", "length": 123}}}).to_string(),
        json!({"network": {"manifest": {"digest": artifact()["manifest"]["digest"], "length": "123"}}}).to_string(),
        json!({"network": {"manifest": {"digest": artifact()["manifest"]["digest"], "length": 0}}}).to_string(),
    ] {
        assert_eq!(ask("render", &payload).code, "child_call_refused", "{payload}");
    }
    // An installation held by Alice cannot be used through another actor's job context.
    let (other, _) = service
        .engine
        .accept_run(
            "bob",
            "parent",
            "parent",
            Invocation {
                job: true,
                ..Default::default()
            },
        )
        .unwrap();
    service
        .engine
        .with_journal(|journal| {
            journal.bind_job_context(&other.id, &serde_json::to_vec(&context).unwrap())
        })
        .unwrap();
    let denied = runs.prefetch(&other, "", "render", vec![]).unwrap_err();
    assert_eq!(denied.code, "child_call_refused");
    assert_eq!(
        service.engine.nonterminal(usize::MAX).unwrap().len(),
        2,
        "a hint must not create a child execution"
    );

    let unsupported = Jobs::configure(
        &service,
        Arc::new(Store::open(&root.join("store")).unwrap()),
        None,
    )
    .unwrap();
    let reply = unsupported
        .model_prefetch(
            &parent,
            &Frame {
                module: "fixture".into(),
                export: "render".into(),
                payload: "{}".into(),
                ..Default::default()
            },
        )
        .unwrap_err();
    assert_eq!(reply.0, "model_prefetch_unsupported");
    drop(unsupported);
    drop(jobs);
    drop(runs);
    drop(service);
    std::fs::remove_dir_all(root).unwrap();
}
