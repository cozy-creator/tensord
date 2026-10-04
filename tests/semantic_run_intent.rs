//! Real TLS/API and journal checks for semantic reattachment and exact application integers.
//! No model inference is simulated or qualified by these tests.
use cozy_machine::{
    api::{
        self,
        capability::{self, Grant},
        pb, v1, MachineIdentity,
    },
    journal::{Invocation, Outcome, ResultRecord},
    machine_api::NativeBackend,
    objects::Objects,
    runs::Runs,
    service::Service,
};
use ed25519_dalek::{Signer, SigningKey};
use std::{
    fs,
    path::PathBuf,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tonic::transport::{Certificate, Channel, ClientTlsConfig, Endpoint};

struct Machine {
    root: PathBuf,
    service: Arc<Service>,
    objects: Arc<Objects>,
    store: Arc<tensorfs_core::store::Store>,
    client: v1::machine_client::MachineClient<Channel>,
    token: String,
    legacy: pb::pod_host_client::PodHostClient<Channel>,
    claim: pb::Claim,
    task: tokio::task::JoinHandle<()>,
    actor: String,
    lifecycle: Arc<cozy_machine::machine::lifecycle::Lifecycle>,
}
impl Drop for Machine {
    fn drop(&mut self) {
        self.task.abort();
        let _ = self.service.stop();
        let _ = fs::remove_dir_all(&self.root);
    }
}
impl Machine {
    async fn open() -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target/semantic-intent")
            .join(uuid::Uuid::new_v4().to_string());
        let service = Service::open(&root.join("state"), &root.join("generations"), 1).unwrap();
        let store = Arc::new(tensorfs_core::store::Store::ensure(&root.join("store")).unwrap());
        let objects = Arc::new(
            Objects::new(&root.join("writes"), store.clone(), service.engine.clone()).unwrap(),
        );
        let signer = SigningKey::from_bytes(&[71; 32]);
        let actor = tensorfs_core::sha256::hex(signer.verifying_key().as_bytes());
        let mut identity =
            MachineIdentity::ephemeral("intent".into(), vec![signer.verifying_key()], vec![7; 32])
                .unwrap();
        let lifecycle =
            cozy_machine::machine::lifecycle::Lifecycle::open(root.join("idle.json"), false, true)
                .unwrap();
        identity.lifecycle = Some(lifecycle.clone());
        let pem = identity.cert_pem.clone();
        let claim = pb::Claim { worker_id: identity.authority.worker_id.clone(), worker_boot_id: identity.authority.boot_id.clone(),
            record_owner_epoch: 1, proof: signer.sign(&identity.authority.transcript(1).unwrap()).to_bytes().to_vec(), ..Default::default() };
        let uploads =
            api::workspaces::WorkspaceUploads::open(&root.join("uploads"), store.clone()).unwrap();
        let mut backend = NativeBackend::new(
            service.clone(),
            identity.authority.clone(),
            store.clone(),
            uploads,
        );
        backend.runs = Some(Arc::new(Runs {
            service: service.clone(),
            objects: objects.clone(),
            publisher: None,
            local: None,
            own_hub: None,
            jobs: Default::default(),
        }));
        let token = capability::mint(
            &signer,
            Grant {
                machine: "intent".into(),
                action: capability::MACHINE.into(),
                expires: now() + 600,
                ..Default::default()
            },
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            api::serve(listener, identity, Arc::new(backend))
                .await
                .unwrap()
        });
        let channel = Endpoint::from_shared(format!("https://{address}"))
            .unwrap()
            .tls_config(
                ClientTlsConfig::new()
                    .ca_certificate(Certificate::from_pem(pem))
                    .domain_name("localhost"),
            )
            .unwrap()
            .connect()
            .await
            .unwrap();
        Self {
            root,
            service,
            objects,
            store,
            client: v1::machine_client::MachineClient::new(channel.clone()),
            legacy: pb::pod_host_client::PodHostClient::new(channel),
            claim,
            token,
            task,
            actor,
            lifecycle,
        }
    }
    async fn run(
        &mut self,
        id: &str,
        spec: Option<v1::RunSpec>,
    ) -> Result<Vec<v1::RunEvent>, tonic::Status> {
        let request = authorized(
            v1::RunRequest {
                id: id.into(),
                spec,
                after: 0,
            },
            &self.token,
        );
        let mut stream = self.client.run(request).await?.into_inner();
        let mut events = vec![];
        while let Some(event) = tokio::time::timeout(Duration::from_secs(30), stream.message())
            .await
            .expect("test stream stalled")?
        {
            events.push(event);
        }
        Ok(events)
    }
    fn input(&self) -> v1::InputFile {
        let data = b"accepted input";
        let digest = format!("sha256:{}", tensorfs_core::sha256::hex_digest(data));
        let mut writer = self
            .objects
            .begin(&self.actor, &digest, data.len() as u64, 0)
            .unwrap();
        writer.append(data).unwrap();
        writer.finish().unwrap();
        v1::InputFile {
            field: "reference".into(),
            digest,
            length: data.len() as u64,
            media_type: "application/octet-stream".into(),
            order: 0,
        }
    }
}
fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}
fn authorized<T>(value: T, token: &str) -> tonic::Request<T> {
    let mut request = tonic::Request::new(value);
    request.metadata_mut().insert(
        "authorization",
        format!("Cozy-Cap {token}").parse().unwrap(),
    );
    request
}
fn warm(input: v1::InputFile) -> v1::RunSpec {
    v1::RunSpec {
        kind: v1::RunKind::Warm as i32,
        payload: br#" { "nested":{"b":2,"a":1}, "seed":9007199254740993 } "#.to_vec(),
        inputs: vec![input],
        binding_revision: "old-hint".into(),
        publication: "old-publication".into(),
        owner: "org".into(),
        known_results: vec![v1::MemoResult::default()],
        hub: Some(v1::HubAccess {
            origin: "https://hub.example.test/".into(),
            token: "old-token".into(),
            expires_at: now() + 600,
            ca_der: vec![1],
            object_hosts: vec!["old.example.test".into()],
        }),
        providers: Some(v1::ProviderAccess {
            huggingface: "old-provider".into(),
            civitai: "old-civitai".into(),
        }),
        ..Default::default()
    }
}
fn number(events: &[v1::RunEvent]) -> u64 {
    events
        .iter()
        .find_map(|e| match &e.event {
            Some(v1::run_event::Event::State(s)) if s.number > 0 => Some(s.number),
            _ => None,
        })
        .unwrap()
}

#[tokio::test]
async fn formatted_refreshed_access_attaches_without_old_input_blobs_but_a_changed_seed_conflicts()
{
    let mut machine = Machine::open().await;
    let input = machine.input();
    let original = warm(input.clone());
    let first = machine.run("same", Some(original.clone())).await.unwrap();
    assert!(
        matches!(first.last().unwrap().event,Some(v1::run_event::Event::Outcome(ref o)) if o.status=="succeeded")
    );
    fs::remove_file(
        machine
            .store
            .object_path(input.digest.trim_start_matches("sha256:")),
    )
    .unwrap();
    let mut repeated = original;
    repeated.payload = br#"{"seed":9007199254740993,"nested":{"a":1,"b":2}}"#.to_vec();
    repeated.binding_revision = "new-hint".into();
    repeated.publication = "new-publication".into();
    repeated.known_results.clear();
    let hub = repeated.hub.as_mut().unwrap();
    hub.origin = "https://hub.example.test".into();
    hub.expires_at = 1;
    hub.token = "new-token".into();
    hub.ca_der = vec![2];
    hub.object_hosts.clear();
    repeated.providers = Some(v1::ProviderAccess {
        huggingface: "new-provider".into(),
        civitai: "new-civitai".into(),
    });
    let activation = machine.lifecycle.begin_activation().unwrap();
    let attached = machine.run("same", Some(repeated.clone())).await.unwrap();
    assert_eq!(number(&first), number(&attached));
    let mut changed = repeated.clone();
    changed.payload = br#"{"seed":9007199254740992,"nested":{"a":1,"b":2}}"#.to_vec();
    assert_eq!(
        machine.run("same", Some(changed)).await.unwrap_err().code(),
        tonic::Code::AlreadyExists
    );
    assert_eq!(
        machine
            .run("fresh-frozen", Some(repeated.clone()))
            .await
            .unwrap_err()
            .code(),
        tonic::Code::Unavailable
    );
    drop(activation);
    assert_eq!(
        machine
            .run("fresh-expired", Some(repeated))
            .await
            .unwrap_err()
            .code(),
        tonic::Code::FailedPrecondition
    );
    assert!(machine
        .service
        .engine
        .get_public(&machine.actor, "fresh-expired")
        .is_err());
}

#[tokio::test]
async fn native_outcome_preserves_exact_application_uint64_and_float_kind() {
    let mut machine = Machine::open().await;
    let digest = format!(
        "sha256:{}",
        tensorfs_core::sha256::hex_digest(b"result-intent")
    );
    let record = machine
        .service
        .engine
        .accept_run(
            &machine.actor,
            "result",
            &digest,
            Invocation {
                package: "local/precision".into(),
                input: serde_json::json!({}),
                ..Default::default()
            },
        )
        .unwrap()
        .0;
    machine
        .service
        .engine
        .end_preparation(
            &record.id,
            Outcome::Completed(ResultRecord {
                value: serde_json::json!({"seed":u64::MAX,"float":1.0}),
                artifacts: vec![],
                asset_bindings: vec![],
            }),
        )
        .unwrap();
    let events = machine.run("result", None).await.unwrap();
    let outcome = events
        .iter()
        .find_map(|e| match &e.event {
            Some(v1::run_event::Event::Outcome(o)) => Some(o),
            _ => None,
        })
        .unwrap();
    let result: serde_json::Value = serde_json::from_slice(&outcome.result).unwrap();
    assert_eq!(result["seed"].as_u64(), Some(u64::MAX));
    assert!(result["float"].as_number().unwrap().is_f64());
}


#[tokio::test]
async fn legacy_existing_intent_replays_during_activation_but_fresh_work_cannot_enter() {
    let mut machine = Machine::open().await;
    let generation = "d".repeat(32);
    let directory = machine.service.catalog.root().join(&generation);
    let python = directory.join("env/bin/python");
    fs::create_dir_all(python.parent().unwrap()).unwrap();
    std::os::unix::fs::symlink("/usr/bin/python3", &python).unwrap();
    let interface = serde_json::json!({"entrypoints":[{"name":"run","models":[],"invocable":{}}]});
    fs::write(directory.join(".hold"), []).unwrap();
    fs::write(directory.join("generation.json"), serde_json::to_vec(&cozy_machine::catalog::Generation {
        identity: generation.clone(), package: "fixture/legacy".into(), version: "1.0.0".into(), application: "fixture:app".into(),
        python, dependencies: vec![], interface: interface.clone(), cpu_bridge: "CPU intent fixture has no model execution".into()
    }).unwrap()).unwrap();
    machine.service.engine.bind_installation(cozy_machine::journal::Installation { actor: machine.actor.clone(),
        alias: "legacy".into(), generation, package: "fixture/legacy".into(), release: "1.0.0".into(),
        interface: serde_json::to_vec(&interface).unwrap() }).unwrap();
    let original = pb::MachineExecutionSubmit {
        claim: Some(machine.claim.clone()), submission_id: "legacy-submission".into(),
        expected_execution_workspace_id: machine.service.engine.workspace_id(),
        offer: Some(pb::AttemptOffer { request_id: "legacy-run".into(), attempt_ordinal: 1, ..Default::default() }),
        release_root: Some(pb::ReleaseRoot { installation_id: "legacy".into(), entrypoint: "run".into(), ..Default::default() }),
        payload_canonical_bytes: b"{}".to_vec(), ..Default::default()
    };
    let accepted = machine.legacy.submit_machine_execution(original.clone()).await.unwrap().into_inner();
    let activation = machine.lifecycle.begin_activation().unwrap();
    assert_eq!(machine.legacy.submit_machine_execution(original.clone()).await.unwrap().into_inner(), accepted);
    let mut changed = original.clone(); changed.payload_canonical_bytes = br#"{"seed":1}"#.to_vec();
    assert_eq!(machine.legacy.submit_machine_execution(changed).await.unwrap_err().metadata().get("cozy-error-code").unwrap(), "execution_intent_conflict");
    let mut fresh = original; fresh.submission_id = "fresh-submission".into();
    fresh.offer.as_mut().unwrap().request_id = "fresh-run".into();
    let refused = machine.legacy.submit_machine_execution(fresh).await.unwrap_err();
    assert_eq!(refused.code(), tonic::Code::Unavailable);
    assert!(refused.message().contains("machine_updating"));
    assert_eq!(machine.service.engine.list().unwrap().len(), 1);
    drop(activation);
}
