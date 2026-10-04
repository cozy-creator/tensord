//! Real TLS/API and journal checks for semantic reattachment and exact application integers.
//! No model inference is simulated or qualified by these tests.
use cozy_machine::{
    api::{
        self,
        capability::{self, Grant},
        v1, MachineIdentity,
    },
    journal::{Invocation, Outcome, ResultRecord},
    machine_api::NativeBackend,
    objects::Objects,
    runs::Runs,
    service::Service,
};
use ed25519_dalek::SigningKey;
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
    task: tokio::task::JoinHandle<()>,
    actor: String,
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
        let identity =
            MachineIdentity::ephemeral("intent".into(), vec![signer.verifying_key()], vec![7; 32])
                .unwrap();
        let pem = identity.cert_pem.clone();
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
            client: v1::machine_client::MachineClient::new(channel),
            token,
            task,
            actor,
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
