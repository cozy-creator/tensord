//! Real TLS/socket callback dispatch. This recording backend claims no native custody or inference.
use cozy_machine::api::{
    self,
    auth::VerifiedActor,
    backend::{InputTreeReceiver, Observation},
    pb, MachineBackend, MachineIdentity,
};
use ed25519_dalek::{Signer, SigningKey};
use std::sync::{Arc, Mutex};
use tonic::{
    transport::{Certificate, ClientTlsConfig, Endpoint},
    Status,
};

#[derive(Default)]
struct Probe {
    calls: Arc<Mutex<Vec<String>>>,
}
impl MachineBackend for Probe {
    fn workspace(
        &self,
        _: VerifiedActor,
        _: pb::MachineExecutionWorkspaceQuery,
    ) -> Result<pb::MachineExecutionWorkspace, Status> {
        Err(Status::unimplemented("probe has no execution workspace"))
    }
    fn describe_runtime(&self, actor: VerifiedActor) -> Result<pb::MachineRuntime, Status> {
        self.calls
            .lock()
            .unwrap()
            .push(format!("runtime:{}", actor.public_key[0]));
        Ok(pb::MachineRuntime {
            version: env!("CARGO_PKG_VERSION").into(),
            wire_minor: api::WIRE_MINOR,
            minimum_wire_minor: 0,
            tensorfs_version: tensorfs_core::VERSION.into(),
            accelerator_backend: "none".into(),
            ..Default::default()
        })
    }
    fn list_packages(
        &self,
        _: VerifiedActor,
        _: pb::PackageListQuery,
    ) -> Result<pb::PackageList, Status> {
        self.calls.lock().unwrap().push("packages".into());
        Ok(pb::PackageList::default())
    }
    fn list_models(
        &self,
        _: VerifiedActor,
        _: pb::ModelListQuery,
    ) -> Result<pb::ModelList, Status> {
        self.calls.lock().unwrap().push("models".into());
        Ok(pb::ModelList::default())
    }
    fn retain_bytes(
        &self,
        _: VerifiedActor,
        _: pb::NativeByteRetentionCall,
    ) -> Result<pb::NativeByteRetentionResult, Status> {
        self.calls.lock().unwrap().push("retain".into());
        Err(Status::unimplemented(
            "recording backend does not claim custody",
        ))
    }
    fn release_bytes(
        &self,
        _: VerifiedActor,
        _: pb::NativeByteRetentionCall,
    ) -> Result<pb::NativeByteRetentionResult, Status> {
        self.calls.lock().unwrap().push("release".into());
        Err(Status::unimplemented(
            "recording backend does not claim custody",
        ))
    }
    fn begin_input_tree(
        &self,
        _: VerifiedActor,
        _: pb::InputTreeImportHeader,
    ) -> Result<Box<dyn InputTreeReceiver>, Status> {
        self.calls.lock().unwrap().push("input".into());
        Ok(Box::new(InputProbe(self.calls.clone())))
    }
    fn events_observed(
        &self,
        _: VerifiedActor,
        request: pb::MachineExecutionEventsQuery,
        _: Observation,
    ) -> Result<pb::MachineExecutionEventPage, Status> {
        self.calls
            .lock()
            .unwrap()
            .push(format!("events-limit:{}", request.limit));
        Ok(pb::MachineExecutionEventPage::default())
    }
    fn list_observed(
        &self,
        _: VerifiedActor,
        request: pb::MachineExecutionListQuery,
        _: Observation,
    ) -> Result<pb::MachineExecutionList, Status> {
        self.calls
            .lock()
            .unwrap()
            .push(format!("list-limit:{}", request.limit));
        Ok(pb::MachineExecutionList::default())
    }
}
struct InputProbe(Arc<Mutex<Vec<String>>>);
impl InputTreeReceiver for InputProbe {
    fn blob(&mut self, blob: pb::InputTreeImportBlob) -> Result<(), Status> {
        self.0
            .lock()
            .unwrap()
            .push(format!("blob:{}", blob.data.len()));
        Ok(())
    }
    fn commit(
        self: Box<Self>,
        commit: pb::InputTreeImportCommit,
    ) -> Result<pb::NativeByteRetentionResult, Status> {
        self.0
            .lock()
            .unwrap()
            .push(format!("commit:{}", commit.abort));
        Err(Status::unimplemented(
            "recording backend does not claim native commit",
        ))
    }
}

async fn endpoint() -> (
    pb::pod_host_client::PodHostClient<tonic::transport::Channel>,
    pb::Claim,
    Arc<Mutex<Vec<String>>>,
    tokio::task::JoinHandle<()>,
) {
    let signer = SigningKey::from_bytes(&[21; 32]);
    let identity = MachineIdentity::ephemeral(
        "owned-transport-probe".into(),
        vec![signer.verifying_key()],
        vec![7; 32],
    )
    .unwrap();
    let claim = pb::Claim {
        worker_id: identity.authority.worker_id.clone(),
        worker_boot_id: identity.authority.boot_id.clone(),
        record_owner_epoch: 1,
        proof: signer
            .sign(&identity.authority.transcript(1).unwrap())
            .to_bytes()
            .to_vec(),
        ..Default::default()
    };
    let pem = identity.cert_pem.clone();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let backend = Arc::new(Probe::default());
    let calls = backend.calls.clone();
    let server = tokio::spawn(async move {
        api::serve(listener, identity, backend).await.unwrap();
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
    (
        pb::pod_host_client::PodHostClient::new(channel),
        claim,
        calls,
        server,
    )
}

#[tokio::test]
async fn authenticated_runtime_inventory_and_page_bounds_enter_typed_backend() {
    let (mut client, claim, calls, server) = endpoint().await;
    let described = client
        .describe_machine(pb::DescribeMachineQuery {
            claim: Some(claim.clone()),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(described.runtime_absent.is_empty());
    assert_eq!(
        described.runtime.unwrap().version,
        env!("CARGO_PKG_VERSION")
    );
    client
        .list_packages(pb::PackageListQuery {
            claim: Some(claim.clone()),
        })
        .await
        .unwrap();
    client
        .list_models(pb::ModelListQuery {
            claim: Some(claim.clone()),
        })
        .await
        .unwrap();
    client
        .list_machine_executions(pb::MachineExecutionListQuery {
            claim: Some(claim.clone()),
            limit: u32::MAX,
            ..Default::default()
        })
        .await
        .unwrap();
    client
        .list_machine_execution_events(pb::MachineExecutionEventsQuery {
            execution: Some(pb::MachineExecutionQuery {
                claim: Some(claim),
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
        .unwrap();
    let calls = calls.lock().unwrap();
    assert!(calls.iter().any(|s| s.starts_with("runtime:")));
    assert!(calls.contains(&"packages".into()));
    assert!(calls.contains(&"models".into()));
    assert!(calls.contains(&"list-limit:256".into()));
    assert!(calls.contains(&"events-limit:256".into()));
    server.abort();
}

#[tokio::test]
async fn unauthenticated_custody_and_input_never_reach_business_callback() {
    let (mut client, claim, calls, server) = endpoint().await;
    assert_eq!(
        client
            .retain_byte_tree(pb::NativeByteRetentionCall::default())
            .await
            .unwrap_err()
            .code(),
        tonic::Code::Unauthenticated
    );
    assert_eq!(
        client
            .release_byte_tree(pb::NativeByteRetentionCall::default())
            .await
            .unwrap_err()
            .code(),
        tonic::Code::Unauthenticated
    );
    let frames = vec![pb::InputTreeImportFrame {
        body: Some(pb::input_tree_import_frame::Body::Header(
            pb::InputTreeImportHeader::default(),
        )),
    }];
    assert_eq!(
        client
            .import_input_tree(tokio_stream::iter(frames))
            .await
            .unwrap_err()
            .code(),
        tonic::Code::Unauthenticated
    );
    assert!(calls.lock().unwrap().is_empty());
    assert_eq!(
        client
            .retain_byte_tree(pb::NativeByteRetentionCall {
                claim: Some(claim.clone()),
                ..Default::default()
            })
            .await
            .unwrap_err()
            .code(),
        tonic::Code::Unimplemented
    );
    assert_eq!(
        client
            .release_byte_tree(pb::NativeByteRetentionCall {
                claim: Some(claim),
                ..Default::default()
            })
            .await
            .unwrap_err()
            .code(),
        tonic::Code::Unimplemented
    );
    assert_eq!(
        *calls.lock().unwrap(),
        vec!["retain".to_string(), "release".to_string()]
    );
    server.abort();
}

#[tokio::test]
async fn bounded_input_frames_and_explicit_final_commit_are_forwarded_without_false_custody() {
    let (mut client, claim, calls, server) = endpoint().await;
    let header = pb::InputTreeImportFrame {
        body: Some(pb::input_tree_import_frame::Body::Header(
            pb::InputTreeImportHeader {
                claim: Some(claim),
                ..Default::default()
            },
        )),
    };
    let blob = pb::InputTreeImportFrame {
        body: Some(pb::input_tree_import_frame::Body::Blob(
            pb::InputTreeImportBlob {
                data: vec![1; 1 << 20],
                ..Default::default()
            },
        )),
    };
    let commit = pb::InputTreeImportFrame {
        body: Some(pb::input_tree_import_frame::Body::Commit(
            pb::InputTreeImportCommit { abort: false },
        )),
    };
    assert_eq!(
        client
            .import_input_tree(tokio_stream::iter(vec![header.clone(), blob, commit]))
            .await
            .unwrap_err()
            .code(),
        tonic::Code::Unimplemented
    );
    assert_eq!(
        *calls.lock().unwrap(),
        vec![
            "input".to_string(),
            "blob:1048576".to_string(),
            "commit:false".to_string()
        ]
    );
    calls.lock().unwrap().clear();
    let oversized = pb::InputTreeImportFrame {
        body: Some(pb::input_tree_import_frame::Body::Blob(
            pb::InputTreeImportBlob {
                data: vec![1; (1 << 20) + 1],
                ..Default::default()
            },
        )),
    };
    assert_eq!(
        client
            .import_input_tree(tokio_stream::iter(vec![header.clone(), oversized]))
            .await
            .unwrap_err()
            .code(),
        tonic::Code::InvalidArgument
    );
    assert_eq!(*calls.lock().unwrap(), vec!["input".to_string()]);
    calls.lock().unwrap().clear();
    assert_eq!(
        client
            .import_input_tree(tokio_stream::iter(vec![header]))
            .await
            .unwrap_err()
            .code(),
        tonic::Code::InvalidArgument
    );
    assert_eq!(*calls.lock().unwrap(), vec!["input".to_string()]);
    server.abort();
}
