use super::{auth::Authority, backend::MachineBackend, pb, WIRE_MINIMUM, WIRE_MINOR};
use axum::{
    http::{HeaderMap, StatusCode},
    routing::get,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::{
    pin::Pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{SystemTime, UNIX_EPOCH},
};
use tokio_stream::{wrappers::TcpListenerStream, Stream};
use tonic::{
    transport::{Identity, Server, ServerTlsConfig},
    Request, Response, Status,
};

pub struct MachineIdentity {
    pub authority: Authority,
    pub cert_pem: String,
    pub key_pem: String,
    pub cert_der: Vec<u8>,
    pub receipt_key: Vec<u8>,
    pub started_at_ms: u64,
}

impl MachineIdentity {
    pub fn ephemeral(
        worker_id: String,
        keys: Vec<ed25519_dalek::VerifyingKey>,
        receipt_key: Vec<u8>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if keys.is_empty() || receipt_key.len() < 32 {
            return Err("authorized owner keys and a >=32-byte readiness key are required".into());
        }
        let boot_id = std::fs::read_to_string("/proc/sys/kernel/random/uuid")?
            .trim()
            .to_owned();
        let rcgen::CertifiedKey { cert, signing_key } =
            rcgen::generate_simple_self_signed(vec!["cozy.worker".into(), "localhost".into()])?;
        let cert_der = cert.der().to_vec();
        let authority = Authority {
            worker_id,
            boot_id,
            leaf_digest: tensorfs_core::sha256::digest(&cert_der),
            keys,
        };
        authority.transcript(1)?;
        Ok(Self {
            authority,
            cert_pem: cert.pem(),
            key_pem: signing_key.serialize_pem(),
            cert_der,
            receipt_key,
            started_at_ms: now_ms(),
        })
    }

    fn receipt(&self, port: u16) -> Result<Vec<u8>, serde_json::Error> {
        #[derive(serde::Serialize)]
        struct Receipt<'a> {
            pod_boot_id: &'a str,
            worker_internal_port: u16,
            tls_certificate_der_base64: String,
            runtime_gpus: Vec<serde_json::Value>,
            machine_version: &'a str,
            machine_capabilities: Vec<&'a str>,
        }
        #[derive(serde::Serialize)]
        struct Envelope {
            payload: String,
            hmac_sha256: String,
        }
        let payload = serde_json::to_vec(&Receipt {
            pod_boot_id: &self.authority.boot_id,
            worker_internal_port: port,
            tls_certificate_der_base64: STANDARD.encode(&self.cert_der),
            runtime_gpus: vec![],
            machine_version: env!("CARGO_PKG_VERSION"),
            machine_capabilities: vec![],
        })?;
        let mut mac =
            Hmac::<Sha256>::new_from_slice(&self.receipt_key).expect("HMAC admits any key size");
        mac.update(b"cozy.pod-readiness/1\0");
        mac.update(&payload);
        serde_json::to_vec(&Envelope {
            payload: STANDARD.encode(payload),
            hmac_sha256: tensorfs_core::sha256::hex(&mac.finalize().into_bytes()),
        })
    }
}

pub async fn serve<B: MachineBackend>(
    listener: tokio::net::TcpListener,
    identity: MachineIdentity,
    backend: Arc<B>,
) -> Result<(), Box<dyn std::error::Error>> {
    let port = listener.local_addr()?.port();
    let receipt = Arc::new(identity.receipt(port)?);
    let tls =
        ServerTlsConfig::new().identity(Identity::from_pem(&identity.cert_pem, &identity.key_pem));
    let service = Api {
        identity: Arc::new(identity),
        backend,
        control_epoch: Arc::new(AtomicU64::new(0)),
    };
    let mut routes = tonic::service::Routes::new(
        pb::pod_host_server::PodHostServer::new(service.clone())
            .max_decoding_message_size(16 << 20)
            .max_encoding_message_size(16 << 20),
    )
    .add_service(
        pb::worker_control_server::WorkerControlServer::new(service)
            .max_decoding_message_size(16 << 20)
            .max_encoding_message_size(16 << 20),
    );
    *routes.axum_router_mut() = routes
        .axum_router_mut()
        .clone()
        .route(
            "/v1/health",
            get(|headers: HeaderMap| async move {
                if headers.contains_key("authorization") {
                    StatusCode::UNAUTHORIZED
                } else {
                    StatusCode::NO_CONTENT
                }
            }),
        )
        .route(
            "/v1/bootstrap/receipt",
            get(move || {
                let receipt = receipt.clone();
                async move {
                    (
                        [
                            ("content-type", "application/json"),
                            ("cache-control", "no-store"),
                        ],
                        receipt.as_ref().clone(),
                    )
                }
            }),
        );
    Server::builder()
        .accept_http1(true)
        .tls_config(tls)?
        .add_routes(routes)
        .serve_with_incoming(TcpListenerStream::new(listener))
        .await?;
    Ok(())
}

struct Api<B> {
    identity: Arc<MachineIdentity>,
    backend: Arc<B>,
    control_epoch: Arc<AtomicU64>,
}
impl<B> Clone for Api<B> {
    fn clone(&self) -> Self {
        Self {
            identity: self.identity.clone(),
            backend: self.backend.clone(),
            control_epoch: self.control_epoch.clone(),
        }
    }
}
impl<B: MachineBackend> Api<B> {
    fn auth(&self, claim: Option<&pb::Claim>) -> Result<(), Status> {
        self.identity.authority.verify(claim)
    }
    fn protocol(&self) -> pb::ProtocolInfoResult {
        pb::ProtocolInfoResult {
            wire_minor: WIRE_MINOR,
            minimum_wire_minor: WIRE_MINIMUM,
        }
    }
    fn description(&self) -> pb::MachineDescription {
        pb::MachineDescription { worker_id: self.identity.authority.worker_id.clone(), worker_boot_id: self.identity.authority.boot_id.clone(),
            host: Some(pb::MachineHost { version: env!("CARGO_PKG_VERSION").into(), wire_minor: WIRE_MINOR, minimum_wire_minor: WIRE_MINIMUM,
                platform: format!("{}/{}", std::env::consts::OS, std::env::consts::ARCH), phase: "ready".into(), started_at_unix_ms: self.identity.started_at_ms, ..Default::default() }),
            runtime_absent: "CPU front door: package SDK measurements are supplied by the execution backend; full Runtime inventory is not implemented".into(), ..Default::default() }
    }
    async fn call<T: Send + 'static>(
        &self,
        f: impl FnOnce(Arc<B>) -> Result<T, Status> + Send + 'static,
    ) -> Result<Response<T>, Status> {
        let backend = self.backend.clone();
        tokio::task::spawn_blocking(move || f(backend))
            .await
            .map_err(|_| Status::internal("machine backend operation stopped"))?
            .map(Response::new)
    }
}

type ResponseStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

#[tonic::async_trait]
impl<B: MachineBackend> pb::pod_host_server::PodHost for Api<B> {
    async fn protocol_info(
        &self,
        _: Request<pb::ProtocolInfoRequest>,
    ) -> Result<Response<pb::ProtocolInfoResult>, Status> {
        Ok(Response::new(self.protocol()))
    }
    async fn describe_machine(
        &self,
        request: Request<pb::DescribeMachineQuery>,
    ) -> Result<Response<pb::MachineDescription>, Status> {
        self.auth(request.get_ref().claim.as_ref())?;
        Ok(Response::new(self.description()))
    }
    async fn get_machine_execution_workspace(
        &self,
        request: Request<pb::MachineExecutionWorkspaceQuery>,
    ) -> Result<Response<pb::MachineExecutionWorkspace>, Status> {
        self.auth(request.get_ref().claim.as_ref())?;
        let request = request.into_inner();
        self.call(move |backend| backend.workspace(request)).await
    }
    async fn submit_machine_execution(
        &self,
        request: Request<pb::MachineExecutionSubmit>,
    ) -> Result<Response<pb::MachineExecutionReceipt>, Status> {
        self.auth(request.get_ref().claim.as_ref())?;
        let request = request.into_inner();
        self.call(move |backend| backend.submit(request)).await
    }
    async fn get_machine_execution(
        &self,
        request: Request<pb::MachineExecutionQuery>,
    ) -> Result<Response<pb::MachineExecutionState>, Status> {
        self.auth(request.get_ref().claim.as_ref())?;
        let request = request.into_inner();
        self.call(move |backend| backend.get(request)).await
    }
    async fn list_machine_execution_events(
        &self,
        request: Request<pb::MachineExecutionEventsQuery>,
    ) -> Result<Response<pb::MachineExecutionEventPage>, Status> {
        self.auth(
            request
                .get_ref()
                .execution
                .as_ref()
                .and_then(|q| q.claim.as_ref()),
        )?;
        let request = request.into_inner();
        self.call(move |backend| backend.events(request)).await
    }
    async fn control_machine_execution(
        &self,
        request: Request<pb::MachineExecutionControl>,
    ) -> Result<Response<pb::MachineExecutionState>, Status> {
        self.auth(
            request
                .get_ref()
                .execution
                .as_ref()
                .and_then(|q| q.claim.as_ref()),
        )?;
        let request = request.into_inner();
        self.call(move |backend| backend.control(request)).await
    }
    async fn list_machine_executions(
        &self,
        request: Request<pb::MachineExecutionListQuery>,
    ) -> Result<Response<pb::MachineExecutionList>, Status> {
        self.auth(request.get_ref().claim.as_ref())?;
        let request = request.into_inner();
        self.call(move |backend| backend.list(request)).await
    }
    async fn close_machine_submission(
        &self,
        request: Request<pb::MachineSubmissionClose>,
    ) -> Result<Response<pb::MachineSubmissionClosure>, Status> {
        self.auth(request.get_ref().claim.as_ref())?;
        let request = request.into_inner();
        self.call(move |backend| backend.close_submission(request))
            .await
    }
    async fn collect_machine_execution(
        &self,
        request: Request<pb::MachineExecutionCollect>,
    ) -> Result<Response<pb::AttemptOutcome>, Status> {
        self.auth(
            request
                .get_ref()
                .execution
                .as_ref()
                .and_then(|q| q.claim.as_ref()),
        )?;
        let request = request.into_inner();
        self.call(move |backend| backend.collect(request)).await
    }
    async fn acknowledge_machine_execution_collection(
        &self,
        request: Request<pb::MachineExecutionCollectionAck>,
    ) -> Result<Response<pb::MachineExecutionState>, Status> {
        self.auth(
            request
                .get_ref()
                .execution
                .as_ref()
                .and_then(|q| q.claim.as_ref()),
        )?;
        let request = request.into_inner();
        self.call(move |backend| backend.ack_collection(request))
            .await
    }
    async fn read_byte_tree_object(
        &self,
        request: Request<pb::NativeByteReadCall>,
    ) -> Result<Response<ResponseStream<pb::NativeByteReadChunk>>, Status> {
        self.auth(request.get_ref().claim.as_ref())?;
        let request = request.into_inner();
        let chunks = self
            .call(move |backend| backend.read_bytes(request))
            .await?
            .into_inner();
        Ok(Response::new(Box::pin(tokio_stream::iter(
            chunks.into_iter().map(Ok),
        ))))
    }
}

#[tonic::async_trait]
impl<B: MachineBackend> pb::worker_control_server::WorkerControl for Api<B> {
    async fn control(
        &self,
        request: Request<tonic::Streaming<pb::RecordOwnerFrame>>,
    ) -> Result<Response<ResponseStream<pb::WorkerFrame>>, Status> {
        let mut stream = request.into_inner();
        let first = stream
            .message()
            .await?
            .ok_or_else(|| Status::unauthenticated("Control requires a Claim first"))?;
        let claim = match first.msg {
            Some(pb::record_owner_frame::Msg::Claim(claim)) => claim,
            _ => return Err(Status::unauthenticated("Control requires a Claim first")),
        };
        self.auth(Some(&claim))?;
        let epoch = self.control_epoch.fetch_add(1, Ordering::Relaxed) + 1;
        let ack = pb::WorkerFrame {
            msg: Some(pb::worker_frame::Msg::ClaimAck(pb::ClaimAck {
                accepted: true,
                record_owner_epoch: claim.record_owner_epoch,
                control_stream_epoch: epoch,
                worker_id: self.identity.authority.worker_id.clone(),
                worker_boot_id: self.identity.authority.boot_id.clone(),
                wire_minor: WIRE_MINOR,
                ..Default::default()
            })),
        };
        // Creator uses this stream only to establish its authenticated machine boot.
        // No observer stream is an execution lease. Snapshot/desired-state ownership is
        // not advertised or implemented in this bounded front-door slice.
        let (sender, receiver) = tokio::sync::mpsc::channel(1);
        sender
            .send(Ok(ack))
            .await
            .map_err(|_| Status::cancelled("Control observer disconnected"))?;
        tokio::spawn(async move {
            match stream.message().await {
                Ok(None) => (),
                Ok(Some(_)) => {
                    let _ = sender.send(Err(Status::unimplemented("capability_unavailable: legacy desired-state Control is not implemented"))).await;
                }
                Err(error) => {
                    let _ = sender.send(Err(error)).await;
                }
            }
        });
        Ok(Response::new(Box::pin(
            tokio_stream::wrappers::ReceiverStream::new(receiver),
        )))
    }
    async fn describe_machine(
        &self,
        request: Request<pb::DescribeMachineQuery>,
    ) -> Result<Response<pb::MachineDescription>, Status> {
        <Self as pb::pod_host_server::PodHost>::describe_machine(self, request).await
    }
    async fn get_machine_execution_workspace(
        &self,
        request: Request<pb::MachineExecutionWorkspaceQuery>,
    ) -> Result<Response<pb::MachineExecutionWorkspace>, Status> {
        <Self as pb::pod_host_server::PodHost>::get_machine_execution_workspace(self, request).await
    }
    async fn submit_machine_execution(
        &self,
        request: Request<pb::MachineExecutionSubmit>,
    ) -> Result<Response<pb::MachineExecutionReceipt>, Status> {
        <Self as pb::pod_host_server::PodHost>::submit_machine_execution(self, request).await
    }
    async fn get_machine_execution(
        &self,
        request: Request<pb::MachineExecutionQuery>,
    ) -> Result<Response<pb::MachineExecutionState>, Status> {
        <Self as pb::pod_host_server::PodHost>::get_machine_execution(self, request).await
    }
    async fn list_machine_execution_events(
        &self,
        request: Request<pb::MachineExecutionEventsQuery>,
    ) -> Result<Response<pb::MachineExecutionEventPage>, Status> {
        <Self as pb::pod_host_server::PodHost>::list_machine_execution_events(self, request).await
    }
    async fn control_machine_execution(
        &self,
        request: Request<pb::MachineExecutionControl>,
    ) -> Result<Response<pb::MachineExecutionState>, Status> {
        <Self as pb::pod_host_server::PodHost>::control_machine_execution(self, request).await
    }
    async fn list_machine_executions(
        &self,
        request: Request<pb::MachineExecutionListQuery>,
    ) -> Result<Response<pb::MachineExecutionList>, Status> {
        <Self as pb::pod_host_server::PodHost>::list_machine_executions(self, request).await
    }
    async fn close_machine_submission(
        &self,
        request: Request<pb::MachineSubmissionClose>,
    ) -> Result<Response<pb::MachineSubmissionClosure>, Status> {
        <Self as pb::pod_host_server::PodHost>::close_machine_submission(self, request).await
    }
    async fn collect_machine_execution(
        &self,
        request: Request<pb::MachineExecutionCollect>,
    ) -> Result<Response<pb::AttemptOutcome>, Status> {
        <Self as pb::pod_host_server::PodHost>::collect_machine_execution(self, request).await
    }
    async fn acknowledge_machine_execution_collection(
        &self,
        request: Request<pb::MachineExecutionCollectionAck>,
    ) -> Result<Response<pb::MachineExecutionState>, Status> {
        <Self as pb::pod_host_server::PodHost>::acknowledge_machine_execution_collection(
            self, request,
        )
        .await
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
