use super::{auth::Authority, backend::MachineBackend, pb, WIRE_MINIMUM, WIRE_MINOR};
use crate::machine::receipt::{self, Readiness};
use axum::{
    body::Bytes,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response as HttpResponse},
    routing::get,
};
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
    pub readiness: Arc<Readiness>,
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
            readiness: Readiness::open(None, Some(receipt_key), false)?,
            started_at_ms: now_ms(),
        })
    }
}

pub async fn serve<B: MachineBackend>(
    listener: tokio::net::TcpListener,
    identity: MachineIdentity,
    backend: Arc<B>,
) -> Result<(), Box<dyn std::error::Error>> {
    let port = listener.local_addr()?.port();
    let readiness = identity.readiness.clone();
    tokio::spawn(prove_readiness(
        port,
        readiness.clone(),
        MeasuredIdentity::of(&identity),
    ));
    let tls =
        ServerTlsConfig::new().identity(Identity::from_pem(&identity.cert_pem, &identity.key_pem));
    let service = Api {
        identity: Arc::new(identity),
        backend,
        control_epoch: Arc::new(AtomicU64::new(0)),
    };
    let (post_access, delete_access) = (service.clone(), service.clone());
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
            "/v1/hubs/access",
            axum::routing::post(move |headers: HeaderMap, body: Bytes| {
                let api = post_access.clone();
                async move { api.hub_access(headers, body, false).await }
            })
            .delete(move |headers: HeaderMap, body: Bytes| {
                let api = delete_access.clone();
                async move { api.hub_access(headers, body, true).await }
            }),
        )
        .route(
            "/v1/bootstrap/receipt",
            get(move || {
                let envelope = readiness.envelope();
                async move {
                    let headers = [
                        ("content-type", "application/json"),
                        ("cache-control", "no-store"),
                    ];
                    match envelope {
                        Some(body) => (StatusCode::OK, headers, body),
                        // The Hub probes again on 503 until this boot has proved itself.
                        None => (StatusCode::SERVICE_UNAVAILABLE, headers, vec![]),
                    }
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

struct MeasuredIdentity {
    worker_id: String,
    boot_id: String,
    cert_pem: String,
    cert_der: Vec<u8>,
}
impl MeasuredIdentity {
    fn of(identity: &MachineIdentity) -> Self {
        Self {
            worker_id: identity.authority.worker_id.clone(),
            boot_id: identity.authority.boot_id.clone(),
            cert_pem: identity.cert_pem.clone(),
            cert_der: identity.cert_der.clone(),
        }
    }
}

/// Observes this listener as the Hub requires and seals the result once. A retained envelope
/// of this boot is kept as is; a measurement that contradicts it is reported, never signed.
async fn prove_readiness(port: u16, readiness: Arc<Readiness>, id: MeasuredIdentity) {
    let listener_bound = crate::machine::probe::presents_leaf(port, &id.cert_der).await;
    let foreign_credential_refused = crate::machine::probe::refuses_foreign_claim(
        port,
        &id.cert_pem,
        &id.worker_id,
        &id.boot_id,
    )
    .await;
    let gpus = match readiness.measures_gpus() {
        false => Ok(vec![]),
        true => tokio::task::spawn_blocking(receipt::gpus)
            .await
            .unwrap_or_else(|e| Err(std::io::Error::other(e))),
    };
    let result = gpus.and_then(|gpus| {
        readiness.seal(
            receipt::Measured {
                boot_id: &id.boot_id,
                worker_port: port,
                cert_der: &id.cert_der,
                gpus,
                listener_bound,
                foreign_credential_refused,
                capabilities: crate::machine::CAPABILITIES,
            }
            .payload(),
        )
    });
    match result {
        Ok(true) => eprintln!("cozy-machine: readiness sealed (listener {listener_bound}, foreign Claim refused {foreign_credential_refused})"),
        Ok(false) => eprintln!("cozy-machine: this boot's retained readiness still holds"),
        Err(error) => eprintln!("cozy-machine: readiness not proved: {error}"),
    }
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
    fn auth(&self, claim: Option<&pb::Claim>) -> Result<super::auth::VerifiedActor, Status> {
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

impl<B: MachineBackend> Api<B> {
    /// The Go agent's scoped-access contract: an owner-signed `hub-access` capability,
    /// one JSON body, and typed refusals. Unknown body members are ignored.
    async fn hub_access(&self, headers: HeaderMap, body: Bytes, forget: bool) -> HttpResponse {
        let refuse = |status: u16, code: &str, message: &str| {
            let status = StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_REQUEST);
            let body = serde_json::json!({"error":{"code":code,"message":message}});
            (
                status,
                [("content-type", "application/json")],
                body.to_string(),
            )
                .into_response()
        };
        let token = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Cozy-Cap "))
            .unwrap_or_default();
        let authority = &self.identity.authority;
        let Some(key) = crate::hub::verify_capability(
            token,
            &authority.worker_id,
            &authority.keys,
            now_ms() as i64 / 1000,
            crate::hub::ACTION,
        ) else {
            return refuse(
                403,
                "capability_required",
                "a hub-access capability is required",
            );
        };
        if body.len() > 64 << 10 {
            return refuse(400, "invalid_access", "Hub access body exceeds 64 KiB");
        }
        let actor = super::auth::VerifiedActor {
            public_key: key.to_bytes(),
        };
        let backend = self.backend.clone();
        let answer = tokio::task::spawn_blocking(move || {
            if forget {
                #[derive(serde::Deserialize)]
                #[serde(deny_unknown_fields)]
                struct Forget {
                    origin: String,
                }
                let request: Forget = serde_json::from_slice(&body).map_err(|_| {
                    (
                        400,
                        "invalid_access",
                        "send one valid Hub origin".to_string(),
                    )
                })?;
                backend
                    .forget_hub_access(actor, &request.origin)
                    .map(|()| None)
            } else {
                let access: crate::hub::Access = serde_json::from_slice(&body).map_err(|_| {
                    (
                        400,
                        "invalid_access",
                        "invalid Hub access grant".to_string(),
                    )
                })?;
                backend.hub_access(actor, access).map(Some)
            }
        })
        .await;
        match answer {
            Ok(Ok(None)) => StatusCode::NO_CONTENT.into_response(),
            Ok(Ok(Some((origin, expires_at)))) => (
                [("content-type", "application/json")],
                serde_json::json!({"origin":origin,"expires_at":expires_at}).to_string(),
            )
                .into_response(),
            Ok(Err((status, code, message))) => refuse(status, code, &message),
            Err(_) => refuse(
                503,
                "hub_access_unavailable",
                "Hub access operation stopped",
            ),
        }
    }
}

type ResponseStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;
struct ReaderGuard(super::backend::Observation);
impl Drop for ReaderGuard {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

#[tonic::async_trait]
impl<B: MachineBackend> pb::pod_host_server::PodHost for Api<B> {
    async fn import_input_tree(
        &self,
        request: Request<tonic::Streaming<pb::InputTreeImportFrame>>,
    ) -> Result<Response<pb::NativeByteRetentionResult>, Status> {
        let mut incoming = request.into_inner();
        let first = incoming
            .message()
            .await?
            .ok_or_else(|| Status::unauthenticated("input transfer requires a signed header"))?;
        let header = match first.body {
            Some(pb::input_tree_import_frame::Body::Header(header)) => header,
            _ => {
                return Err(Status::unauthenticated(
                    "input transfer requires a signed header",
                ))
            }
        };
        let actor = self.auth(header.claim.as_ref())?;
        if header.manifest_canonical_bytes.len() > 1 << 20 {
            return Err(Status::invalid_argument(
                "input manifest exceeds its control record bound",
            ));
        }
        let mut receiver = self
            .call(move |backend| backend.begin_input_tree(actor, header))
            .await?
            .into_inner();
        loop {
            match incoming.message().await?.and_then(|frame| frame.body) {
                Some(pb::input_tree_import_frame::Body::Blob(blob)) => {
                    if blob.data.len() > 1 << 20 {
                        return Err(Status::invalid_argument(
                            "input blob exceeds its transport chunk bound",
                        ));
                    }
                    receiver = tokio::task::spawn_blocking(move || {
                        receiver.blob(blob)?;
                        Ok::<_, Status>(receiver)
                    })
                    .await
                    .map_err(|_| Status::internal("input transfer storage task stopped"))??;
                }
                Some(pb::input_tree_import_frame::Body::Commit(commit)) => {
                    if incoming.message().await?.is_some() {
                        return Err(Status::invalid_argument(
                            "input commit must be the final transfer frame",
                        ));
                    }
                    return tokio::task::spawn_blocking(move || receiver.commit(commit))
                        .await
                        .map_err(|_| Status::internal("input commit storage task stopped"))?
                        .map(Response::new);
                }
                Some(pb::input_tree_import_frame::Body::Header(_)) => {
                    return Err(Status::invalid_argument(
                        "input header must appear exactly once",
                    ))
                }
                None => {
                    return Err(Status::invalid_argument(
                        "input transfer ended before an explicit commit",
                    ))
                }
            }
        }
    }
    async fn retain_byte_tree(
        &self,
        request: Request<pb::NativeByteRetentionCall>,
    ) -> Result<Response<pb::NativeByteRetentionResult>, Status> {
        let actor = self.auth(request.get_ref().claim.as_ref())?;
        let request = request.into_inner();
        self.call(move |backend| backend.retain_bytes(actor, request))
            .await
    }
    async fn release_byte_tree(
        &self,
        request: Request<pb::NativeByteRetentionCall>,
    ) -> Result<Response<pb::NativeByteRetentionResult>, Status> {
        let actor = self.auth(request.get_ref().claim.as_ref())?;
        let request = request.into_inner();
        self.call(move |backend| backend.release_bytes(actor, request))
            .await
    }
    async fn list_packages(
        &self,
        request: Request<pb::PackageListQuery>,
    ) -> Result<Response<pb::PackageList>, Status> {
        let actor = self.auth(request.get_ref().claim.as_ref())?;
        let request = request.into_inner();
        self.call(move |backend| backend.list_packages(actor, request))
            .await
    }
    async fn list_models(
        &self,
        request: Request<pb::ModelListQuery>,
    ) -> Result<Response<pb::ModelList>, Status> {
        let actor = self.auth(request.get_ref().claim.as_ref())?;
        let request = request.into_inner();
        self.call(move |backend| backend.list_models(actor, request))
            .await
    }
    async fn local_package_upload(
        &self,
        request: Request<tonic::Streaming<pb::LocalPackageUploadFrame>>,
    ) -> Result<Response<ResponseStream<pb::LocalPackageFileStatus>>, Status> {
        let mut incoming = request.into_inner();
        let first = incoming
            .message()
            .await?
            .ok_or_else(|| Status::unauthenticated("package upload requires a signed header"))?;
        let header = match first.body {
            Some(pb::local_package_upload_frame::Body::Header(header)) => header,
            _ => {
                return Err(Status::unauthenticated(
                    "package upload requires a signed header",
                ))
            }
        };
        let actor = self.auth(header.claim.as_ref())?;
        let uploads = self.backend.uploads().ok_or_else(|| {
            Status::unimplemented("capability_unavailable: package carrier ingress is unavailable")
        })?;
        let authority = self.identity.authority.clone();
        let (sender, receiver) = tokio::sync::mpsc::channel(4);
        tokio::spawn(async move {
            let begin_header = header.clone();
            let begin =
                tokio::task::spawn_blocking(move || uploads.begin(actor, &begin_header)).await;
            let mut session = match begin {
                Ok(Ok(session)) => session,
                Ok(Err(error)) => {
                    let _ = sender.send(Err(error)).await;
                    return;
                }
                Err(_) => {
                    let _ = sender
                        .send(Err(Status::internal("upload storage task stopped")))
                        .await;
                    return;
                }
            };
            let report = |session: &super::workspaces::UploadSession| {
                let file = header.file.as_ref().expect("begin validated one file");
                pb::LocalPackageFileStatus {
                    record_owner_epoch: header
                        .claim
                        .as_ref()
                        .expect("verified claim")
                        .record_owner_epoch,
                    worker_boot_id: authority.boot_id.clone(),
                    operation_id: header.operation_id.clone(),
                    digest: file.digest.clone(),
                    filename: file.filename.clone(),
                    length: file.length,
                    received_bytes: session.received(),
                    state: if session.verified() {
                        pb::LocalPackageFileState::Verified as i32
                    } else {
                        pb::LocalPackageFileState::Receiving as i32
                    },
                    ..Default::default()
                }
            };
            if sender.send(Ok(report(&session))).await.is_err() || session.verified() {
                return;
            }
            loop {
                let chunk = match incoming.message().await {
                    Ok(Some(pb::LocalPackageUploadFrame {
                        body: Some(pb::local_package_upload_frame::Body::Chunk(chunk)),
                    })) => chunk,
                    Ok(None) => return,
                    Ok(Some(_)) => {
                        let _ = sender
                            .send(Err(Status::invalid_argument(
                                "only chunks follow a package upload header",
                            )))
                            .await;
                        return;
                    }
                    Err(error) => {
                        let _ = sender.send(Err(error)).await;
                        return;
                    }
                };
                let appended = tokio::task::spawn_blocking(move || {
                    let result = session.append(chunk);
                    (session, result)
                })
                .await;
                session = match appended {
                    Ok((session, Ok(()))) => session,
                    Ok((_, Err(error))) => {
                        let _ = sender.send(Err(error)).await;
                        return;
                    }
                    Err(_) => {
                        let _ = sender
                            .send(Err(Status::internal("upload storage task stopped")))
                            .await;
                        return;
                    }
                };
                if sender.send(Ok(report(&session))).await.is_err() || session.verified() {
                    return;
                }
            }
        });
        Ok(Response::new(Box::pin(
            tokio_stream::wrappers::ReceiverStream::new(receiver),
        )))
    }

    async fn prepare_local_package(
        &self,
        request: Request<pb::PrepareLocalPackageCall>,
    ) -> Result<Response<ResponseStream<pb::PrepareEvent>>, Status> {
        let actor = self.auth(request.get_ref().claim.as_ref())?;
        let request = request.into_inner();
        let events = self
            .call(move |backend| {
                let selected = request.local_package_set.as_ref().ok_or_else(|| {
                    Status::invalid_argument("local package selection is required")
                })?;
                let uploaded = match backend.uploads() {
                    Some(uploads) => uploads.package(actor, selected)?,
                    None => None,
                };
                backend.prepare_local(actor, request, uploaded)
            })
            .await?
            .into_inner();
        Ok(Response::new(Box::pin(tokio_stream::iter(
            events.into_iter().map(Ok),
        ))))
    }
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
        let actor = self.auth(request.get_ref().claim.as_ref())?;
        let mut description = self.description();
        match self
            .call(move |backend| backend.describe_runtime(actor))
            .await
        {
            Ok(runtime) => {
                description.runtime = Some(runtime.into_inner());
                description.runtime_absent.clear();
            }
            Err(status) if status.code() == tonic::Code::Unimplemented => {
                description.runtime_absent =
                    "this backend does not implement machine runtime observations".into();
            }
            Err(status) => return Err(status),
        }
        Ok(Response::new(description))
    }
    async fn get_machine_execution_workspace(
        &self,
        request: Request<pb::MachineExecutionWorkspaceQuery>,
    ) -> Result<Response<pb::MachineExecutionWorkspace>, Status> {
        let actor = self.auth(request.get_ref().claim.as_ref())?;
        let request = request.into_inner();
        self.call(move |backend| backend.workspace(actor, request))
            .await
    }
    async fn submit_machine_execution(
        &self,
        request: Request<pb::MachineExecutionSubmit>,
    ) -> Result<Response<pb::MachineExecutionReceipt>, Status> {
        let actor = self.auth(request.get_ref().claim.as_ref())?;
        let request = request.into_inner();
        self.call(move |backend| backend.submit(actor, request))
            .await
    }
    async fn get_machine_execution(
        &self,
        request: Request<pb::MachineExecutionQuery>,
    ) -> Result<Response<pb::MachineExecutionState>, Status> {
        let actor = self.auth(request.get_ref().claim.as_ref())?;
        let request = request.into_inner();
        self.call(move |backend| backend.get(actor, request)).await
    }
    async fn list_machine_execution_events(
        &self,
        request: Request<pb::MachineExecutionEventsQuery>,
    ) -> Result<Response<pb::MachineExecutionEventPage>, Status> {
        let actor = self.auth(
            request
                .get_ref()
                .execution
                .as_ref()
                .and_then(|q| q.claim.as_ref()),
        )?;
        let mut request = request.into_inner();
        request.limit = if request.limit == 0 {
            256
        } else {
            request.limit.min(256)
        };
        let observation = super::backend::Observation::default();
        let _reader = ReaderGuard(observation.clone());
        self.call(move |backend| backend.events_observed(actor, request, observation))
            .await
    }
    async fn control_machine_execution(
        &self,
        request: Request<pb::MachineExecutionControl>,
    ) -> Result<Response<pb::MachineExecutionState>, Status> {
        let actor = self.auth(
            request
                .get_ref()
                .execution
                .as_ref()
                .and_then(|q| q.claim.as_ref()),
        )?;
        let request = request.into_inner();
        self.call(move |backend| backend.control(actor, request))
            .await
    }
    async fn list_machine_executions(
        &self,
        request: Request<pb::MachineExecutionListQuery>,
    ) -> Result<Response<pb::MachineExecutionList>, Status> {
        let actor = self.auth(request.get_ref().claim.as_ref())?;
        let mut request = request.into_inner();
        request.limit = if request.limit == 0 {
            64
        } else {
            request.limit.min(256)
        };
        let observation = super::backend::Observation::default();
        let _reader = ReaderGuard(observation.clone());
        self.call(move |backend| backend.list_observed(actor, request, observation))
            .await
    }
    async fn close_machine_submission(
        &self,
        request: Request<pb::MachineSubmissionClose>,
    ) -> Result<Response<pb::MachineSubmissionClosure>, Status> {
        let actor = self.auth(request.get_ref().claim.as_ref())?;
        let request = request.into_inner();
        self.call(move |backend| backend.close_submission(actor, request))
            .await
    }
    async fn collect_machine_execution(
        &self,
        request: Request<pb::MachineExecutionCollect>,
    ) -> Result<Response<pb::AttemptOutcome>, Status> {
        let actor = self.auth(
            request
                .get_ref()
                .execution
                .as_ref()
                .and_then(|q| q.claim.as_ref()),
        )?;
        let request = request.into_inner();
        self.call(move |backend| backend.collect(actor, request))
            .await
    }
    async fn acknowledge_machine_execution_collection(
        &self,
        request: Request<pb::MachineExecutionCollectionAck>,
    ) -> Result<Response<pb::MachineExecutionState>, Status> {
        let actor = self.auth(
            request
                .get_ref()
                .execution
                .as_ref()
                .and_then(|q| q.claim.as_ref()),
        )?;
        let request = request.into_inner();
        self.call(move |backend| backend.ack_collection(actor, request))
            .await
    }
    async fn read_machine_log(
        &self,
        request: Request<pb::MachineLogQuery>,
    ) -> Result<Response<ResponseStream<pb::MachineLogChunk>>, Status> {
        const CHUNK: usize = 64 << 10; // MaxMachineLogChunkBytes
        let actor = self.auth(request.get_ref().claim.as_ref())?;
        let request = request.into_inner();
        let data = self
            .call(move |backend| backend.read_machine_log(actor, request))
            .await?
            .into_inner();
        let chunks: Vec<_> = data
            .chunks(CHUNK)
            .map(|data| {
                Ok(pb::MachineLogChunk {
                    data: data.to_vec(),
                })
            })
            .collect();
        Ok(Response::new(Box::pin(tokio_stream::iter(chunks))))
    }
    async fn read_byte_tree_object(
        &self,
        request: Request<pb::NativeByteReadCall>,
    ) -> Result<Response<ResponseStream<pb::NativeByteReadChunk>>, Status> {
        let actor = self.auth(request.get_ref().claim.as_ref())?;
        let request = request.into_inner();
        let chunks = self
            .call(move |backend| backend.read_stream(actor, request))
            .await?
            .into_inner();
        let (sender, receiver) = tokio::sync::mpsc::channel(2);
        tokio::task::spawn_blocking(move || {
            for chunk in chunks {
                let chunk = chunk.and_then(|chunk| {
                    if chunk.data.is_empty() || chunk.data.len() > 1 << 20 {
                        return Err(Status::data_loss(
                            "backend byte read did not provide one bounded nonempty chunk",
                        ));
                    }
                    Ok(chunk)
                });
                let failed = chunk.is_err();
                if sender.blocking_send(chunk).is_err() || failed {
                    break;
                }
            }
        });
        Ok(Response::new(Box::pin(
            tokio_stream::wrappers::ReceiverStream::new(receiver),
        )))
    }
}

#[tonic::async_trait]
impl<B: MachineBackend> pb::worker_control_server::WorkerControl for Api<B> {
    async fn list_packages(
        &self,
        request: Request<pb::PackageListQuery>,
    ) -> Result<Response<pb::PackageList>, Status> {
        <Self as pb::pod_host_server::PodHost>::list_packages(self, request).await
    }
    async fn list_models(
        &self,
        request: Request<pb::ModelListQuery>,
    ) -> Result<Response<pb::ModelList>, Status> {
        <Self as pb::pod_host_server::PodHost>::list_models(self, request).await
    }
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
        if let Err(status) = self.auth(Some(&claim)) {
            if status.code() != tonic::Code::Unauthenticated {
                return Err(status);
            }
            // The deployed answer to a foreign credential, observed by readiness.
            let refused = pb::WorkerFrame {
                msg: Some(pb::worker_frame::Msg::ClaimAck(pb::ClaimAck {
                    accepted: false,
                    rejection: pb::ClaimRejection::Unauthenticated as i32,
                    record_owner_epoch: claim.record_owner_epoch,
                    worker_id: self.identity.authority.worker_id.clone(),
                    worker_boot_id: self.identity.authority.boot_id.clone(),
                    wire_minor: WIRE_MINOR,
                    ..Default::default()
                })),
            };
            return Ok(Response::new(Box::pin(tokio_stream::iter([Ok(refused)]))));
        }
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
