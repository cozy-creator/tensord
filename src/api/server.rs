use super::{auth::Authority, backend::MachineBackend, pb, WIRE_MINIMUM, WIRE_MINOR};
use crate::machine::receipt::{self, Readiness};
use axum::{
    body::Bytes,
    extract::Path,
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
    /// A rental's idle ledger; None on development front doors.
    pub lifecycle: Option<Arc<crate::machine::lifecycle::Lifecycle>>,
    /// The Hubs this machine is registered with: (origin, worker id there).
    pub hubs: Vec<(String, String)>,
    /// `runtime-update/1`; None on development front doors.
    pub updates: Option<Arc<crate::machine::update::Updates>>,
    /// A granted media port: the CLI's machine launcher reads the receipt there.
    pub media: Option<std::net::TcpListener>,
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
            keys: keys.into(),
        };
        authority.transcript(1)?;
        Ok(Self {
            authority,
            cert_pem: cert.pem(),
            key_pem: signing_key.serialize_pem(),
            cert_der,
            readiness: Readiness::open(None, Some(receipt_key), false)?,
            started_at_ms: now_ms(),
            lifecycle: None,
            hubs: vec![],
            updates: None,
            media: None,
        })
    }
}

pub async fn serve<B: MachineBackend>(
    listener: tokio::net::TcpListener,
    mut identity: MachineIdentity,
    backend: Arc<B>,
) -> Result<(), Box<dyn std::error::Error>> {
    let port = listener.local_addr()?.port();
    let readiness = identity.readiness.clone();
    if let Some(media) = identity.media.take() {
        media.set_nonblocking(true)?;
        let media = tokio::net::TcpListener::from_std(media)?;
        let tls = ServerTlsConfig::new()
            .identity(Identity::from_pem(&identity.cert_pem, &identity.key_pem));
        let readiness = readiness.clone();
        let mut routes = tonic::service::Routes::default();
        *routes.axum_router_mut() = axum::Router::new().route(
            "/v1/bootstrap/receipt",
            get(move || {
                let envelope = readiness.envelope();
                async move {
                    match envelope {
                        Some(body) => (StatusCode::OK, body),
                        None => (StatusCode::SERVICE_UNAVAILABLE, vec![]),
                    }
                }
            }),
        );
        let server = Server::builder()
            .accept_http1(true)
            .tls_config(tls)?
            .add_routes(routes)
            .serve_with_incoming(TcpListenerStream::new(media));
        tokio::spawn(async move {
            if let Err(error) = server.await {
                eprintln!("cozy-machine: the receipt listener stopped: {error}");
            }
        });
    }
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
    let (output, listed) = (service.clone(), service.clone());
    let (state_api, stage_api, update_api) = (service.clone(), service.clone(), service.clone());
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
            "/v1/runs/{run}/outputs/{output}",
            get(
                move |headers: HeaderMap, Path((run, name)): Path<(String, String)>| {
                    let api = output.clone();
                    async move { api.output(headers, run, name, None).await }
                },
            ),
        )
        .route(
            "/v1/runs/{run}/outputs/{output}/{index}",
            get(
                move |headers: HeaderMap,
                      Path((run, name, index)): Path<(String, String, String)>| {
                    let api = listed.clone();
                    async move { api.output(headers, run, name, Some(index)).await }
                },
            ),
        )
        .route(
            "/v1/machine/runtime",
            get(move |headers: HeaderMap| {
                let api = state_api.clone();
                async move { api.maintenance(headers, Maintenance::State).await }
            }),
        )
        .route(
            "/v1/machine/runtime/wheels/{file}",
            axum::routing::put(
                move |axum::extract::Path(file): axum::extract::Path<String>,
                      headers: HeaderMap,
                      body: axum::body::Body| {
                    let api = stage_api.clone();
                    async move {
                        api.maintenance(headers, Maintenance::Stage(file, body))
                            .await
                    }
                },
            ),
        )
        .route(
            "/v1/machine/runtime/update",
            axum::routing::post(move |headers: HeaderMap, body: Bytes| {
                let api = update_api.clone();
                async move { api.maintenance(headers, Maintenance::Update(body)).await }
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

enum Maintenance {
    State,
    Stage(String, axum::body::Body),
    Update(Bytes),
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
                platform: format!("{}/{}", std::env::consts::OS, std::env::consts::ARCH), phase: self.phase().into(), started_at_unix_ms: self.identity.started_at_ms,
                idle_deadline_unix_ms: self.identity.lifecycle.as_ref().map_or(0, |l| l.deadline_ms().max(0) as u64),
                hubs: self.identity.hubs.iter().map(|(origin, machine_id)| pb::MachineHub { origin: origin.clone(), machine_id: machine_id.clone() }).collect(),
                ..Default::default() }),
            runtime_absent: "CPU front door: package SDK measurements are supplied by the execution backend; full Runtime inventory is not implemented".into(), ..Default::default() }
    }
    /// The accelerator facts this boot measured for its readiness receipt.
    fn resources(&self) -> pb::WorkerResources {
        let gpus = self.identity.readiness.gpus();
        let first = gpus.first();
        pb::WorkerResources {
            platform: format!("{}/{}", std::env::consts::OS, std::env::consts::ARCH),
            backend: if gpus.is_empty() { "" } else { "cuda" }.into(),
            memory_model: if gpus.is_empty() { "" } else { "discrete" }.into(),
            device_count: gpus.len() as u32,
            device_name: first.map(|g| g.device_name.clone()).unwrap_or_default(),
            device_memory_total_bytes: first.map_or(0, |g| g.memory_bytes),
            driver_version: first.map(|g| g.driver_version.clone()).unwrap_or_default(),
            ..Default::default()
        }
    }
    fn phase(&self) -> &'static str {
        match &self.identity.lifecycle {
            Some(lifecycle) if lifecycle.released() => "releasing",
            _ => "ready",
        }
    }
    /// Holds a rental's idle release while a call that may start work runs.
    fn admit(&self) -> Result<Option<crate::machine::lifecycle::Admission>, Status> {
        self.identity
            .lifecycle
            .as_ref()
            .map(|l| l.admit())
            .transpose()
    }
    fn worked(&self) {
        if let Some(lifecycle) = &self.identity.lifecycle {
            if let Err(error) = lifecycle.work() {
                eprintln!("cozy-machine: idle ledger: {error}");
            }
        }
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
    /// One run output's current bytes for a `Cozy-Cap` holder: ETag `"r<rev>"`, a single
    /// byte range, and `Repr-Digest` once the output is final.
    async fn output(
        &self,
        headers: HeaderMap,
        run: String,
        name: String,
        index: Option<String>,
    ) -> HttpResponse {
        let text = |status: StatusCode, message: String| {
            (
                status,
                [("content-type", "text/plain; charset=utf-8")],
                message,
            )
                .into_response()
        };
        let number = run.parse::<u64>().ok().filter(|n| *n > 0);
        // A list item's index is 1-based; an unparsable one names no output.
        let index = match index {
            None => Some(None),
            Some(text) => text.parse::<u32>().ok().filter(|i| *i > 0).map(Some),
        };
        let (Some(number), Some(index)) = (number, index) else {
            return text(StatusCode::NOT_FOUND, "the path names no run output".into());
        };
        let token = headers
            .get("authorization")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Cozy-Cap "))
            .unwrap_or_default();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| since.as_secs() as i64);
        let authority = &self.identity.authority;
        let keys = authority.keys.admitted();
        let granted = super::capability::verify(token, &authority.worker_id, &keys, now, "")
            .and_then(|grant| {
                if grant.allows(&run, &name, index) {
                    Ok(grant)
                } else {
                    Err(super::capability::Refusal::Scope)
                }
            });
        if let Err(refusal) = granted {
            return text(StatusCode::FORBIDDEN, refusal.to_string());
        }
        let backend = self.backend.clone();
        let snapshot =
            match tokio::task::spawn_blocking(move || backend.open_output(number, &name, index))
                .await
            {
                Ok(Ok(snapshot)) => snapshot,
                Ok(Err(status)) => {
                    let code = match status.code() {
                        tonic::Code::NotFound => StatusCode::NOT_FOUND,
                        tonic::Code::Unimplemented => StatusCode::NOT_IMPLEMENTED,
                        tonic::Code::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
                        _ => StatusCode::BAD_GATEWAY,
                    };
                    return text(code, status.message().into());
                }
                Err(_) => {
                    return text(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "output read stopped".into(),
                    )
                }
            };
        let etag = format!("\"r{}\"", snapshot.rev);
        let mut response = axum::http::Response::builder()
            .header("etag", &etag)
            .header("cache-control", "private, no-cache")
            .header("accept-ranges", "bytes")
            .header(
                "content-type",
                if snapshot.media_type.is_empty() {
                    "application/octet-stream"
                } else {
                    &snapshot.media_type
                },
            );
        if let Some(raw) = snapshot
            .sha256
            .as_deref()
            .and_then(|digest| digest.strip_prefix("sha256:"))
            .and_then(|hex| {
                (0..hex.len())
                    .step_by(2)
                    .map(|i| u8::from_str_radix(hex.get(i..i + 2)?, 16).ok())
                    .collect::<Option<Vec<u8>>>()
            })
        {
            use base64::{engine::general_purpose::STANDARD, Engine as _};
            response =
                response.header("repr-digest", format!("sha-256=:{}:", STANDARD.encode(raw)));
        }
        if headers
            .get("if-none-match")
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.split(',').any(|tag| tag.trim() == etag))
        {
            return response
                .status(StatusCode::NOT_MODIFIED)
                .body(axum::body::Body::empty())
                .unwrap_or_default();
        }
        let length = snapshot.length;
        let (status, start, count) = match headers
            .get("range")
            .and_then(|value| value.to_str().ok())
            .map(|value| byte_range(value, length))
        {
            None => (StatusCode::OK, 0, length),
            Some(Some((start, end))) => {
                response =
                    response.header("content-range", format!("bytes {start}-{end}/{length}"));
                (StatusCode::PARTIAL_CONTENT, start, end + 1 - start)
            }
            Some(None) => {
                return response
                    .status(StatusCode::RANGE_NOT_SATISFIABLE)
                    .header("content-range", format!("bytes */{length}"))
                    .body(axum::body::Body::empty())
                    .unwrap_or_default();
            }
        };
        let (sender, receiver) = tokio::sync::mpsc::channel::<std::io::Result<Bytes>>(4);
        tokio::task::spawn_blocking(move || send_range(snapshot.parts, start, count, sender));
        response
            .status(status)
            .header("content-length", count)
            .body(axum::body::Body::from_stream(
                tokio_stream::wrappers::ReceiverStream::new(receiver),
            ))
            .unwrap_or_default()
    }
    /// The owner's maintenance routes (`runtime-update` capability), as `cozy rental update`
    /// drives them.
    async fn maintenance(&self, headers: HeaderMap, call: Maintenance) -> HttpResponse {
        let text = |status: u16, message: String| {
            (
                StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_REQUEST),
                message,
            )
                .into_response()
        };
        let token = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Cozy-Cap "))
            .unwrap_or_default();
        let authority = &self.identity.authority;
        let keys = authority.keys.admitted();
        let now = now_ms() as i64 / 1000;
        if crate::hub::verify_capability(token, &authority.worker_id, &keys, now, "runtime-update")
            .is_none()
        {
            return text(403, "a runtime-update capability is required".into());
        }
        let Some(updates) = self.identity.updates.clone() else {
            return text(404, "this machine does not update its own Runtime".into());
        };
        let starting = || {
            let body = serde_json::json!({"code":"runtime_starting","message":"this machine has not proved readiness; observe GET /v1/machine/runtime"});
            (
                StatusCode::SERVICE_UNAVAILABLE,
                [("content-type", "application/json")],
                body.to_string(),
            )
                .into_response()
        };
        let json = |status: StatusCode, value: serde_json::Value| {
            (
                status,
                [("content-type", "application/json")],
                value.to_string(),
            )
                .into_response()
        };
        match call {
            Maintenance::State => json(StatusCode::OK, updates.state(crate::machine::CAPABILITIES)),
            Maintenance::Stage(_, _) | Maintenance::Update(_)
                if !self.identity.readiness.proved() =>
            {
                starting()
            }
            Maintenance::Stage(file, body) => {
                let body = match axum::body::to_bytes(body, 256 << 20).await {
                    Ok(body) => body,
                    Err(error) => return text(400, format!("the wheel upload broke: {error}")),
                };
                let staged = tokio::task::spawn_blocking(move || {
                    let length = body.len() as u64;
                    updates
                        .stage(&file, &mut body.as_ref())
                        .map(|(sha, _)| (file, sha, length))
                })
                .await;
                match staged {
                    Ok(Ok((file, sha256, length))) => json(
                        StatusCode::OK,
                        serde_json::json!({"file": file, "sha256": sha256, "length": length}),
                    ),
                    Ok(Err((status, message))) => text(status, message),
                    Err(_) => text(500, "staging stopped".into()),
                }
            }
            Maintenance::Update(body) => {
                let request: crate::machine::update::Request = match serde_json::from_slice(&body) {
                    Ok(request) => request,
                    Err(error) => return text(400, format!("an update requires operation, valid agent selection, and a Runtime or TensorFS: {error}")),
                };
                match updates.request(request, |code| std::process::exit(code)) {
                    Ok(status) => json(
                        StatusCode::ACCEPTED,
                        serde_json::to_value(status).unwrap_or_default(),
                    ),
                    Err((status, message)) => text(status, message),
                }
            }
        }
    }

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
            &authority.keys.admitted(),
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
        let held = self.admit()?;
        let uploads = self.backend.uploads().ok_or_else(|| {
            Status::unimplemented("capability_unavailable: package carrier ingress is unavailable")
        })?;
        let authority = self.identity.authority.clone();
        let (sender, receiver) = tokio::sync::mpsc::channel(4);
        tokio::spawn(async move {
            let _held = held; // an upload in flight holds idle release
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
        let _held = self.admit()?;
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
        self.worked();
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
        let _held = self.admit()?;
        let request = request.into_inner();
        let receipt = self
            .call(move |backend| backend.submit(actor, request))
            .await?;
        self.worked();
        Ok(receipt)
    }
    async fn keep_rental_alive(
        &self,
        request: Request<pb::KeepRentalAliveRequest>,
    ) -> Result<Response<pb::KeepRentalAliveResult>, Status> {
        self.auth(request.get_ref().claim.as_ref())?;
        let request = request.into_inner();
        let lifecycle = self
            .identity
            .lifecycle
            .as_ref()
            .ok_or_else(|| Status::unimplemented("this machine has no rental lifecycle"))?;
        if request.request_id.is_empty() || request.request_id.len() > 128 {
            return Err(Status::invalid_argument(
                "a keepalive names one bounded request_id",
            ));
        }
        let (acknowledged, deadline) = lifecycle.keepalive(&request.request_id)?;
        Ok(Response::new(pb::KeepRentalAliveResult {
            request_id: request.request_id,
            worker_id: self.identity.authority.worker_id.clone(),
            worker_boot_id: self.identity.authority.boot_id.clone(),
            acknowledged_at_unix_ms: acknowledged,
            idle_deadline_unix_ms: deadline,
        }))
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
    async fn forget_package(
        &self,
        request: Request<pb::ForgetPackageCall>,
    ) -> Result<Response<pb::ForgetPackageResult>, Status> {
        let actor = self.auth(request.get_ref().claim.as_ref())?;
        let request = request.into_inner();
        self.call(move |backend| backend.forget_package(actor, request))
            .await
    }
    async fn read_machine_execution_triage(
        &self,
        request: Request<pb::MachineExecutionTriageQuery>,
    ) -> Result<Response<pb::MachineExecutionTriage>, Status> {
        let actor = self.auth(
            request
                .get_ref()
                .execution
                .as_ref()
                .and_then(|q| q.claim.as_ref()),
        )?;
        let request = request.into_inner();
        self.call(move |backend| backend.read_triage(actor, request))
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
                // One provisioned machine lifetime, as the Runtime names it: the boot id.
                worker_instance_id: self.identity.authority.boot_id.clone(),
                resources: Some(self.resources()),
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
    async fn read_machine_execution_triage(
        &self,
        request: Request<pb::MachineExecutionTriageQuery>,
    ) -> Result<Response<pb::MachineExecutionTriage>, Status> {
        <Self as pb::pod_host_server::PodHost>::read_machine_execution_triage(self, request).await
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// One `bytes=` range of a `length`-byte body as inclusive (start, end); None when it cannot
/// be satisfied. Several ranges are answered as the first.
fn byte_range(header: &str, length: u64) -> Option<(u64, u64)> {
    let spec = header.strip_prefix("bytes=")?.split(',').next()?.trim();
    let (first, last) = spec.split_once('-')?;
    let (start, end) = if first.is_empty() {
        let suffix: u64 = last.parse().ok()?;
        (
            length.checked_sub(suffix.min(length))?,
            length.checked_sub(1)?,
        )
    } else {
        let start: u64 = first.parse().ok()?;
        let end = if last.is_empty() {
            length.checked_sub(1)?
        } else {
            last.parse::<u64>().ok()?.min(length.checked_sub(1)?)
        };
        (start, end)
    };
    (start <= end && end < length).then_some((start, end))
}

/// Reads `count` bytes from `start` across the snapshot's parts into the response stream.
fn send_range(
    parts: Vec<(std::fs::File, u64)>,
    mut start: u64,
    mut count: u64,
    sender: tokio::sync::mpsc::Sender<std::io::Result<Bytes>>,
) {
    use std::os::unix::fs::FileExt;
    let mut buffer = vec![0; 64 << 10];
    for (file, length) in parts {
        if start >= length {
            start -= length;
            continue;
        }
        while count > 0 && start < length {
            let want = (buffer.len() as u64).min(count).min(length - start) as usize;
            let read = match file.read_at(&mut buffer[..want], start) {
                Ok(0) => Err(std::io::Error::other("output part ended early")),
                Ok(read) => Ok(Bytes::copy_from_slice(&buffer[..read])),
                Err(error) => Err(error),
            };
            let failed = read.is_err();
            let read_len = read.as_ref().map_or(0, |bytes| bytes.len() as u64);
            if sender.blocking_send(read).is_err() || failed {
                return;
            }
            start += read_len;
            count -= read_len;
        }
        if count == 0 {
            return;
        }
        start = 0;
    }
}
