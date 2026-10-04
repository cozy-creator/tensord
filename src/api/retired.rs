//! What a client that still speaks `cozy.worker.v1` hears from a machine that no longer serves
//! it. A path nothing routes answers UNIMPLEMENTED, which released CLI 0.1.26 reports as a
//! machine too old for it, the wrong side. These routes answer FAILED_PRECONDITION, whose
//! message that CLI prints as it is. The build that drops the worker.v1 services mounts them in
//! their place.
use axum::{body::Body, http::Response, routing::any, Router};

pub const WORKER_V1: &str =
    "this machine serves cozy.machine.v1 only: upgrade the cozy CLI to 0.2.0 or newer";

/// `router` with every call of the two worker.v1 services answered [`WORKER_V1`].
pub fn worker_v1(router: Router) -> Router {
    ["PodHost", "WorkerControl"]
        .into_iter()
        .fold(router, |router, service| {
            router.route(&format!("/cozy.worker.v1.{service}/{{method}}"), any(gone))
        })
}

async fn gone() -> Response<Body> {
    tonic::Status::failed_precondition(WORKER_V1).into_http()
}
