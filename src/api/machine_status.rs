//! Status (G/API.md, D2). Anyone who reaches the machine gets one identity frame with the sealed
//! readiness receipt; a machine-scope cap gets the whole picture, sent again whenever it changes.
//! An open stream is not activity: only `keepalive` resets the idle deadline, once.
use super::{auth::VerifiedActor, backend::MachineBackend, domain, v1, MachineIdentity};
use std::{pin::Pin, sync::Arc, time::Duration};
use tokio_stream::Stream;
use tonic::Status;

pub(super) type Frames = Pin<Box<dyn Stream<Item = Result<v1::StatusFrame, Status>> + Send>>;

/// What this machine serves on `cozy.machine.v1`, for clients that adapt to it.
/// `warm/1`: a warm run with no entrypoint installs its code and makes its model choices
/// present. `warm/2`: a warm run's `set` replaces the caller's warm set, and Status reports the
/// set and each environment's level. `upload/1`: a warm run of one provider source puts it in
/// its weights destination.
pub const CAPABILITIES: &[&str] = &[
    "status/1",
    "run/1",
    "control/1",
    "read/1",
    "warm/1",
    "warm/2",
    "upload/1",
    "local-models/1",
];

const LIVE: [&str; 4] = ["queued", "starting", "running", "paused"];

pub(super) async fn status<B: MachineBackend>(
    identity: Arc<MachineIdentity>,
    backend: Arc<B>,
    caller: Option<VerifiedActor>,
    keepalive: bool,
) -> Result<Frames, Status> {
    let Some(actor) = caller else {
        if keepalive {
            return Err(Status::permission_denied(
                "keepalive needs a machine-scope Cozy-Cap",
            ));
        }
        return Ok(Box::pin(tokio_stream::once(Ok(identity_frame(&identity)))));
    };
    if keepalive {
        identity
            .lifecycle
            .as_ref()
            .ok_or_else(|| Status::failed_precondition("this machine has no idle deadline"))?
            .renew()?;
    }
    let first = frame(&identity, &backend, actor).await?;
    let (sender, receiver) = tokio::sync::mpsc::channel(4);
    tokio::spawn(async move {
        let revoked = identity.authority.keys.revoked(actor.public_key);
        tokio::pin!(revoked);
        let mut last = first.clone();
        if sender.send(Ok(first)).await.is_err() {
            return;
        }
        let mut tick = tokio::time::interval(Duration::from_secs(1));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        tick.tick().await;
        loop {
            tokio::select! {
                _ = &mut revoked => {
                    let _ = sender.send(Err(Status::unauthenticated("the key that opened this stream no longer authorizes it"))).await;
                    return;
                }
                _ = sender.closed() => return,
                _ = tick.tick() => {}
            }
            let next = match frame(&identity, &backend, actor).await {
                Ok(next) => next,
                Err(status) => {
                    let _ = sender.send(Err(status)).await;
                    return;
                }
            };
            if changed(&last, &next) {
                last = next.clone();
                if sender.send(Ok(next)).await.is_err() {
                    return;
                }
            }
        }
    });
    Ok(Box::pin(tokio_stream::wrappers::ReceiverStream::new(
        receiver,
    )))
}

/// Disk use moves all the time; it alone is news once it moved by a hundredth of the disk.
fn changed(last: &v1::StatusFrame, next: &v1::StatusFrame) -> bool {
    let moved = match (&last.disk, &next.disk) {
        (Some(a), Some(b)) => a.free_bytes.abs_diff(b.free_bytes) >= b.total_bytes / 100,
        (a, b) => a != b,
    };
    moved
        || v1::StatusFrame {
            disk: None,
            ..last.clone()
        } != v1::StatusFrame {
            disk: None,
            ..next.clone()
        }
}

fn identity_frame(identity: &MachineIdentity) -> v1::StatusFrame {
    v1::StatusFrame {
        worker_id: identity.authority.worker_id.clone(),
        boot_id: identity.authority.boot_id.clone(),
        version: env!("CARGO_PKG_VERSION").into(),
        capabilities: capabilities(identity),
        receipt: identity.readiness.envelope().unwrap_or_default(),
        ..Default::default()
    }
}

pub(super) fn capabilities(identity: &MachineIdentity) -> Vec<String> {
    let update = identity.updates.as_ref().map(|_| "update/1");
    CAPABILITIES
        .iter()
        .copied()
        .chain(update)
        .map(str::to_string)
        .collect()
}

async fn frame<B: MachineBackend>(
    identity: &Arc<MachineIdentity>,
    backend: &Arc<B>,
    actor: VerifiedActor,
) -> Result<v1::StatusFrame, Status> {
    let (backend, store) = (backend.clone(), identity.store.clone());
    let ((runs, environments), warm, models) = tokio::task::spawn_blocking(move || {
        let models = store.as_deref().map(crate::held_models::listing);
        let held = held(&*backend, actor)?;
        Ok::<_, Status>((held, backend.warm_set(actor)?, models.unwrap_or_default()))
    })
    .await
    .map_err(|_| Status::internal("machine operation stopped"))??;
    let software = identity
        .updates
        .as_ref()
        .map(|updates| updates.software())
        .unwrap_or_default();
    let lifecycle = identity.lifecycle.as_ref();
    Ok(v1::StatusFrame {
        idle_deadline_unix_ms: lifecycle.map_or(0, |l| l.deadline_ms()),
        runs,
        phase: match lifecycle {
            Some(l) if l.released() => "releasing",
            _ if !identity.readiness.proved() => "booting",
            _ => "ready",
        }
        .into(),
        platform: format!("{}/{}", std::env::consts::OS, std::env::consts::ARCH),
        started_at_unix_ms: identity.started_at_ms as i64,
        gpus: identity
            .readiness
            .gpus()
            .into_iter()
            .map(|g| v1::Gpu {
                index: g.device_index,
                name: g.device_name,
                uuid: g.device_uuid,
                pci_bus_id: g.pci_bus_id,
                memory_bytes: g.memory_bytes,
                driver: g.driver_version,
            })
            .collect(),
        hubs: identity
            .hubs
            .iter()
            .map(|(origin, machine_id)| v1::Hub {
                origin: origin.clone(),
                machine_id: machine_id.clone(),
            })
            .collect(),
        runtime: software.runtime,
        tensorfs: software.tensorfs,
        environments,
        warm,
        disk: identity.store.as_deref().and_then(disk),
        models: models.models,
        models_bytes: models.bytes,
        webrtc: identity.webrtc_port.map(|port| {
            crate::machine::player::endpoint(port, &identity.player, &identity.cert_der)
        }),
        ..identity_frame(identity)
    })
}

fn held(
    backend: &impl MachineBackend,
    actor: VerifiedActor,
) -> Result<(Vec<v1::RunState>, Vec<v1::Environment>), Status> {
    let query = domain::MachineExecutionListQuery {
        limit: 256,
        states: LIVE.map(String::from).to_vec(),
        ..Default::default()
    };
    let runs = held_or_none(backend.list(actor, query).map(|l| l.executions))?
        .iter()
        .map(|run| super::machine_v1::state(&run.request_id, run))
        .collect();
    let packages = held_or_none(
        backend
            .list_packages(actor, domain::PackageListQuery::default())
            .map(|l| l.packages),
    )?;
    let levels = backend.levels(actor)?;
    let environments = packages
        .into_iter()
        .map(|p| v1::Environment {
            level: levels.get(&p.installation_id).copied().unwrap_or_default().into(),
            installation: p.installation_id,
            package: p.package,
            release: p.release,
            entrypoints: p.entrypoints,
            runtime: p
                .sdk
                .into_iter()
                .find(|d| d.distribution == "cozy-runtime")
                .map(|d| d.version)
                .unwrap_or_default(),
            warning: p.warning,
        })
        .collect();
    Ok((runs, environments))
}

/// A backend that keeps no such record holds none.
fn held_or_none<T>(listed: Result<Vec<T>, Status>) -> Result<Vec<T>, Status> {
    match listed {
        Err(status) if status.code() == tonic::Code::Unimplemented => Ok(vec![]),
        other => other,
    }
}

fn disk(path: &std::path::Path) -> Option<v1::Disk> {
    let stat = nix::sys::statvfs::statvfs(path).ok()?;
    let block = stat.fragment_size() as u64;
    Some(v1::Disk {
        total_bytes: stat.blocks() as u64 * block,
        free_bytes: stat.blocks_available() as u64 * block,
    })
}
