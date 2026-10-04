//! One adapter into the owned execution engine. This module owns no journal.
use super::{auth::VerifiedActor, pb, v1};
use std::collections::BTreeMap;
use tonic::Status;

pub trait MachineBackend: Send + Sync + 'static {
    /// `cozy.machine.v1` Run sources and Write (`runs`).
    fn runs(&self) -> Option<std::sync::Arc<crate::runs::Runs>> {
        None
    }
    fn list_packages(
        &self,
        _: VerifiedActor,
        _: pb::PackageListQuery,
    ) -> Result<pb::PackageList, Status> {
        unsupported()
    }
    /// What each of the caller's installations holds now (`Environment.level`), by
    /// installation id.
    fn levels(&self, _: VerifiedActor) -> Result<BTreeMap<String, &'static str>, Status> {
        Ok(BTreeMap::new())
    }
    /// The caller's warm set, each member with what it holds now (`StatusFrame.warm`).
    fn warm_set(&self, _: VerifiedActor) -> Result<Vec<v1::WarmItem>, Status> {
        Ok(vec![])
    }
    fn workspace(
        &self,
        _: VerifiedActor,
        _: pb::MachineExecutionWorkspaceQuery,
    ) -> Result<pb::MachineExecutionWorkspace, Status>;
    fn get(
        &self,
        _: VerifiedActor,
        _: pb::MachineExecutionQuery,
    ) -> Result<pb::MachineExecutionState, Status> {
        unsupported()
    }
    fn events(
        &self,
        _: VerifiedActor,
        _: pb::MachineExecutionEventsQuery,
    ) -> Result<pb::MachineExecutionEventPage, Status> {
        unsupported()
    }
    /// What a run's executor measured (canonical JSON), when it ran on a device.
    fn measurements(
        &self,
        _: VerifiedActor,
        _: pb::MachineExecutionQuery,
    ) -> Result<Option<Vec<u8>>, Status> {
        Ok(None)
    }
    /// A run's own time running its callable, every attempt summed (0: unknown).
    fn execution_ms(&self, _: VerifiedActor, _: pb::MachineExecutionQuery) -> Result<u64, Status> {
        Ok(0)
    }
    fn control(
        &self,
        _: VerifiedActor,
        _: pb::MachineExecutionControl,
    ) -> Result<pb::MachineExecutionState, Status> {
        unsupported()
    }
    fn list(
        &self,
        _: VerifiedActor,
        _: pb::MachineExecutionListQuery,
    ) -> Result<pb::MachineExecutionList, Status> {
        unsupported()
    }
    /// One run output's current bytes (`GET /v1/runs/{run}/outputs/{output}[/{index}]`).
    /// `index` is a list item's 1-based index. The caller has verified `actor`'s capability;
    /// a run another actor submitted is absent.
    fn open_output(
        &self,
        _: VerifiedActor,
        _run: u64,
        _output: &str,
        _index: Option<u32>,
    ) -> Result<OutputSnapshot, Status> {
        unsupported()
    }
    /// One kept log's bytes, oldest first (wire 72). A log not written yet is empty.
    /// A failed attempt's retained triage bundle.
    fn read_triage(
        &self,
        _: VerifiedActor,
        _: pb::MachineExecutionTriageQuery,
    ) -> Result<pb::MachineExecutionTriage, Status> {
        unsupported()
    }
    fn read_machine_log(
        &self,
        _: VerifiedActor,
        _: pb::MachineLogQuery,
    ) -> Result<Vec<u8>, Status> {
        unsupported()
    }
    /// A complete content-addressed object this signer wrote with `Write`, verified against
    /// `sha256` (hex).
    fn object(&self, _: VerifiedActor, _sha256: &str) -> Result<std::fs::File, Status> {
        unsupported()
    }
}
/// One consistent view of an output's current bytes: its parts in order.
pub struct OutputSnapshot {
    pub parts: Vec<(std::fs::File, u64)>,
    pub length: u64,
    /// The 1-based ordinal of this (output, index)'s products in the run's log.
    pub rev: u64,
    pub media_type: String,
    /// `sha256:<hex>`, set once the run is terminal (the output is final).
    pub sha256: Option<String>,
}
/// One bounded native-input transfer. The store/journal owner validates identity,
/// membership, contiguous offsets, native custody and durable commit/abort semantics.
/// Dropping a disconnected transfer never supplies execution cancellation authority.
pub trait InputTreeReceiver: Send {
    fn blob(&mut self, blob: pb::InputTreeImportBlob) -> Result<(), Status>;
    fn commit(
        self: Box<Self>,
        commit: pb::InputTreeImportCommit,
    ) -> Result<pb::NativeByteRetentionResult, Status>;
}

fn unsupported<T>() -> Result<T, Status> {
    Err(Status::unimplemented(
        "capability_unavailable: this machine backend does not implement this operation",
    ))
}
