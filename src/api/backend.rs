//! One adapter into the owned execution engine. This module owns no journal.
use super::{
    auth::VerifiedActor,
    pb,
    workspaces::{UploadedPackage, WorkspaceUploads},
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use tonic::Status;

pub trait MachineBackend: Send + Sync + 'static {
    fn workspace(
        &self,
        _: VerifiedActor,
        _: pb::MachineExecutionWorkspaceQuery,
    ) -> Result<pb::MachineExecutionWorkspace, Status>;
    fn submit(
        &self,
        _: VerifiedActor,
        _: pb::MachineExecutionSubmit,
    ) -> Result<pb::MachineExecutionReceipt, Status> {
        unsupported()
    }
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
    fn events_observed(
        &self,
        actor: VerifiedActor,
        request: pb::MachineExecutionEventsQuery,
        _: Observation,
    ) -> Result<pb::MachineExecutionEventPage, Status> {
        self.events(actor, request)
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
    fn close_submission(
        &self,
        _: VerifiedActor,
        _: pb::MachineSubmissionClose,
    ) -> Result<pb::MachineSubmissionClosure, Status> {
        unsupported()
    }
    fn collect(
        &self,
        _: VerifiedActor,
        _: pb::MachineExecutionCollect,
    ) -> Result<pb::AttemptOutcome, Status> {
        unsupported()
    }
    fn ack_collection(
        &self,
        _: VerifiedActor,
        _: pb::MachineExecutionCollectionAck,
    ) -> Result<pb::MachineExecutionState, Status> {
        unsupported()
    }
    fn read_bytes(
        &self,
        _: VerifiedActor,
        _: pb::NativeByteReadCall,
    ) -> Result<Vec<pb::NativeByteReadChunk>, Status> {
        unsupported()
    }
    fn read_stream(
        &self,
        actor: VerifiedActor,
        request: pb::NativeByteReadCall,
    ) -> Result<NativeByteStream, Status> {
        Ok(Box::new(
            self.read_bytes(actor, request)?.into_iter().map(Ok),
        ))
    }
    fn uploads(&self) -> Option<Arc<WorkspaceUploads>> {
        None
    }
    fn prepare_local(
        &self,
        _: VerifiedActor,
        _: pb::PrepareLocalPackageCall,
        _: Option<UploadedPackage>,
    ) -> Result<Vec<pb::PrepareEvent>, Status> {
        unsupported()
    }
}
pub type NativeByteStream =
    Box<dyn Iterator<Item = Result<pb::NativeByteReadChunk, Status>> + Send>;

/// Observation cancellation only: it carries no durable run-control authority.
#[derive(Clone, Default)]
pub struct Observation {
    canceled: Arc<AtomicBool>,
}
impl Observation {
    pub fn canceled(&self) -> bool {
        self.canceled.load(Ordering::Acquire)
    }
    pub(crate) fn cancel(&self) {
        self.canceled.store(true, Ordering::Release);
    }
}

fn unsupported<T>() -> Result<T, Status> {
    Err(Status::unimplemented(
        "capability_unavailable: this machine backend does not implement this operation",
    ))
}
