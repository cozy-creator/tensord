//! One adapter into the owned execution engine. This module owns no journal.
use super::pb;
use tonic::Status;

pub trait MachineBackend: Send + Sync + 'static {
    fn workspace(
        &self,
        _: pb::MachineExecutionWorkspaceQuery,
    ) -> Result<pb::MachineExecutionWorkspace, Status>;
    fn submit(&self, _: pb::MachineExecutionSubmit) -> Result<pb::MachineExecutionReceipt, Status> {
        unsupported()
    }
    fn get(&self, _: pb::MachineExecutionQuery) -> Result<pb::MachineExecutionState, Status> {
        unsupported()
    }
    fn events(
        &self,
        _: pb::MachineExecutionEventsQuery,
    ) -> Result<pb::MachineExecutionEventPage, Status> {
        unsupported()
    }
    fn control(&self, _: pb::MachineExecutionControl) -> Result<pb::MachineExecutionState, Status> {
        unsupported()
    }
    fn list(&self, _: pb::MachineExecutionListQuery) -> Result<pb::MachineExecutionList, Status> {
        unsupported()
    }
    fn close_submission(
        &self,
        _: pb::MachineSubmissionClose,
    ) -> Result<pb::MachineSubmissionClosure, Status> {
        unsupported()
    }
    fn collect(&self, _: pb::MachineExecutionCollect) -> Result<pb::AttemptOutcome, Status> {
        unsupported()
    }
    fn ack_collection(
        &self,
        _: pb::MachineExecutionCollectionAck,
    ) -> Result<pb::MachineExecutionState, Status> {
        unsupported()
    }
    fn read_bytes(
        &self,
        _: pb::NativeByteReadCall,
    ) -> Result<Vec<pb::NativeByteReadChunk>, Status> {
        unsupported()
    }
}

fn unsupported<T>() -> Result<T, Status> {
    Err(Status::unimplemented(
        "capability_unavailable: this machine backend does not implement this operation",
    ))
}
