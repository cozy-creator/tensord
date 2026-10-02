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

/// An HTTP refusal of `/v1/hubs/access`: status, stable code, message.
pub type HubAccessRefusal = (u16, &'static str, String);

pub trait MachineBackend: Send + Sync + 'static {
    /// Retains delegated Hub access for this owner key; answers the retained origin.
    fn hub_access(
        &self,
        _: VerifiedActor,
        _: crate::hub::Access,
    ) -> Result<(String, i64), HubAccessRefusal> {
        Err((
            404,
            "capability_unavailable",
            "this machine does not accept Hub access".into(),
        ))
    }
    fn forget_hub_access(&self, _: VerifiedActor, _: &str) -> Result<(), HubAccessRefusal> {
        Err((
            404,
            "capability_unavailable",
            "this machine does not accept Hub access".into(),
        ))
    }
    fn describe_runtime(&self, _: VerifiedActor) -> Result<pb::MachineRuntime, Status> {
        unsupported()
    }
    fn list_packages(
        &self,
        _: VerifiedActor,
        _: pb::PackageListQuery,
    ) -> Result<pb::PackageList, Status> {
        unsupported()
    }
    fn list_models(
        &self,
        _: VerifiedActor,
        _: pb::ModelListQuery,
    ) -> Result<pb::ModelList, Status> {
        unsupported()
    }
    fn retain_bytes(
        &self,
        _: VerifiedActor,
        _: pb::NativeByteRetentionCall,
    ) -> Result<pb::NativeByteRetentionResult, Status> {
        unsupported()
    }
    fn release_bytes(
        &self,
        _: VerifiedActor,
        _: pb::NativeByteRetentionCall,
    ) -> Result<pb::NativeByteRetentionResult, Status> {
        unsupported()
    }
    fn begin_input_tree(
        &self,
        _: VerifiedActor,
        _: pb::InputTreeImportHeader,
    ) -> Result<Box<dyn InputTreeReceiver>, Status> {
        unsupported()
    }
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
    fn list_observed(
        &self,
        actor: VerifiedActor,
        request: pb::MachineExecutionListQuery,
        _: Observation,
    ) -> Result<pb::MachineExecutionList, Status> {
        self.list(actor, request)
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
    /// One kept log's bytes, oldest first (wire 72). A log not written yet is empty.
    fn read_machine_log(
        &self,
        _: VerifiedActor,
        _: pb::MachineLogQuery,
    ) -> Result<Vec<u8>, Status> {
        unsupported()
    }
}
pub type NativeByteStream =
    Box<dyn Iterator<Item = Result<pb::NativeByteReadChunk, Status>> + Send>;

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
