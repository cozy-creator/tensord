//! Current native execution values. These are not an RPC protocol.

#[derive(Clone, Debug, Default, PartialEq)]
pub struct AttemptOutcome {
    pub worker_boot_id: String,
    pub request_id: String,
    pub attempt_ordinal: u64,
    pub invocation_spec_digest: Vec<u8>,
    pub outcome_id: String,
    pub outcome_digest: Vec<u8>,
    pub outcome_canonical_bytes: Vec<u8>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DesiredLocalPackageSet {
    pub operation_id: String,
    pub package: Option<DevelopmentPackage>,
    pub files: Vec<LocalPackageFileRef>,
    pub dependency_requirements: Vec<u8>,
    pub python_requires: String,
    pub python_version: String,
    pub source_archive: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DevelopmentPackage {
    pub package: String,
    pub release: String,
    pub installation_id: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DownloadAdapterRef {
    pub component: String,
    pub model: String,
    pub release: String,
    pub lane: String,
    pub manifest: String,
    pub source_component: String,
    pub scale: String,
    pub source: String,
    pub profiles: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImageDistribution {
    pub distribution: String,
    pub version: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct InputTreeImportBlob {
    pub object: Option<Ref>,
    pub offset: u64,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct InputTreeImportCommit {
    pub abort: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct InputTreeImportHeader {
    pub request_id: String,
    pub input_id: String,
    pub manifest: Option<Ref>,
    pub manifest_canonical_bytes: Vec<u8>,
    pub content_bytes: u64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LocalPackageFileRef {
    pub digest: Vec<u8>,
    pub filename: String,
    pub length: u64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LocalPackageUploadChunk {
    pub offset: u64,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LocalPackageUploadHeader {
    pub operation_id: String,
    pub file: Option<LocalPackageFileRef>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ::prost::Enumeration)]
#[repr(i32)]
pub enum MachineExecutionAction {
    Unspecified = 0,
    Pause = 1,
    Resume = 2,
    Cancel = 3,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MachineExecutionControl {
    pub execution: Option<MachineExecutionQuery>,
    pub action: i32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MachineExecutionEvent {
    pub sequence: u64,
    pub attempt_ordinal: u64,
    pub at_ms: u64,
    pub kind: String,
    pub body_canonical_bytes: Vec<u8>,
    pub product: Option<RunProduct>,
    pub outcome: Option<AttemptOutcome>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MachineExecutionEventPage {
    pub events: Vec<MachineExecutionEvent>,
    pub next_after: u64,
    pub head_sequence: u64,
    pub compacted_through: u64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MachineExecutionEventsQuery {
    pub execution: Option<MachineExecutionQuery>,
    pub after: u64,
    pub limit: u32,
    pub wait: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MachineExecutionList {
    pub executions: Vec<MachineExecutionState>,
    pub head_number: u64,
    pub execution_workspace_id: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MachineExecutionListQuery {
    pub after_number: u64,
    pub newest_first: bool,
    pub before_number: u64,
    pub limit: u32,
    pub states: Vec<String>,
    pub wait: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MachineExecutionQuery {
    pub request_id: String,
    pub expected_execution_workspace_id: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MachineExecutionState {
    pub request_id: String,
    pub attempt_ordinal: u64,
    pub generation: u64,
    pub state: String,
    pub collected: bool,
    pub sequence: u64,
    pub worker_id: String,
    pub worker_boot_id: String,
    pub execution_workspace_id: String,
    pub number: u64,
    pub accepted_at_ms: u64,
    pub finished_at_ms: u64,
    pub target: Option<MachineExecutionTarget>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MachineExecutionTarget {
    pub package: String,
    pub release: String,
    pub entrypoint: String,
    pub installation_id: String,
    pub job: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MachineExecutionTriage {
    pub bundle: Option<TriageBundleRef>,
    pub bundle_canonical_bytes: Vec<u8>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MachineExecutionTriageQuery {
    pub execution: Option<MachineExecutionQuery>,
    pub attempt_ordinal: u64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MachineExecutionWorkspace {
    pub worker_id: String,
    pub worker_boot_id: String,
    pub execution_workspace_id: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MachineExecutionWorkspaceQuery {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ::prost::Enumeration)]
#[repr(i32)]
pub enum MachineLog {
    Unspecified = 0,
    TensorfsTransport = 1,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MachineLogQuery {
    pub log: i32,
    pub tail_bytes: u64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MachinePackage {
    pub installation_id: String,
    pub package: String,
    pub release: String,
    pub origin: String,
    pub installed_at_ms: u64,
    pub sdk: Vec<ImageDistribution>,
    pub entrypoints: Vec<String>,
    /// Why this environment does not run the machine's own SDK; empty when it does.
    pub warning: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelChoice {
    pub parameter: String,
    pub repository: String,
    pub release: String,
    pub lane: String,
    pub manifest: Option<Ref>,
    pub source: String,
    pub profiles: Vec<String>,
    pub adapters: Vec<DownloadAdapterRef>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct NativeByteRetentionRequest {
    pub source: Option<NativeByteTreeRef>,
    pub retention_id: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct NativeByteRetentionResult {
    pub source: Option<NativeByteTreeRef>,
    pub retention_id: String,
    pub released: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct NativeByteTreeRef {
    pub producer_root_id: String,
    pub receipt_digest: Vec<u8>,
    pub manifest: Option<Ref>,
    pub content_bytes: u64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PackageList {
    pub packages: Vec<MachinePackage>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PackageListQuery {}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Ref {
    pub digest: Vec<u8>,
    pub length: u64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RunProduct {
    pub output: String,
    pub op: i32,
    pub index: u32,
    pub content: Option<Ref>,
    pub media_type: String,
    pub label: String,
    pub source: Option<NativeByteRetentionRequest>,
    pub parts: Vec<RunProductPart>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ::prost::Enumeration)]
#[repr(i32)]
pub enum RunProductOp {
    Unspecified = 0,
    Set = 1,
    Append = 2,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RunProductPart {
    pub content: Option<Ref>,
    pub source: Option<NativeByteRetentionRequest>,
    pub duration_us: u64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TriageBundleRef {
    pub subject_id: String,
    pub write_receipt_digest: Vec<u8>,
    pub length: u64,
}
