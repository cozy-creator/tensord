//! Private journal codecs. Field numbers preserve already-held custody and terminal bytes.
//! No services, client stubs, or worker protocol identity live here.
use crate::api::domain;
use prost::Message;

#[derive(Clone, PartialEq, Eq, Hash, ::prost::Message)]
pub struct AttemptOutcome {
    #[prost(uint64, tag = "1")]
    pub record_owner_epoch: u64,
    #[prost(uint64, tag = "2")]
    pub control_stream_epoch: u64,
    #[prost(string, tag = "3")]
    pub worker_boot_id: ::prost::alloc::string::String,
    #[prost(string, tag = "5")]
    pub request_id: ::prost::alloc::string::String,
    #[prost(uint64, tag = "6")]
    pub attempt_ordinal: u64,
    #[prost(bytes = "vec", tag = "7")]
    pub invocation_spec_digest: ::prost::alloc::vec::Vec<u8>,
    #[prost(string, tag = "8")]
    pub outcome_id: ::prost::alloc::string::String,
    #[prost(bytes = "vec", tag = "9")]
    pub outcome_digest: ::prost::alloc::vec::Vec<u8>,
    #[prost(bytes = "vec", tag = "10")]
    pub outcome_canonical_bytes: ::prost::alloc::vec::Vec<u8>,
    #[prost(string, tag = "11")]
    pub placement_id: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, Eq, Hash, ::prost::Message)]
pub struct DownloadAdapterRef {
    #[prost(string, tag = "1")]
    pub component: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub model: ::prost::alloc::string::String,
    #[prost(string, tag = "3")]
    pub release: ::prost::alloc::string::String,
    #[prost(string, tag = "4")]
    pub lane: ::prost::alloc::string::String,
    #[prost(string, tag = "5")]
    pub manifest: ::prost::alloc::string::String,
    #[prost(string, tag = "6")]
    pub source_component: ::prost::alloc::string::String,
    #[prost(string, tag = "7")]
    pub scale: ::prost::alloc::string::String,
    #[prost(string, tag = "8")]
    pub source: ::prost::alloc::string::String,
    #[prost(string, repeated, tag = "9")]
    pub profiles: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct MachineExecutionEvent {
    #[prost(uint64, tag = "1")]
    pub sequence: u64,
    #[prost(uint64, tag = "2")]
    pub attempt_ordinal: u64,
    #[prost(uint64, tag = "3")]
    pub at_ms: u64,
    #[prost(string, tag = "4")]
    pub kind: ::prost::alloc::string::String,
    #[prost(bytes = "vec", tag = "5")]
    pub body_canonical_bytes: ::prost::alloc::vec::Vec<u8>,
    #[prost(message, optional, tag = "6")]
    pub product: ::core::option::Option<RunProduct>,
    #[prost(message, optional, tag = "7")]
    pub outcome: ::core::option::Option<AttemptOutcome>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct MachineExecutionEventPage {
    #[prost(message, repeated, tag = "1")]
    pub events: ::prost::alloc::vec::Vec<MachineExecutionEvent>,
    #[prost(uint64, tag = "2")]
    pub next_after: u64,
    #[prost(uint64, tag = "3")]
    pub head_sequence: u64,
    #[prost(uint64, tag = "4")]
    pub compacted_through: u64,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ModelChoice {
    #[prost(string, tag = "1")]
    pub parameter: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub repository: ::prost::alloc::string::String,
    #[prost(string, tag = "3")]
    pub release: ::prost::alloc::string::String,
    #[prost(string, tag = "4")]
    pub lane: ::prost::alloc::string::String,
    #[prost(message, optional, tag = "5")]
    pub manifest: ::core::option::Option<Ref>,
    #[prost(string, tag = "6")]
    pub source: ::prost::alloc::string::String,
    #[prost(string, repeated, tag = "7")]
    pub profiles: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    #[prost(message, repeated, tag = "8")]
    pub adapters: ::prost::alloc::vec::Vec<DownloadAdapterRef>,
}

#[derive(Clone, PartialEq, Eq, Hash, ::prost::Message)]
pub struct NativeByteRetentionRequest {
    #[prost(message, optional, tag = "1")]
    pub source: ::core::option::Option<NativeByteTreeRef>,
    #[prost(string, tag = "2")]
    pub retention_id: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, Eq, Hash, ::prost::Message)]
pub struct NativeByteRetentionResult {
    #[prost(message, optional, tag = "1")]
    pub source: ::core::option::Option<NativeByteTreeRef>,
    #[prost(string, tag = "2")]
    pub retention_id: ::prost::alloc::string::String,
    #[prost(bool, tag = "3")]
    pub released: bool,
}

#[derive(Clone, PartialEq, Eq, Hash, ::prost::Message)]
pub struct NativeByteTreeRef {
    #[prost(string, tag = "1")]
    pub producer_root_id: ::prost::alloc::string::String,
    #[prost(bytes = "vec", tag = "2")]
    pub receipt_digest: ::prost::alloc::vec::Vec<u8>,
    #[prost(message, optional, tag = "3")]
    pub manifest: ::core::option::Option<Ref>,
    #[prost(uint64, tag = "4")]
    pub content_bytes: u64,
}

#[derive(Clone, PartialEq, Eq, Hash, ::prost::Message)]
pub struct Ref {
    #[prost(bytes = "vec", tag = "1")]
    pub digest: ::prost::alloc::vec::Vec<u8>,
    #[prost(uint64, tag = "2")]
    pub length: u64,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RunProduct {
    #[prost(string, tag = "1")]
    pub output: ::prost::alloc::string::String,
    #[prost(int32, tag = "2")]
    pub op: i32,
    #[prost(uint32, tag = "3")]
    pub index: u32,
    #[prost(message, optional, tag = "4")]
    pub content: ::core::option::Option<Ref>,
    #[prost(string, tag = "5")]
    pub media_type: ::prost::alloc::string::String,
    #[prost(string, tag = "6")]
    pub label: ::prost::alloc::string::String,
    #[prost(message, optional, tag = "7")]
    pub source: ::core::option::Option<NativeByteRetentionRequest>,
    #[prost(message, repeated, tag = "8")]
    pub parts: ::prost::alloc::vec::Vec<RunProductPart>,
}

#[derive(Clone, PartialEq, Eq, Hash, ::prost::Message)]
pub struct RunProductPart {
    #[prost(message, optional, tag = "1")]
    pub content: ::core::option::Option<Ref>,
    #[prost(message, optional, tag = "2")]
    pub source: ::core::option::Option<NativeByteRetentionRequest>,
    #[prost(uint64, tag = "3")]
    pub duration_us: u64,
}

impl From<domain::AttemptOutcome> for AttemptOutcome {
    fn from(value: domain::AttemptOutcome) -> Self {
        Self {
            record_owner_epoch: Default::default(),
            control_stream_epoch: Default::default(),
            worker_boot_id: value.worker_boot_id,
            request_id: value.request_id,
            attempt_ordinal: value.attempt_ordinal,
            invocation_spec_digest: value.invocation_spec_digest,
            outcome_id: value.outcome_id,
            outcome_digest: value.outcome_digest,
            outcome_canonical_bytes: value.outcome_canonical_bytes,
            placement_id: Default::default(),
        }
    }
}

impl From<AttemptOutcome> for domain::AttemptOutcome {
    fn from(value: AttemptOutcome) -> Self {
        Self {
            worker_boot_id: value.worker_boot_id,
            request_id: value.request_id,
            attempt_ordinal: value.attempt_ordinal,
            invocation_spec_digest: value.invocation_spec_digest,
            outcome_id: value.outcome_id,
            outcome_digest: value.outcome_digest,
            outcome_canonical_bytes: value.outcome_canonical_bytes,
        }
    }
}

impl From<domain::DownloadAdapterRef> for DownloadAdapterRef {
    fn from(value: domain::DownloadAdapterRef) -> Self {
        Self {
            component: value.component,
            model: value.model,
            release: value.release,
            lane: value.lane,
            manifest: value.manifest,
            source_component: value.source_component,
            scale: value.scale,
            source: value.source,
            profiles: value.profiles,
        }
    }
}

impl From<DownloadAdapterRef> for domain::DownloadAdapterRef {
    fn from(value: DownloadAdapterRef) -> Self {
        Self {
            component: value.component,
            model: value.model,
            release: value.release,
            lane: value.lane,
            manifest: value.manifest,
            source_component: value.source_component,
            scale: value.scale,
            source: value.source,
            profiles: value.profiles,
        }
    }
}

impl From<domain::MachineExecutionEvent> for MachineExecutionEvent {
    fn from(value: domain::MachineExecutionEvent) -> Self {
        Self {
            sequence: value.sequence,
            attempt_ordinal: value.attempt_ordinal,
            at_ms: value.at_ms,
            kind: value.kind,
            body_canonical_bytes: value.body_canonical_bytes,
            product: value.product.map(Into::into),
            outcome: value.outcome.map(Into::into),
        }
    }
}

impl From<MachineExecutionEvent> for domain::MachineExecutionEvent {
    fn from(value: MachineExecutionEvent) -> Self {
        Self {
            sequence: value.sequence,
            attempt_ordinal: value.attempt_ordinal,
            at_ms: value.at_ms,
            kind: value.kind,
            body_canonical_bytes: value.body_canonical_bytes,
            product: value.product.map(Into::into),
            outcome: value.outcome.map(Into::into),
        }
    }
}

impl From<domain::MachineExecutionEventPage> for MachineExecutionEventPage {
    fn from(value: domain::MachineExecutionEventPage) -> Self {
        Self {
            events: value.events.into_iter().map(Into::into).collect(),
            next_after: value.next_after,
            head_sequence: value.head_sequence,
            compacted_through: value.compacted_through,
        }
    }
}

impl From<MachineExecutionEventPage> for domain::MachineExecutionEventPage {
    fn from(value: MachineExecutionEventPage) -> Self {
        Self {
            events: value.events.into_iter().map(Into::into).collect(),
            next_after: value.next_after,
            head_sequence: value.head_sequence,
            compacted_through: value.compacted_through,
        }
    }
}

pub fn encode_machine_execution_event_page(value: &domain::MachineExecutionEventPage) -> Vec<u8> {
    MachineExecutionEventPage::from(value.clone()).encode_to_vec()
}

pub fn decode_machine_execution_event_page(
    bytes: &[u8],
) -> Result<domain::MachineExecutionEventPage, prost::DecodeError> {
    MachineExecutionEventPage::decode(bytes).map(Into::into)
}

impl From<domain::ModelChoice> for ModelChoice {
    fn from(value: domain::ModelChoice) -> Self {
        Self {
            parameter: value.parameter,
            repository: value.repository,
            release: value.release,
            lane: value.lane,
            manifest: value.manifest.map(Into::into),
            source: value.source,
            profiles: value.profiles,
            adapters: value.adapters.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<ModelChoice> for domain::ModelChoice {
    fn from(value: ModelChoice) -> Self {
        Self {
            parameter: value.parameter,
            repository: value.repository,
            release: value.release,
            lane: value.lane,
            manifest: value.manifest.map(Into::into),
            source: value.source,
            profiles: value.profiles,
            adapters: value.adapters.into_iter().map(Into::into).collect(),
        }
    }
}

pub fn encode_model_choice(value: &domain::ModelChoice) -> Vec<u8> {
    ModelChoice::from(value.clone()).encode_to_vec()
}

pub fn decode_model_choice(bytes: &[u8]) -> Result<domain::ModelChoice, prost::DecodeError> {
    ModelChoice::decode(bytes).map(Into::into)
}

impl From<domain::NativeByteRetentionRequest> for NativeByteRetentionRequest {
    fn from(value: domain::NativeByteRetentionRequest) -> Self {
        Self {
            source: value.source.map(Into::into),
            retention_id: value.retention_id,
        }
    }
}

impl From<NativeByteRetentionRequest> for domain::NativeByteRetentionRequest {
    fn from(value: NativeByteRetentionRequest) -> Self {
        Self {
            source: value.source.map(Into::into),
            retention_id: value.retention_id,
        }
    }
}

pub fn encode_native_byte_retention_request(value: &domain::NativeByteRetentionRequest) -> Vec<u8> {
    NativeByteRetentionRequest::from(value.clone()).encode_to_vec()
}

impl From<domain::NativeByteRetentionResult> for NativeByteRetentionResult {
    fn from(value: domain::NativeByteRetentionResult) -> Self {
        Self {
            source: value.source.map(Into::into),
            retention_id: value.retention_id,
            released: value.released,
        }
    }
}

impl From<NativeByteRetentionResult> for domain::NativeByteRetentionResult {
    fn from(value: NativeByteRetentionResult) -> Self {
        Self {
            source: value.source.map(Into::into),
            retention_id: value.retention_id,
            released: value.released,
        }
    }
}

pub fn encode_native_byte_retention_result(value: &domain::NativeByteRetentionResult) -> Vec<u8> {
    NativeByteRetentionResult::from(value.clone()).encode_to_vec()
}

pub fn decode_native_byte_retention_result(
    bytes: &[u8],
) -> Result<domain::NativeByteRetentionResult, prost::DecodeError> {
    NativeByteRetentionResult::decode(bytes).map(Into::into)
}

impl From<domain::NativeByteTreeRef> for NativeByteTreeRef {
    fn from(value: domain::NativeByteTreeRef) -> Self {
        Self {
            producer_root_id: value.producer_root_id,
            receipt_digest: value.receipt_digest,
            manifest: value.manifest.map(Into::into),
            content_bytes: value.content_bytes,
        }
    }
}

impl From<NativeByteTreeRef> for domain::NativeByteTreeRef {
    fn from(value: NativeByteTreeRef) -> Self {
        Self {
            producer_root_id: value.producer_root_id,
            receipt_digest: value.receipt_digest,
            manifest: value.manifest.map(Into::into),
            content_bytes: value.content_bytes,
        }
    }
}

impl From<domain::Ref> for Ref {
    fn from(value: domain::Ref) -> Self {
        Self {
            digest: value.digest,
            length: value.length,
        }
    }
}

impl From<Ref> for domain::Ref {
    fn from(value: Ref) -> Self {
        Self {
            digest: value.digest,
            length: value.length,
        }
    }
}

impl From<domain::RunProduct> for RunProduct {
    fn from(value: domain::RunProduct) -> Self {
        Self {
            output: value.output,
            op: value.op,
            index: value.index,
            content: value.content.map(Into::into),
            media_type: value.media_type,
            label: value.label,
            source: value.source.map(Into::into),
            parts: value.parts.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<RunProduct> for domain::RunProduct {
    fn from(value: RunProduct) -> Self {
        Self {
            output: value.output,
            op: value.op,
            index: value.index,
            content: value.content.map(Into::into),
            media_type: value.media_type,
            label: value.label,
            source: value.source.map(Into::into),
            parts: value.parts.into_iter().map(Into::into).collect(),
        }
    }
}

pub fn encode_run_product(value: &domain::RunProduct) -> Vec<u8> {
    RunProduct::from(value.clone()).encode_to_vec()
}

pub fn decode_run_product(bytes: &[u8]) -> Result<domain::RunProduct, prost::DecodeError> {
    RunProduct::decode(bytes).map(Into::into)
}

impl From<domain::RunProductPart> for RunProductPart {
    fn from(value: domain::RunProductPart) -> Self {
        Self {
            content: value.content.map(Into::into),
            source: value.source.map(Into::into),
            duration_us: value.duration_us,
        }
    }
}

impl From<RunProductPart> for domain::RunProductPart {
    fn from(value: RunProductPart) -> Self {
        Self {
            content: value.content.map(Into::into),
            source: value.source.map(Into::into),
            duration_us: value.duration_us,
        }
    }
}

pub fn encode_attempt_outcome(value: &domain::AttemptOutcome) -> Vec<u8> {
    AttemptOutcome::from(value.clone()).encode_to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bytes emitted by the pre-retirement native machine remain readable and stable. These
    /// are stored custody/journal records, not a deployed-peer synchronization requirement.
    #[test]
    fn stored_native_custody_and_terminal_codecs_keep_their_bytes() {
        let model_bytes = include_bytes!("../tests/fixtures/journal-codecs/model.bin");
        let model = decode_model_choice(model_bytes).unwrap();
        assert_eq!(model.parameter, "callee/render.models.source");
        assert_eq!(model.adapters[0].scale, "0.5");
        assert_eq!(encode_model_choice(&model), model_bytes);

        let product_bytes = include_bytes!("../tests/fixtures/journal-codecs/product.bin");
        let product = decode_run_product(product_bytes).unwrap();
        assert_eq!(product.output, "files");
        assert_eq!(product.content.as_ref().unwrap().length, 7);
        assert_eq!(product.source.as_ref().unwrap().retention_id, "retention");
        assert_eq!(encode_run_product(&product), product_bytes);

        let intake_bytes = include_bytes!("../tests/fixtures/journal-codecs/intake.bin");
        let intake = decode_native_byte_retention_result(intake_bytes).unwrap();
        assert_eq!(intake.source.as_ref().unwrap().content_bytes, 8);
        assert!(!intake.released);
        assert_eq!(encode_native_byte_retention_result(&intake), intake_bytes);

        let terminal_bytes = include_bytes!("../tests/fixtures/journal-codecs/terminal.bin");
        let terminal = decode_machine_execution_event_page(terminal_bytes).unwrap();
        assert_eq!(terminal.events[0].product.as_ref().unwrap(), &product);
        assert_eq!(
            terminal.events[1]
                .outcome
                .as_ref()
                .unwrap()
                .outcome_canonical_bytes,
            br#"{"status":4}"#
        );
        assert_eq!(
            encode_machine_execution_event_page(&terminal),
            terminal_bytes
        );
    }
}
