pub mod auth;
pub mod backend;
pub mod capability;
pub mod identity;
pub mod install;
mod machine_status;
mod machine_update;
mod machine_v1;
mod server;
pub mod workspaces;

#[allow(clippy::large_enum_variant)] // generated protobuf message shapes
pub mod pb {
    tonic::include_proto!("cozy.worker.v1");
}

/// `cozy.machine.v1`, the machine's API (G/API.md).
pub mod v1 {
    tonic::include_proto!("cozy.machine.v1");
}

pub use backend::MachineBackend;
pub use server::{serve, MachineIdentity};
pub const WIRE_MINOR: u32 = 72;
// The deployed baseline the Go agent reports; released CLIs read 0 as "no usable range".
// This machine still refuses no peer by version: a missing operation fails alone.
pub const WIRE_MINIMUM: u32 = 64;
