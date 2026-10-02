pub mod auth;
pub mod backend;
pub mod identity;
pub mod install;
mod server;
pub mod workspaces;

#[allow(clippy::large_enum_variant)] // generated protobuf message shapes
pub mod pb {
    tonic::include_proto!("cozy.worker.v1");
}

pub use backend::MachineBackend;
pub use server::{serve, MachineIdentity};
pub const WIRE_MINOR: u32 = 72;
// Baseline identity/read operations consume their known fields at every minor.
// Missing operations are refused individually, never through a peer version floor.
pub const WIRE_MINIMUM: u32 = 0;
