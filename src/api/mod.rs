pub mod auth;
pub mod backend;
mod server;

pub mod pb {
    tonic::include_proto!("cozy.worker.v1");
}

pub use backend::MachineBackend;
pub use server::{serve, MachineIdentity};
pub const WIRE_MINOR: u32 = 72;
pub const WIRE_MINIMUM: u32 = 64;
