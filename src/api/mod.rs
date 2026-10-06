pub mod auth;
pub mod backend;
pub mod domain;
pub mod capability;
mod cozy1;
pub mod identity;
pub mod install;
mod machine_status;
mod machine_update;
mod machine_v1;
pub mod retired;
mod server;
pub mod workspaces;


/// `cozy.machine.v1`, the machine's API (G/API.md).
pub mod v1 {
    tonic::include_proto!("cozy.machine.v1");
}

pub use backend::MachineBackend;
pub use machine_status::CAPABILITIES;
pub use server::{serve, MachineIdentity};
