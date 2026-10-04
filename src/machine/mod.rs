//! The machine role on a pod or this computer: its launch grant, lifetime identity,
//! readiness receipt and rental lifecycle. Contracts are the Go agent's and the Hub's.
pub mod client;
pub mod grant;
pub mod hub;
pub mod identity;
pub mod lifecycle;
pub mod probe;
pub mod pypi;
pub mod receipt;
pub mod ssh;
pub mod supervise;
pub mod update;

/// Machine contracts this service implements, as named in its readiness receipt.
pub const CAPABILITIES: &[&str] = &["hub-access/1", "runtime-update/1", "machine-bootstrap/1"];

pub mod media;
