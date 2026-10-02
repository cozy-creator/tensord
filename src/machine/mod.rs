//! The machine role on a pod or this computer: its launch grant, lifetime identity,
//! readiness receipt and rental lifecycle. Contracts are the Go agent's and the Hub's.
pub mod grant;
pub mod hub;
pub mod identity;
pub mod lifecycle;
pub mod probe;
pub mod receipt;

/// Machine contracts this service implements, as named in its readiness receipt.
pub const CAPABILITIES: &[&str] = &[];
