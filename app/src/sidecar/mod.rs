//! The Python sidecar (`127.0.0.1:8765`): HTTP client, API types and the
//! process supervisor.

mod client;
mod supervisor;
pub mod types;

pub use client::SidecarClient;
pub use supervisor::{Sidecar, SidecarGlobal, SidecarStatus, install_signal_handler};
pub use types::*;
