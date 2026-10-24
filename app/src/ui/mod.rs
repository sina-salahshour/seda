//! Views. `app::SedaApp` hosts the sidebar and one page at a time; each page
//! is an `Entity` that lives for the app's lifetime so its state (selection,
//! drafts, running streams) survives page switches.

pub mod chats;
pub mod common;
pub mod models;
pub mod settings;
pub mod voices;
