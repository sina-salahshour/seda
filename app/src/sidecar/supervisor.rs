//! Keeps track of the sidecar: polls `/health`, starts it with `uv run` when
//! it is down and autostart is on, and kills the child it started on quit.

use std::{
    os::unix::process::CommandExt as _,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

use gpui_kit::{App, AppContext as _, Context, Entity, Global, Task};

use super::{SidecarClient, types::Health};
use crate::store::{AppSettings, data_dir};

/// Process group of the sidecar we spawned (0 = none), for the signal handler.
static CHILD_PGID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Whether we can create files in `dir` (probes with a throwaway file).
fn is_writable(dir: &std::path::Path) -> bool {
    let probe = dir.join(format!(".seda-write-test-{}", std::process::id()));
    let ok = std::fs::File::create(&probe).is_ok();
    _ = std::fs::remove_file(&probe);
    ok
}

/// On SIGTERM/SIGINT/SIGHUP, stop the sidecar we started and exit, so a killed
/// app does not leave the GPU-hungry child behind. Quitting normally goes
/// through `Sidecar::shutdown` instead.
pub fn install_signal_handler() {
    use signal_hook::consts::{SIGHUP, SIGINT, SIGTERM};
    let Ok(mut signals) = signal_hook::iterator::Signals::new([SIGTERM, SIGINT, SIGHUP]) else {
        return;
    };
    std::thread::spawn(move || {
        if let Some(signal) = signals.forever().next() {
            let pgid = CHILD_PGID.load(std::sync::atomic::Ordering::SeqCst);
            if pgid != 0 {
                _ = Command::new("kill")
                    .args(["-TERM", &format!("-{pgid}")])
                    .status();
            }
            std::process::exit(128 + signal);
        }
    });
}

const POLL: Duration = Duration::from_secs(3);
const START_TIMEOUT: Duration = Duration::from_secs(240);

#[derive(Debug, Clone, PartialEq)]
pub enum SidecarStatus {
    /// First health check has not answered yet.
    Checking,
    /// We spawned it and are waiting for `/health`.
    Starting,
    Online(Health),
    Offline(String),
}

pub struct Sidecar {
    status: SidecarStatus,
    child: Option<Child>,
    started_at: Option<Instant>,
    checking: bool,
    _poll: Task<()>,
}

/// The app-wide sidecar entity, reachable from any view with `SidecarGlobal::get(cx)`.
pub struct SidecarGlobal(pub Entity<Sidecar>);

impl Global for SidecarGlobal {}

impl SidecarGlobal {
    pub fn get(cx: &App) -> Entity<Sidecar> {
        cx.global::<SidecarGlobal>().0.clone()
    }
}

impl Sidecar {
    pub fn init(cx: &mut App) -> Entity<Sidecar> {
        let entity = cx.new(Sidecar::new);
        cx.set_global(SidecarGlobal(entity.clone()));
        let weak = entity.downgrade();
        cx.on_app_quit(move |cx| {
            _ = weak.update(cx, |this, _| this.shutdown());
            async {}
        })
        .detach();
        entity
    }

    fn new(cx: &mut Context<Self>) -> Self {
        let poll = cx.spawn(async move |this, cx| {
            loop {
                let Ok(should_check) = this.update(cx, |this, _| !this.checking) else {
                    return;
                };
                if should_check {
                    Self::check_now(this.clone(), cx).await;
                }
                cx.background_executor().timer(POLL).await;
            }
        });
        Self {
            status: SidecarStatus::Checking,
            child: None,
            started_at: None,
            checking: false,
            _poll: poll,
        }
    }

    async fn check_now(this: gpui_kit::WeakEntity<Self>, cx: &mut gpui_kit::AsyncApp) {
        let Ok(client) = this.update(cx, |this, cx| {
            this.checking = true;
            SidecarClient::from_settings(cx)
        }) else {
            return;
        };
        let result = cx.background_spawn(async move { client.health() }).await;
        _ = this.update(cx, |this, cx| {
            this.checking = false;
            this.apply_health(result, cx);
        });
    }

    fn apply_health(&mut self, result: anyhow::Result<Health>, cx: &mut Context<Self>) {
        let next = match result {
            Ok(health) => {
                self.started_at = None;
                SidecarStatus::Online(health)
            }
            Err(err) => {
                let child_alive = self
                    .child
                    .as_mut()
                    .is_some_and(|child| matches!(child.try_wait(), Ok(None)));
                if child_alive
                    && self
                        .started_at
                        .is_some_and(|started| started.elapsed() < START_TIMEOUT)
                {
                    SidecarStatus::Starting
                } else if self.child.is_none() && AppSettings::get(cx).sidecar_autostart {
                    match self.spawn(cx) {
                        Ok(()) => SidecarStatus::Starting,
                        Err(spawn_err) => SidecarStatus::Offline(format!("{spawn_err:#}")),
                    }
                } else {
                    if !child_alive {
                        self.child = None;
                    }
                    SidecarStatus::Offline(format!("{err:#}"))
                }
            }
        };
        if next != self.status {
            self.status = next;
            cx.notify();
        }
    }

    fn spawn(&mut self, cx: &App) -> anyhow::Result<()> {
        let project = AppSettings::get(cx).sidecar_project.clone();
        if !project.join("pyproject.toml").exists() {
            anyhow::bail!("پروژهٔ سایدکار پیدا نشد: {}", project.display());
        }
        let log_path = data_dir().join("sidecar.log");
        std::fs::create_dir_all(data_dir())?;
        let log = std::fs::File::create(&log_path)?;
        let mut command = Command::new("uv");
        // An installed (root-owned) project can't hold `.venv`; keep it in the
        // data dir and use the shipped lock as is.
        if !is_writable(&project) && std::env::var_os("UV_PROJECT_ENVIRONMENT").is_none() {
            command
                .env("UV_PROJECT_ENVIRONMENT", data_dir().join("sidecar-venv"))
                .env("UV_FROZEN", "1");
        }
        let child = command
            .arg("run")
            .arg("--project")
            .arg(&project)
            .arg("seda-sidecar")
            .current_dir(&project)
            .stdin(Stdio::null())
            .stdout(log.try_clone()?)
            .stderr(log)
            // Own process group, so shutdown can stop uv and its python child together.
            .process_group(0)
            .spawn()?;
        CHILD_PGID.store(child.id(), std::sync::atomic::Ordering::SeqCst);
        self.child = Some(child);
        self.started_at = Some(Instant::now());
        Ok(())
    }

    pub fn status(&self) -> &SidecarStatus {
        &self.status
    }

    pub fn health(&self) -> Option<&Health> {
        match &self.status {
            SidecarStatus::Online(health) => Some(health),
            _ => None,
        }
    }

    pub fn is_online(&self) -> bool {
        matches!(self.status, SidecarStatus::Online(_))
    }

    /// Whether this app started the sidecar process.
    pub fn owns_process(&self) -> bool {
        self.child.is_some()
    }

    /// Check health now; when offline and autostart is on, (re)start it.
    pub fn retry(&mut self, cx: &mut Context<Self>) {
        if self.checking {
            return;
        }
        if let Some(child) = self.child.as_mut()
            && !matches!(child.try_wait(), Ok(None))
        {
            self.child = None;
        }
        if matches!(self.status, SidecarStatus::Offline(_)) {
            self.status = SidecarStatus::Checking;
            cx.notify();
        }
        cx.spawn(async move |this, cx| Self::check_now(this, cx).await)
            .detach();
    }

    /// Kill the child we started, if any.
    pub fn shutdown(&mut self) {
        if let Some(mut child) = self.child.take() {
            CHILD_PGID.store(0, std::sync::atomic::Ordering::SeqCst);
            let pid = child.id();
            _ = Command::new("kill")
                .args(["-TERM", &format!("-{pid}")])
                .status();
            for _ in 0..20 {
                if matches!(child.try_wait(), Ok(Some(_))) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            _ = Command::new("kill")
                .args(["-KILL", &format!("-{pid}")])
                .status();
            _ = child.wait();
        }
    }
}

impl Drop for Sidecar {
    fn drop(&mut self) {
        self.shutdown();
    }
}
