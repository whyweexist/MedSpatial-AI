// MedSpatial AI — Backend Sidecar Manager
// =========================================
// Responsible for the full lifecycle of the FastAPI backend subprocess:
//
//  1. Port selection  — tries port 8765; if occupied, scans 8766-8800
//  2. Token generation — 32-byte cryptographically random hex string
//  3. Sidecar launch  — spawns medspatial_backend.exe with env vars
//  4. Health polling  — GET /api/health every 500ms until 200 or timeout
//  5. Config injection — writes port + token into WebView2 via JS eval
//  6. Shutdown        — kills the process tree on app exit
//
// The backend binary is expected at:
//   <resources>/medspatial_backend.exe   (NSIS installer places it here)
//   OR next to the .exe during dev testing

use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use rand::Rng;
use tauri::{AppHandle, Manager, Runtime};

// ---------------------------------------------------------------------------
// Public state stored in Tauri's managed state
// ---------------------------------------------------------------------------

pub struct BackendState {
    pub port:    u16,
    pub token:   String,
    pub process: Arc<Mutex<Option<Child>>>,
}

// ---------------------------------------------------------------------------
// Port selection
// ---------------------------------------------------------------------------

/// Try to bind a TCP listener to find a free port.
/// Starts at preferred_port and scans up to preferred_port + 35.
fn find_free_port(preferred: u16) -> u16 {
    for p in preferred..preferred + 35 {
        if TcpListener::bind(format!("127.0.0.1:{p}")).is_ok() {
            return p;
        }
    }
    panic!("No free port found in range {}–{}", preferred, preferred + 35);
}

// ---------------------------------------------------------------------------
// Token generation
// ---------------------------------------------------------------------------

fn generate_token() -> String {
    let bytes: Vec<u8> = (0..32).map(|_| rand::thread_rng().gen::<u8>()).collect();
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------
// Resolve the sidecar binary path
// ---------------------------------------------------------------------------

fn resolve_backend_binary<R: Runtime>(app: &AppHandle<R>) -> PathBuf {
    // 1. Check Tauri's resource directory (production install)
    if let Ok(res_path) = app.path().resource_dir() {
        let bin = res_path.join("medspatial_backend.exe");
        if bin.exists() {
            return bin;
        }
    }

    // 2. Check next to the main executable (dev / portable run)
    if let Ok(exe_path) = std::env::current_exe() {
        let bin = exe_path.parent().unwrap_or(&exe_path).join("medspatial_backend.exe");
        if bin.exists() {
            return bin;
        }
    }

    // 3. Absolute path from env var (CI / testing override)
    if let Ok(env_path) = std::env::var("MEDSPATIAL_BACKEND_BIN") {
        let bin = PathBuf::from(env_path);
        if bin.exists() {
            return bin;
        }
    }

    panic!(
        "medspatial_backend.exe not found. \
         Expected in the resources/ directory next to the installer output."
    );
}

// ---------------------------------------------------------------------------
// Launch
// ---------------------------------------------------------------------------

/// Spawn the backend sidecar process.
/// Returns the Child handle so we can kill it on shutdown.
pub fn launch_backend<R: Runtime>(
    app: &AppHandle<R>,
    port: u16,
    token: &str,
) -> Child {
    let binary = resolve_backend_binary(app);
    let exe_dir = binary
        .parent()
        .expect("binary must have a parent dir")
        .to_string_lossy()
        .into_owned();

    log::info!("Launching backend: {}", binary.display());
    log::info!("  Port  : {port}");
    log::info!("  ExeDir: {exe_dir}");

    let child = Command::new(&binary)
        .env("MEDSPATIAL_PORT",    port.to_string())
        .env("MEDSPATIAL_TOKEN",   token)
        .env("MEDSPATIAL_EXE_DIR", &exe_dir)
        // Suppress console window on Windows
        .creation_flags(0x08000000)  // CREATE_NO_WINDOW
        .spawn()
        .unwrap_or_else(|e| panic!("Failed to spawn backend process: {e}"));

    log::info!("Backend process PID: {}", child.id());
    child
}

// ---------------------------------------------------------------------------
// Health polling
// ---------------------------------------------------------------------------

const HEALTH_POLL_MS:  u64 = 500;
const HEALTH_TIMEOUT_S: u64 = 60;

/// Block until /api/health returns 200 or the timeout expires.
/// Returns Ok(()) if healthy, Err(String) on timeout.
pub fn wait_until_healthy(port: u16) -> Result<(), String> {
    let url     = format!("http://127.0.0.1:{port}/api/health");
    let timeout = Duration::from_secs(HEALTH_TIMEOUT_S);
    let start   = Instant::now();

    log::info!("Waiting for backend health at {url}…");

    loop {
        if start.elapsed() > timeout {
            return Err(format!(
                "Backend did not become healthy within {HEALTH_TIMEOUT_S}s"
            ));
        }

        match ureq::get(&url)
            .timeout(Duration::from_secs(2))
            .call()
        {
            Ok(resp) if resp.status() == 200 => {
                log::info!("Backend healthy after {:.1}s", start.elapsed().as_secs_f32());
                return Ok(());
            }
            Ok(resp) => {
                log::debug!("Health check: HTTP {}", resp.status());
            }
            Err(e) => {
                log::debug!("Health check pending: {e}");
            }
        }

        thread::sleep(Duration::from_millis(HEALTH_POLL_MS));
    }
}

// ---------------------------------------------------------------------------
// Inject config into WebView2
// ---------------------------------------------------------------------------

/// Write the backend URL and auth token into the WebView2 window object so
/// the React bundle can read them without any hardcoded values.
pub fn inject_config_into_webview<R: Runtime>(
    app: &AppHandle<R>,
    port: u16,
    token: &str,
) {
    if let Some(window) = app.get_webview_window("main") {
        let script = format!(
            r#"
            window.__MEDSPATIAL_BACKEND_URL__ = "http://127.0.0.1:{port}";
            window.__MEDSPATIAL_TOKEN__ = "{token}";
            "#
        );
        window
            .eval(&script)
            .unwrap_or_else(|e| log::warn!("WebView eval failed: {e}"));
        log::info!("Config injected into WebView2 (port={port})");
    } else {
        log::warn!("WebView window 'main' not found — config not injected");
    }
}

// ---------------------------------------------------------------------------
// Graceful shutdown
// ---------------------------------------------------------------------------

/// Kill the backend process and its entire process tree.
/// Called from the on_window_event handler in main.rs.
pub fn shutdown_backend(state: &BackendState) {
    let mut guard = state.process.lock().unwrap();
    if let Some(mut child) = guard.take() {
        let pid = child.id();
        log::info!("Shutting down backend PID {pid}…");

        // On Windows, kill() sends SIGTERM-equivalent.
        // We also use taskkill /F /T to ensure child processes are killed.
        #[cfg(target_os = "windows")]
        {
            let _ = Command::new("taskkill")
                .args(["/F", "/T", "/PID", &pid.to_string()])
                .output();
        }

        #[cfg(not(target_os = "windows"))]
        {
            let _ = child.kill();
        }

        log::info!("Backend process terminated.");
    }
}

// ---------------------------------------------------------------------------
// Background health monitor (optional — restarts crashed backend)
// ---------------------------------------------------------------------------

/// Spawn a background thread that monitors backend health every 5 seconds.
/// If the backend crashes, it restarts it automatically.
/// This is a best-effort guard for production stability.
pub fn start_health_monitor<R: Runtime + 'static>(
    app: AppHandle<R>,
    port: u16,
    token: String,
    process_arc: Arc<Mutex<Option<Child>>>,
) {
    thread::spawn(move || {
        let check_url = format!("http://127.0.0.1:{port}/api/health");
        let mut consecutive_failures: u32 = 0;

        // Give the backend 10 s to fully warm up before monitoring starts
        thread::sleep(Duration::from_secs(10));

        loop {
            thread::sleep(Duration::from_secs(5));

            let ok = ureq::get(&check_url)
                .timeout(Duration::from_secs(3))
                .call()
                .map(|r| r.status() == 200)
                .unwrap_or(false);

            if ok {
                consecutive_failures = 0;
            } else {
                consecutive_failures += 1;
                log::warn!(
                    "Backend health check failed ({consecutive_failures}/3)…"
                );

                if consecutive_failures >= 3 {
                    log::error!("Backend unresponsive — attempting restart…");
                    consecutive_failures = 0;

                    // Kill the stale process
                    {
                        let mut guard = process_arc.lock().unwrap();
                        if let Some(mut child) = guard.take() {
                            #[cfg(target_os = "windows")]
                            let _ = Command::new("taskkill")
                                .args(["/F", "/T", "/PID", &child.id().to_string()])
                                .output();
                            #[cfg(not(target_os = "windows"))]
                            let _ = child.kill();
                        }
                    }

                    // Re-launch
                    let new_child = launch_backend(&app, port, &token);
                    {
                        let mut guard = process_arc.lock().unwrap();
                        *guard = Some(new_child);
                    }

                    // Wait for it to be healthy again
                    if let Err(e) = wait_until_healthy(port) {
                        log::error!("Restarted backend still unhealthy: {e}");
                    } else {
                        log::info!("Backend restarted and healthy.");
                    }
                }
            }
        }
    });
}
