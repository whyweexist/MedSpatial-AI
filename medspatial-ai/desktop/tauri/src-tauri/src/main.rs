// MedSpatial AI — Tauri Application Entry Point
// ===============================================
// Orchestrates:
//   - Backend sidecar launch (port selection, token generation, env injection)
//   - Health polling before the window becomes interactive
//   - WebView2 config injection (URL + token)
//   - Background health monitor (auto-restart on crash)
//   - Graceful shutdown on window close

// Prevents a second terminal window from appearing on Windows in release builds
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod backend;

use std::sync::{Arc, Mutex};
use tauri::{Manager, WindowEvent};

fn main() {
    // Initialise logger — in production this goes to stderr which Tauri
    // captures; in debug it goes to the console.
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info"),
    )
    .init();

    tauri::Builder::default()
        // ── Setup: runs before the first window is created ────────────
        .setup(|app| {
            let handle = app.handle().clone();

            // 1. Pick a free port
            let port  = backend::find_free_port(8765);
            // 2. Generate session token
            let token = backend::generate_token();

            log::info!("Selected port  : {port}");
            log::info!("Token generated: {}…", &token[..8]);

            // 3. Launch backend
            let child = backend::launch_backend(app.handle(), port, &token);
            let process_arc = Arc::new(Mutex::new(Some(child)));

            // 4. Store state so the shutdown handler can reach it
            app.manage(backend::BackendState {
                port,
                token: token.clone(),
                process: Arc::clone(&process_arc),
            });

            // 5. Wait for healthy in a thread so we don't block the UI thread
            //    (Tauri setup() must return quickly)
            let handle2      = handle.clone();
            let token_clone  = token.clone();
            let arc_clone    = Arc::clone(&process_arc);

            std::thread::spawn(move || {
                match backend::wait_until_healthy(port) {
                    Ok(()) => {
                        // Inject config into WebView2
                        backend::inject_config_into_webview(&handle2, port, &token_clone);

                        // Start background health monitor
                        backend::start_health_monitor(
                            handle2.clone(),
                            port,
                            token_clone,
                            arc_clone,
                        );
                    }
                    Err(e) => {
                        log::error!("Backend startup failed: {e}");
                        // Show a dialog to the user
                        if let Some(window) = handle2.get_webview_window("main") {
                            let _ = window.eval(&format!(
                                r#"window.__MEDSPATIAL_BACKEND_ERROR__ = "{e}";"#
                            ));
                        }
                    }
                }
            });

            Ok(())
        })
        // ── Window events ─────────────────────────────────────────────
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { .. } = event {
                log::info!("Window close requested — shutting down backend…");
                let state = window.state::<backend::BackendState>();
                backend::shutdown_backend(&state);
            }
        })
        // ── Tauri commands (optional — add custom Rust ↔ JS commands here)
        .invoke_handler(tauri::generate_handler![
            cmd_get_backend_url,
            cmd_get_version,
        ])
        .run(tauri::generate_context!())
        .expect("error while running MedSpatial AI");
}

// ---------------------------------------------------------------------------
// Tauri commands — callable from JavaScript via invoke()
// ---------------------------------------------------------------------------

/// Returns the full backend base URL (e.g. http://127.0.0.1:8765).
/// The frontend can call this as a fallback if window.__MEDSPATIAL_BACKEND_URL__
/// is not yet set when the bundle first evaluates.
#[tauri::command]
fn cmd_get_backend_url(state: tauri::State<backend::BackendState>) -> String {
    format!("http://127.0.0.1:{}", state.port)
}

/// Returns the app version string from Cargo.toml.
#[tauri::command]
fn cmd_get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
