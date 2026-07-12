/**
 * MedSpatial AI — Tauri Bridge
 * ============================
 * Thin adapter that sits between the existing axios-based api.js and Tauri's
 * runtime environment. It does three things:
 *
 *  1. Resolves the backend base URL at runtime (read from the Tauri window
 *     title-bar metadata injected by Rust, or falls back to a safe default).
 *  2. Provides the per-session auth token that Tauri's Rust core generates
 *     and injects via window.__MEDSPATIAL_TOKEN__.
 *  3. Exports an axios instance pre-configured with the correct base URL and
 *     auth header — imported by tauri_api.js instead of the raw axios.
 *
 * In Codespaces / dev the bridge transparently falls through to
 * http://127.0.0.1:8765 so no code path changes between environments.
 */

import axios from "axios";

// ---------------------------------------------------------------------------
// Runtime config injected by Tauri's Rust core into the WebView2 window object
// before the React bundle loads. See backend.rs → inject_config().
// ---------------------------------------------------------------------------
function resolveBackendUrl() {
  if (typeof window !== "undefined" && window.__MEDSPATIAL_BACKEND_URL__) {
    return window.__MEDSPATIAL_BACKEND_URL__;
  }
  // Dev / Codespaces fallback
  const port = import.meta.env.VITE_BACKEND_PORT || "8765";
  return `http://127.0.0.1:${port}`;
}

function resolveToken() {
  if (typeof window !== "undefined" && window.__MEDSPATIAL_TOKEN__) {
    return window.__MEDSPATIAL_TOKEN__;
  }
  // Dev: no token enforced
  return import.meta.env.VITE_API_TOKEN || "";
}

export const BACKEND_URL = resolveBackendUrl();
export const API_TOKEN   = resolveToken();

// ---------------------------------------------------------------------------
// Pre-configured axios instance — import this instead of raw axios everywhere
// ---------------------------------------------------------------------------
const apiClient = axios.create({
  baseURL: BACKEND_URL,
  timeout: 120_000,   // 2 min — 3-D reconstruction can be slow on low-end PCs
  headers: {
    "Content-Type": "application/json",
    ...(API_TOKEN ? { "X-MedSpatial-Token": API_TOKEN } : {}),
  },
});

// Response interceptor: uniform error shape
apiClient.interceptors.response.use(
  (res) => res,
  (err) => {
    const status  = err.response?.status;
    const detail  = err.response?.data?.detail || err.message;
    console.error(`[API] ${status ?? "network"} — ${detail}`);
    return Promise.reject(err);
  }
);

export default apiClient;
