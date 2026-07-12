/**
 * MedSpatial AI — Loading Screen
 * ================================
 * Shown while the Tauri Rust core is starting the FastAPI backend sidecar
 * and waiting for it to become healthy.
 *
 * Lifecycle:
 *   1. App mounts → LoadingScreen appears immediately
 *   2. LoadingScreen polls GET /api/health every 600 ms
 *   3. Once healthy, it calls onReady() and fades out
 *   4. If health check fails for > 30 s, shows an error with a retry button
 *
 * The component reads window.__MEDSPATIAL_BACKEND_URL__ set by Rust so it
 * knows which port to poll — no hardcoded values.
 */

import React, { useEffect, useRef, useState } from "react";
import { checkBackendHealth } from "./tauri_api";

const POLL_INTERVAL_MS  = 600;
const TIMEOUT_MS        = 45_000;

// ── Inline styles — no CSS file dependency ──────────────────────────────────
const S = {
  overlay: {
    position: "fixed",
    inset: 0,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    background: "linear-gradient(135deg, #0d1117 0%, #161b22 60%, #0d1117 100%)",
    zIndex: 9999,
    transition: "opacity 0.4s ease",
  },
  logo: {
    fontSize: 48,
    marginBottom: 24,
    filter: "drop-shadow(0 0 24px #3b82f6aa)",
  },
  title: {
    fontSize: 28,
    fontWeight: 700,
    color: "#e6edf3",
    letterSpacing: "0.04em",
    marginBottom: 8,
    fontFamily: "system-ui, -apple-system, sans-serif",
  },
  subtitle: {
    fontSize: 14,
    color: "#8b949e",
    marginBottom: 40,
    fontFamily: "system-ui, -apple-system, sans-serif",
  },
  spinnerWrapper: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 16,
  },
  spinner: {
    width: 48,
    height: 48,
    border: "4px solid #30363d",
    borderTop: "4px solid #3b82f6",
    borderRadius: "50%",
    animation: "medspatial-spin 0.9s linear infinite",
  },
  statusText: {
    fontSize: 13,
    color: "#8b949e",
    fontFamily: "monospace",
    minHeight: 20,
  },
  progressBar: {
    width: 280,
    height: 3,
    background: "#21262d",
    borderRadius: 2,
    overflow: "hidden",
    marginTop: 16,
  },
  progressFill: {
    height: "100%",
    background: "linear-gradient(90deg, #3b82f6, #60a5fa)",
    borderRadius: 2,
    transition: "width 0.3s ease",
  },
  errorBox: {
    maxWidth: 400,
    padding: "20px 24px",
    background: "#161b22",
    border: "1px solid #f8514944",
    borderRadius: 8,
    textAlign: "center",
  },
  errorTitle: {
    color: "#f85149",
    fontSize: 16,
    fontWeight: 600,
    marginBottom: 8,
    fontFamily: "system-ui, sans-serif",
  },
  errorMsg: {
    color: "#8b949e",
    fontSize: 13,
    marginBottom: 20,
    lineHeight: 1.6,
    fontFamily: "system-ui, sans-serif",
  },
  retryBtn: {
    padding: "8px 24px",
    background: "#3b82f6",
    color: "#fff",
    border: "none",
    borderRadius: 6,
    fontSize: 14,
    cursor: "pointer",
    fontFamily: "system-ui, sans-serif",
  },
};

// Inject keyframes once
if (typeof document !== "undefined") {
  const styleId = "medspatial-loading-keyframes";
  if (!document.getElementById(styleId)) {
    const style = document.createElement("style");
    style.id = styleId;
    style.textContent = `@keyframes medspatial-spin { to { transform: rotate(360deg); } }`;
    document.head.appendChild(style);
  }
}

// ── Status messages shown while waiting ──────────────────────────────────────
const STATUS_STEPS = [
  "Initialising AI engine…",
  "Loading ONNX models…",
  "Preparing database…",
  "Warming up inference sessions…",
  "Almost ready…",
];

export default function LoadingScreen({ onReady }) {
  const [visible,    setVisible]    = useState(true);
  const [fading,     setFading]     = useState(false);
  const [statusIdx,  setStatusIdx]  = useState(0);
  const [progress,   setProgress]   = useState(0);
  const [isError,    setIsError]    = useState(false);
  const [retryCount, setRetryCount] = useState(0);

  const startTime   = useRef(Date.now());
  const pollRef     = useRef(null);
  const stepRef     = useRef(null);
  const isMounted   = useRef(true);

  const startPolling = () => {
    startTime.current = Date.now();
    setIsError(false);
    setProgress(0);
    setStatusIdx(0);

    pollRef.current = setInterval(async () => {
      if (!isMounted.current) return;

      const elapsed = Date.now() - startTime.current;
      const pct = Math.min(95, (elapsed / TIMEOUT_MS) * 100);
      setProgress(pct);

      if (elapsed > TIMEOUT_MS) {
        clearInterval(pollRef.current);
        setIsError(true);
        return;
      }

      const healthy = await checkBackendHealth();
      if (healthy && isMounted.current) {
        clearInterval(pollRef.current);
        clearInterval(stepRef.current);
        setProgress(100);

        // Brief pause so user sees 100% then fade
        setTimeout(() => {
          if (!isMounted.current) return;
          setFading(true);
          setTimeout(() => {
            if (!isMounted.current) return;
            setVisible(false);
            onReady?.();
          }, 420);
        }, 300);
      }
    }, POLL_INTERVAL_MS);

    // Cycle status messages independently of health poll
    stepRef.current = setInterval(() => {
      if (!isMounted.current) return;
      setStatusIdx((i) => (i + 1) % STATUS_STEPS.length);
    }, 2800);
  };

  useEffect(() => {
    isMounted.current = true;
    startPolling();
    return () => {
      isMounted.current = false;
      clearInterval(pollRef.current);
      clearInterval(stepRef.current);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [retryCount]);

  if (!visible) return null;

  return (
    <div
      style={{
        ...S.overlay,
        opacity: fading ? 0 : 1,
        pointerEvents: fading ? "none" : "auto",
      }}
      role="status"
      aria-live="polite"
      aria-label="Loading MedSpatial AI"
    >
      <div style={S.logo} aria-hidden="true">🧠</div>
      <div style={S.title}>MedSpatial AI</div>
      <div style={S.subtitle}>Medical Imaging Intelligence Platform</div>

      {isError ? (
        <div style={S.errorBox} role="alert">
          <div style={S.errorTitle}>Backend failed to start</div>
          <div style={S.errorMsg}>
            The AI backend did not respond within {TIMEOUT_MS / 1000} seconds.
            This can happen on very slow hardware during the first run
            (ONNX models compiling kernels).
            <br /><br />
            Check <code>%APPDATA%\MedSpatialAI\logs\backend.log</code> for details.
          </div>
          <button
            style={S.retryBtn}
            onClick={() => setRetryCount((c) => c + 1)}
          >
            Retry
          </button>
        </div>
      ) : (
        <div style={S.spinnerWrapper}>
          <div style={S.spinner} aria-hidden="true" />
          <div style={S.statusText}>{STATUS_STEPS[statusIdx]}</div>
          <div style={S.progressBar} role="progressbar" aria-valuenow={Math.round(progress)} aria-valuemin={0} aria-valuemax={100}>
            <div style={{ ...S.progressFill, width: `${progress}%` }} />
          </div>
        </div>
      )}
    </div>
  );
}
