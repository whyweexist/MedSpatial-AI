/**
 * MedSpatial AI — Tauri API Client
 * =================================
 * Production-ready replacement for frontend/src/services/api.js.
 *
 * Changes vs dev api.js:
 *  - Imports from tauri_bridge (pre-authenticated axios instance + BACKEND_URL)
 *  - WebSocket URL derived from BACKEND_URL instead of hardcoded localhost:8000
 *  - Multipart upload passes the auth token explicitly (axios interceptors
 *    don't apply to FormData content-type in all browsers)
 *  - All existing function signatures are preserved 1-to-1 so App.jsx and
 *    all components need zero changes.
 *
 * Copy this file to the Tauri src/ folder and rename it api.js, OR update
 * the import path in App.jsx to point here.
 */

import apiClient, { BACKEND_URL, API_TOKEN } from "./tauri_bridge";

// ---------------------------------------------------------------------------
// Auth header helper for multipart requests
// ---------------------------------------------------------------------------
function authHeaders() {
  return API_TOKEN ? { "X-MedSpatial-Token": API_TOKEN } : {};
}

// ---------------------------------------------------------------------------
// Scans
// ---------------------------------------------------------------------------
export async function uploadScan(formData) {
  const res = await apiClient.post("/api/scans/upload", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
      ...authHeaders(),
    },
    timeout: 300_000,  // 5 min for large DICOM series
  });
  return res.data;
}

export async function listScans() {
  const res = await apiClient.get("/api/scans");
  return res.data;
}

export async function getScan(scanId) {
  const res = await apiClient.get(`/api/scans/${scanId}`);
  return res.data;
}

export async function deleteScan(scanId) {
  const res = await apiClient.delete(`/api/scans/${scanId}`);
  return res.data;
}

// ---------------------------------------------------------------------------
// Reconstruction
// ---------------------------------------------------------------------------
export async function startReconstruction(scanId, options = {}) {
  const res = await apiClient.post(`/api/reconstruction/build/${scanId}`, options);
  return res.data;
}

export async function getReconstructionStatus(scanId) {
  const res = await apiClient.get(`/api/reconstruction/status/${scanId}`);
  return res.data;
}

export async function getMesh(scanId) {
  const res = await apiClient.get(`/api/reconstruction/mesh/${scanId}`);
  return res.data;
}

export async function getSegments(scanId) {
  const res = await apiClient.get(`/api/reconstruction/segments/${scanId}`);
  return res.data;
}

export async function getLabels(scanId) {
  const res = await apiClient.get(`/api/reconstruction/labels/${scanId}`);
  return res.data;
}

export async function getSlice(scanId, axis, index) {
  const res = await apiClient.get(`/api/reconstruction/slice/${scanId}`, {
    params: { axis, index },
    responseType: "blob",
  });
  return URL.createObjectURL(res.data);
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------
export async function runAnalysis(scanId, mode = "full") {
  const res = await apiClient.post(`/api/analysis/run/${scanId}`, { mode });
  return res.data;
}

export async function getAnalysisResults(scanId) {
  const res = await apiClient.get(`/api/analysis/results/${scanId}`);
  return res.data;
}

export async function getHeatmap(scanId, analysisId) {
  const res = await apiClient.get(`/api/analysis/heatmap/${scanId}`, {
    params: { analysis_id: analysisId },
    responseType: "blob",
  });
  return URL.createObjectURL(res.data);
}

export async function getSegmentation(scanId, analysisId) {
  const res = await apiClient.get(`/api/analysis/segmentation/${scanId}`, {
    params: { analysis_id: analysisId },
  });
  return res.data;
}

// ---------------------------------------------------------------------------
// Explainability / XAI
// ---------------------------------------------------------------------------
export async function explainFinding(scanId, findingIndex) {
  const res = await apiClient.post(`/api/analysis/explain`, {
    scan_id: scanId,
    finding_index: findingIndex,
  });
  return res.data;
}

// ---------------------------------------------------------------------------
// Chat / Medical QA
// ---------------------------------------------------------------------------
export async function askChat(scanId, question, sessionId = null) {
  const res = await apiClient.post("/api/chat/ask", {
    scan_id: scanId,
    question,
    session_id: sessionId,
  });
  return res.data;
}

export async function getChatHistory(sessionId) {
  const res = await apiClient.get(`/api/chat/history/${sessionId}`);
  return res.data;
}

// ---------------------------------------------------------------------------
// Reports
// ---------------------------------------------------------------------------
export async function generateReport(scanId, format = "pdf") {
  const res = await apiClient.post(
    `/api/reports/generate/${scanId}`,
    { format },
    { responseType: "blob" },
  );
  const ext  = format === "pdf" ? "pdf" : "docx";
  const url  = URL.createObjectURL(res.data);
  const link = document.createElement("a");
  link.href     = url;
  link.download = `medspatial_report_${scanId}.${ext}`;
  link.click();
  URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------------
// WebSocket factory — Tauri-aware
// ---------------------------------------------------------------------------
export function createScanWebSocket(scanId, onMessage, onError) {
  // Derive ws:// URL from the HTTP base URL
  const wsBase = BACKEND_URL.replace(/^http/, "ws");
  const ws = new WebSocket(`${wsBase}/ws/${scanId}`);

  ws.onopen    = () => console.info(`[WS] Connected for scan ${scanId}`);
  ws.onmessage = (event) => {
    try {
      onMessage(JSON.parse(event.data));
    } catch {
      onMessage({ raw: event.data });
    }
  };
  ws.onerror   = (err) => {
    console.error("[WS] Error:", err);
    onError?.(err);
  };
  ws.onclose   = () => console.info(`[WS] Closed for scan ${scanId}`);

  return ws;
}

// ---------------------------------------------------------------------------
// Health check (used by loading screen)
// ---------------------------------------------------------------------------
export async function checkBackendHealth() {
  try {
    // Health endpoint is auth-exempt
    const res = await apiClient.get("/api/health", { timeout: 3000 });
    return res.status === 200 && res.data?.status === "healthy";
  } catch {
    return false;
  }
}
