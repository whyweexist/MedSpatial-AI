/**
 * MedSpatial AI — API Service (Enhanced)
 * Centralized API client for all backend communication including
 * segments, XAI, reports, labels, and body region endpoints.
 */

import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 300000,  // Increased to 5 minutes for analysis/report generation
});

// ── Scans ────────────────────────────────────────────────────

export async function uploadDicomFiles(files, onProgress) {
  const formData = new FormData();
  files.forEach(file => formData.append('files', file));

  const response = await api.post('/scans/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => {
      if (onProgress && e.total) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    },
  });
  return response.data;
}

export async function listScans() {
  const response = await api.get('/scans/');
  return response.data;
}

export async function getScan(scanId) {
  const response = await api.get(`/scans/${scanId}`);
  return response.data;
}

export async function deleteScan(scanId) {
  const response = await api.delete(`/scans/${scanId}`);
  return response.data;
}

// ── Reconstruction ───────────────────────────────────────────

export async function startReconstruction(scanId, options = {}) {
  const response = await api.post('/reconstruction/build', {
    scan_id: scanId,
    iso_level: options.isoLevel || null,
    step_size: options.stepSize || null,
    generate_layers: options.generateLayers !== false,
  });
  return response.data;
}

export async function getReconstructionStatus(scanId) {
  const response = await api.get(`/reconstruction/status/${scanId}`);
  return response.data;
}

export function getMeshUrl(scanId, layer = 'primary') {
  return `/api/reconstruction/mesh/${scanId}/${layer}`;
}

export async function getSlice(scanId, axis = 'axial', index = 0) {
  const response = await api.post('/reconstruction/slice', {
    scan_id: scanId,
    axis,
    index,
  });
  return response.data;
}

// ── Segments / Dissection ────────────────────────────────────

export async function getSegments(scanId) {
  const response = await api.get(`/reconstruction/segments/${scanId}`);
  return response.data;
}

export async function getAnatomyLabels(scanId) {
  const response = await api.get(`/reconstruction/labels/${scanId}`);
  return response.data;
}

// ── Analysis ─────────────────────────────────────────────────

export async function runAnalysis(scanId, analysisType = 'full') {
  const response = await api.post('/analysis/run', {
    scan_id: scanId,
    analysis_type: analysisType,
  });
  return response.data;
}

export async function getAnalysisResults(scanId) {
  const response = await api.get(`/analysis/results/${scanId}`);
  return response.data;
}

// ── XAI / Explainability ─────────────────────────────────────

export async function explainScan(scanId) {
  const response = await api.post(`/analysis/explain/${scanId}`);
  return response.data;
}

export async function getXAIHeatmap(scanId, diseaseClass) {
  const response = await api.get(`/analysis/explain/${scanId}/heatmap/${diseaseClass}`, {
    responseType: 'arraybuffer',
  });
  return response.data;
}

export async function getReasoning(scanId) {
  const response = await api.get(`/analysis/explain/${scanId}/reasoning`);
  return response.data;
}

// ── Reports ──────────────────────────────────────────────────

export async function downloadReport(scanId, format = 'pdf') {
  const response = await api.get(`/reports/generate/${scanId}`, {
    params: { format },
    responseType: 'blob',
  });

  // Trigger browser download
  const blob = new Blob([response.data]);
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `MedSpatial_Report_${scanId.substring(0, 8)}.${format}`;
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(url);
  a.remove();
}

// ── Chat ─────────────────────────────────────────────────────

export async function sendChatMessage(scanId, message, sessionId = null) {
  const response = await api.post('/chat/ask', {
    scan_id: scanId,
    session_id: sessionId,
    message,
  });
  return response.data;
}

export async function getChatHistory(sessionId) {
  const response = await api.get(`/chat/history/${sessionId}`);
  return response.data;
}

// ── Health ───────────────────────────────────────────────────

export async function healthCheck() {
  const response = await api.get('/health');
  return response.data;
}

// ── WebSocket ────────────────────────────────────────────────

export function createWebSocket(scanId, onMessage) {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${protocol}//${window.location.host}/ws/${scanId}`);

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch (e) {
      console.warn('WS parse error:', e);
    }
  };

  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
  };

  return ws;
}

export default api;
