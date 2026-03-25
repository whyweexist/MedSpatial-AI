/**
 * MedSpatial AI — Main Application
 * Root layout component managing global state and three-panel layout.
 */

import React, { useState, useCallback, useEffect } from 'react';
import Header from './components/Header';
import UploadPanel from './components/UploadPanel';
import LayerControls from './components/LayerControls';
import SliceViewer from './components/SliceViewer';
import Viewer3D from './components/Viewer3D';
import ChatPanel from './components/ChatPanel';
import AnomalyOverlay from './components/AnomalyOverlay';
import {
  listScans,
  startReconstruction,
  getReconstructionStatus,
  runAnalysis,
  getAnalysisResults,
} from './services/api';

export default function App() {
  // ── Global State ──────────────────────────────────────────
  const [scans, setScans] = useState([]);
  const [activeScan, setActiveScan] = useState(null);
  const [scanStatus, setScanStatus] = useState('idle');

  // Reconstruction state
  const [meshUrl, setMeshUrl] = useState(null);
  const [layerUrls, setLayerUrls] = useState({});
  const [volumeDimensions, setVolumeDimensions] = useState(null);

  // Layer visibility
  const [layers, setLayers] = useState({
    primary: { visible: true, opacity: 0.8, color: '#d4d8e0' },
    bone: { visible: true, opacity: 0.9, color: '#f0ecd8' },
    soft_tissue: { visible: true, opacity: 0.5, color: '#e6b8a2' },
    air: { visible: false, opacity: 0.2, color: '#4466cc' },
    vessel: { visible: true, opacity: 0.7, color: '#cc4444' },
  });

  // Analysis state
  const [analysisResults, setAnalysisResults] = useState(null);
  const [findings, setFindings] = useState([]);
  const [showHeatmap, setShowHeatmap] = useState(false);

  // Chat state
  const [chatSessionId, setChatSessionId] = useState(null);

  // ── Load scans on mount ─────────────────────────────────
  useEffect(() => {
    loadScans();
  }, []);

  const loadScans = async () => {
    try {
      const data = await listScans();
      setScans(data.scans || []);
    } catch (e) {
      console.error('Failed to load scans:', e);
    }
  };

  // ── Scan Selection ────────────────────────────────────────
  const selectScan = useCallback(async (scan) => {
    setActiveScan(scan);
    setChatSessionId(null);
    setFindings([]);
    setAnalysisResults(null);
    setShowHeatmap(false);

    // Check if already reconstructed
    if (scan.status === 'reconstructed' || scan.status === 'analyzed') {
      try {
        const status = await getReconstructionStatus(scan.id);
        if (status.mesh_url) {
          setMeshUrl(status.mesh_url);
          setLayerUrls(status.layer_urls || {});
          setVolumeDimensions(status.dimensions);
        }

        // Load analysis if available
        if (scan.status === 'analyzed') {
          const results = await getAnalysisResults(scan.id);
          if (results.length > 0) {
            setAnalysisResults(results[0]);
            setFindings(results[0].findings || []);
          }
        }
      } catch (e) {
        console.error('Failed to load reconstruction:', e);
      }
    }
  }, []);

  // ── Trigger Reconstruction ────────────────────────────────
  const handleReconstruct = useCallback(async () => {
    if (!activeScan) return;
    setScanStatus('processing');
    try {
      await startReconstruction(activeScan.id);

      // Poll for completion
      const poll = setInterval(async () => {
        try {
          const status = await getReconstructionStatus(activeScan.id);
          if (status.status === 'reconstructed' || status.status === 'analyzed') {
            clearInterval(poll);
            setMeshUrl(status.mesh_url);
            setLayerUrls(status.layer_urls || {});
            setVolumeDimensions(status.dimensions);
            setScanStatus('idle');
            setActiveScan(prev => ({ ...prev, status: status.status }));
            loadScans();
          } else if (status.status === 'failed') {
            clearInterval(poll);
            setScanStatus('error');
          }
        } catch (e) {
          console.error('Poll error:', e);
        }
      }, 2000);
    } catch (e) {
      console.error('Reconstruction failed:', e);
      setScanStatus('error');
    }
  }, [activeScan]);

  // ── Trigger Analysis ──────────────────────────────────────
  const handleAnalyze = useCallback(async () => {
    if (!activeScan) return;
    setScanStatus('processing');
    try {
      const result = await runAnalysis(activeScan.id, 'full');

      // Poll for completion
      const poll = setInterval(async () => {
        try {
          const results = await getAnalysisResults(activeScan.id);
          const latest = results.find(r => r.analysis_id === result.analysis_id);
          if (latest && latest.status === 'completed') {
            clearInterval(poll);
            setAnalysisResults(latest);
            setFindings(latest.findings || []);
            setScanStatus('idle');
            loadScans();
          } else if (latest && latest.status === 'failed') {
            clearInterval(poll);
            setScanStatus('error');
          }
        } catch (e) {
          console.error('Analysis poll error:', e);
        }
      }, 3000);
    } catch (e) {
      console.error('Analysis failed:', e);
      setScanStatus('error');
    }
  }, [activeScan]);

  // ── Layer Toggle ──────────────────────────────────────────
  const toggleLayer = useCallback((layerName) => {
    setLayers(prev => ({
      ...prev,
      [layerName]: { ...prev[layerName], visible: !prev[layerName].visible },
    }));
  }, []);

  const setLayerOpacity = useCallback((layerName, opacity) => {
    setLayers(prev => ({
      ...prev,
      [layerName]: { ...prev[layerName], opacity },
    }));
  }, []);

  // ── Upload Complete ───────────────────────────────────────
  const handleUploadComplete = useCallback((scan) => {
    loadScans();
    selectScan(scan);
  }, [selectScan]);

  return (
    <div className="app-layout">
      <Header scanStatus={scanStatus} activeScan={activeScan} />

      <div className="app-main">
        {/* ── Left Sidebar ──────────────────────────────────── */}
        <div className="sidebar">
          <UploadPanel onUploadComplete={handleUploadComplete} />

          {/* Scan List */}
          <div className="sidebar-section">
            <div className="sidebar-section-title">
              📁 Scans ({scans.length})
            </div>
            {scans.map(scan => (
              <div
                key={scan.id}
                className={`scan-item ${activeScan?.id === scan.id ? 'active' : ''}`}
                onClick={() => selectScan(scan)}
              >
                <div className="scan-icon">
                  {scan.modality === 'CT' ? '🫁' : scan.modality === 'XR' ? '🦴' : '🧠'}
                </div>
                <div className="scan-info">
                  <div className="scan-name">
                    {scan.body_part || scan.series_description || 'Unknown Scan'}
                  </div>
                  <div className="scan-meta">
                    {scan.modality || '—'} · {scan.num_slices} slices ·{' '}
                    <span className={`badge badge-${scan.status === 'analyzed' ? 'success' : scan.status === 'reconstructed' ? 'info' : 'warning'}`}>
                      {scan.status}
                    </span>
                  </div>
                </div>
              </div>
            ))}
            {scans.length === 0 && (
              <div style={{ textAlign: 'center', padding: '16px', color: 'var(--text-muted)', fontSize: 13 }}>
                No scans uploaded yet
              </div>
            )}
          </div>

          {/* Layer Controls */}
          {activeScan && (
            <div className="sidebar-scrollable">
              <LayerControls
                layers={layers}
                layerUrls={layerUrls}
                onToggle={toggleLayer}
                onOpacityChange={setLayerOpacity}
              />
              <SliceViewer scanId={activeScan?.id} volumeDimensions={volumeDimensions} />
            </div>
          )}
        </div>

        {/* ── Center: 3D Viewer ─────────────────────────────── */}
        <div className="viewer-area">
          {activeScan && meshUrl ? (
            <>
              <Viewer3D
                meshUrl={meshUrl}
                layerUrls={layerUrls}
                layers={layers}
                showHeatmap={showHeatmap}
                findings={findings}
              />
              {showHeatmap && analysisResults?.heatmap_url && (
                <AnomalyOverlay
                  findings={findings}
                  analysisResults={analysisResults}
                />
              )}
            </>
          ) : (
            <div className="empty-state">
              <div className="empty-state-icon">🏥</div>
              <div className="empty-state-title">
                {activeScan ? 'Ready to Reconstruct' : 'Upload a DICOM Scan'}
              </div>
              <div className="empty-state-text">
                {activeScan
                  ? 'Click "Build 3D Model" to generate the interactive volumetric visualization.'
                  : 'Drag and drop DICOM files or click the upload area to get started.'}
              </div>
              {activeScan && activeScan.status === 'uploaded' && (
                <button className="btn btn-primary" style={{ marginTop: 20 }} onClick={handleReconstruct}>
                  🔨 Build 3D Model
                </button>
              )}
            </div>
          )}

          {/* Toolbar */}
          {activeScan && (
            <div className="viewer-toolbar">
              {activeScan.status === 'uploaded' && (
                <button className="btn btn-primary btn-sm" onClick={handleReconstruct}>🔨 Reconstruct</button>
              )}
              {(activeScan.status === 'reconstructed' || activeScan.status === 'analyzed') && (
                <>
                  <button className="btn btn-primary btn-sm" onClick={handleAnalyze}>🔬 Analyze</button>
                  <button
                    className={`btn btn-sm ${showHeatmap ? 'btn-danger' : 'btn-secondary'}`}
                    onClick={() => setShowHeatmap(!showHeatmap)}
                    disabled={!analysisResults}
                  >
                    🌡️ {showHeatmap ? 'Hide' : 'Show'} Heatmap
                  </button>
                </>
              )}
              {scanStatus === 'processing' && (
                <div className="status-indicator">
                  <div className="spinner"></div>
                  <span>Processing...</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── Right Panel: Chat ─────────────────────────────── */}
        <div className="right-panel">
          <ChatPanel
            scanId={activeScan?.id}
            sessionId={chatSessionId}
            onSessionChange={setChatSessionId}
            findings={findings}
          />
        </div>
      </div>
    </div>
  );
}
