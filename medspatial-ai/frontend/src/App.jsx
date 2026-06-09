/**
 * MedSpatial AI — Main Application (Enhanced)
 * Root layout: global state, three-panel layout, dissection module,
 * body region, XAI, reports, and anatomy labels integration.
 */

import React, { useState, useCallback, useEffect, useRef } from 'react';
import Header from './components/Header';
import UploadPanel from './components/UploadPanel';
import LayerControls from './components/LayerControls';
import DissectionControls from './components/DissectionControls';
import SegmentStats from './components/SegmentStats';
import SliceViewer from './components/SliceViewer';
import Viewer3D from './components/Viewer3D';
import ChatPanel from './components/ChatPanel';
import AnomalyOverlay from './components/AnomalyOverlay';
import BodyRegionBadge from './components/BodyRegionBadge';
import ExplainPanel from './components/ExplainPanel';
import ReportButton from './components/ReportButton';
import GroundedAssistantPanel from './assistant/GroundedAssistantPanel';
import SceneGraphPanel from './viewer/SceneGraphPanel';
import ProvenanceBadge from './viewer/ProvenanceBadge';
import UncertaintyOverlay from './viewer/UncertaintyOverlay';
import { getAWM, getSceneGraph } from './world/AWMClient';
import {
  listScans,
  startReconstruction,
  getReconstructionStatus,
  getSegments,
  runAnalysis,
  getAnalysisResults,
} from './services/api';

// 8-layer system matching the backend
const DEFAULT_LAYERS = {
  primary: { visible: true, opacity: 0.8, color: '#d4d8e0' },
  skin: { visible: false, opacity: 0.3, color: '#e6bf9e' },
  bone: { visible: true, opacity: 0.9, color: '#f2ebb0' },
  left_lung: { visible: true, opacity: 0.4, color: '#66a6d9' },
  right_lung: { visible: true, opacity: 0.4, color: '#4d8ccc' },
  heart: { visible: true, opacity: 0.7, color: '#e66666' },
  vessels: { visible: true, opacity: 0.7, color: '#d93333' },
  soft_tissue: { visible: true, opacity: 0.4, color: '#e6b399' },
  pathology: { visible: true, opacity: 0.9, color: '#ff2600' },
  bone_marrow: { visible: true, opacity: 0.55, color: '#bf5950' },
  spinal_canal: { visible: true, opacity: 0.45, color: '#59a6e6' },
  paraspinal_soft_tissue: { visible: true, opacity: 0.35, color: '#b88573' },
};

function layersForStudy(layerUrls = {}) {
  const available = {
    primary: { ...DEFAULT_LAYERS.primary },
  };
  Object.keys(layerUrls).forEach((name) => {
    available[name] = {
      ...(DEFAULT_LAYERS[name] || { visible: true, opacity: 0.6, color: '#808080' }),
    };
  });
  return available;
}

export default function App() {
  // ── Global State ──────────────────────────────────────────
  const [scans, setScans] = useState([]);
  const [activeScan, setActiveScan] = useState(null);
  const [scanStatus, setScanStatus] = useState('idle');

  // Reconstruction state
  const [meshUrl, setMeshUrl] = useState(null);
  const [layerUrls, setLayerUrls] = useState({});
  const [volumeDimensions, setVolumeDimensions] = useState(null);
  const [bodyRegion, setBodyRegion] = useState(null);
  const [reconSummary, setReconSummary] = useState(null);
  const [anatomyLabels, setAnatomyLabels] = useState([]);

  // Segments / Dissection state
  const [segments, setSegments] = useState([]);
  const [peelDepth, setPeelDepth] = useState(0);
  const [isExploded, setIsExploded] = useState(false);
  const [isolatedSegment, setIsolatedSegment] = useState(null);
  const [clipAxis, setClipAxis] = useState(null);
  const [showLabels, setShowLabels] = useState(true);

  // Layer visibility
  const [layers, setLayers] = useState(DEFAULT_LAYERS);

  // Analysis state
  const [analysisResults, setAnalysisResults] = useState(null);
  const [findings, setFindings] = useState([]);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(null);
  const [awm, setAwm] = useState(null);
  const [sceneGraph, setSceneGraph] = useState(null);

  // Chat state
  const [chatSessionId, setChatSessionId] = useState(null);

  // UI Toggle state
  const [showLeftPanel, setShowLeftPanel] = useState(true);
  const [showRightPanel, setShowRightPanel] = useState(true);
  const [showDissection, setShowDissection] = useState(true);
  const [showLayer, setShowLayer] = useState(true);
  const [showSlice, setShowSlice] = useState(true);
  const [showChat, setShowChat] = useState(true);
  const [showSegmentStats, setShowSegmentStats] = useState(true);
  const [sliceSize, setSliceSize] = useState({ width: 400, height: 300 });
  const [slicePosition, setSlicePosition] = useState(() => ({
    x: Math.max(20, window.innerWidth - 420),
    y: Math.max(20, window.innerHeight - 320)
  }));
  const sliceRef = useRef(null);
  const dragRef = useRef({ isDragging: false, startX: 0, startY: 0, startPosX: 0, startPosY: 0 });

  // Handle slice viewer drag
  const handleMouseDown = useCallback((e) => {
    // Only allow dragging from the header
    if (e.target.closest('.slice-viewer-header')) {
      dragRef.current.isDragging = true;
      dragRef.current.startX = e.clientX;
      dragRef.current.startY = e.clientY;
      dragRef.current.startPosX = slicePosition.x;
      dragRef.current.startPosY = slicePosition.y;
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      e.preventDefault();
    }
  }, [slicePosition]);

  const handleMouseMove = useCallback((e) => {
    if (dragRef.current.isDragging) {
      const deltaX = e.clientX - dragRef.current.startX;
      const deltaY = e.clientY - dragRef.current.startY;
      const newX = dragRef.current.startPosX + deltaX;
      const newY = dragRef.current.startPosY + deltaY;

      // Constrain to viewport bounds
      const maxX = window.innerWidth - sliceSize.width - 20;
      const maxY = window.innerHeight - sliceSize.height - 20;
      const constrainedX = Math.min(Math.max(newX, 20), maxX);
      const constrainedY = Math.min(Math.max(newY, 20), maxY);

      setSlicePosition({
        x: constrainedX,
        y: constrainedY,
      });
    }
  }, [sliceSize]);

  const handleMouseUp = useCallback(() => {
    dragRef.current.isDragging = false;
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', handleMouseUp);
  }, [handleMouseMove]);

  // Handle slice viewer resize
  useEffect(() => {
    if (sliceRef.current && showSlice) {
      const resizeObserver = new ResizeObserver((entries) => {
        for (let entry of entries) {
          const { width, height } = entry.contentRect;
          setSliceSize({ width, height });

          // Adjust position if resize would put it out of bounds
          setSlicePosition(prev => {
            const maxX = window.innerWidth - width - 20;
            const maxY = window.innerHeight - height - 20;
            return {
              x: Math.min(prev.x, maxX),
              y: Math.min(prev.y, maxY),
            };
          });
        }
      });
      resizeObserver.observe(sliceRef.current);
      return () => resizeObserver.disconnect();
    }
  }, [showSlice]);

  // Handle window resize to keep slice viewer in bounds
  useEffect(() => {
    const handleResize = () => {
      setSlicePosition(prev => {
        const maxX = window.innerWidth - sliceSize.width - 20;
        const maxY = window.innerHeight - sliceSize.height - 20;
        return {
          x: Math.min(Math.max(prev.x, 20), maxX),
          y: Math.min(Math.max(prev.y, 20), maxY),
        };
      });
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [sliceSize]);

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
    setAnalysisProgress(null);
    setAwm(null);
    setSceneGraph(null);
    setSegments([]);
    setBodyRegion(null);
    setReconSummary(null);
    setAnatomyLabels([]);
    setPeelDepth(0);
    setIsExploded(false);
    setIsolatedSegment(null);
    setClipAxis(null);

    try {
      const [world, graph] = await Promise.all([
        getAWM(scan.id),
        getSceneGraph(scan.id),
      ]);
      setAwm(world);
      setSceneGraph(graph);
    } catch (error) {
      console.warn('AWM is not available for this legacy study:', error);
    }

    if (scan.status === 'reconstructed' || scan.status === 'analyzed') {
      try {
        const status = await getReconstructionStatus(scan.id);
        if (status.mesh_url) {
          setMeshUrl(status.mesh_url);
          setLayerUrls(status.layer_urls || {});
          setVolumeDimensions(status.dimensions);
          setBodyRegion(status.body_region || null);
          setReconSummary(status.summary || null);
          setAnatomyLabels(status.labels || []);

          // Build layers from available URLs
          setLayers(layersForStudy(status.layer_urls || {}));

          // Load segments
          try {
            const segData = await getSegments(scan.id);
            setSegments(segData.segments || []);
            if (segData.body_region) setBodyRegion(segData.body_region);
          } catch (e) {
            console.warn('Segments not available:', e);
          }
        }

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

      const poll = setInterval(async () => {
        try {
          const status = await getReconstructionStatus(activeScan.id);
          if (status.status === 'reconstructed' || status.status === 'analyzed') {
            clearInterval(poll);
            setMeshUrl(status.mesh_url);
            setLayerUrls(status.layer_urls || {});
            setLayers(layersForStudy(status.layer_urls || {}));
            setVolumeDimensions(status.dimensions);
            setBodyRegion(status.body_region || null);
            setReconSummary(status.summary || null);
            setAnatomyLabels(status.labels || []);
            setScanStatus('idle');
            setActiveScan(prev => ({ ...prev, status: status.status }));

            // Load segments
            try {
              const segData = await getSegments(activeScan.id);
              setSegments(segData.segments || []);
            } catch (e) { /* ok */ }

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
    setAnalysisProgress({ progress: 0, stage: 'queued', eta_seconds: null });
    try {
      const result = await runAnalysis(activeScan.id, 'full');

      const poll = setInterval(async () => {
        try {
          const results = await getAnalysisResults(activeScan.id);
          const latest = results.find(r => r.analysis_id === result.analysis_id);
          if (latest) {
            setAnalysisProgress({
              progress: latest.progress ?? 0,
              stage: latest.stage || latest.status,
              eta_seconds: latest.eta_seconds,
            });
          }
          if (latest && latest.status === 'completed') {
            clearInterval(poll);
            setAnalysisResults(latest);
            setFindings(latest.findings || []);
            setScanStatus('idle');
            setAnalysisProgress(null);
            loadScans();
          } else if (latest && latest.status === 'failed') {
            clearInterval(poll);
            setScanStatus('error');
            setAnalysisProgress(null);
            alert('Analysis failed. Check logs for details.');
          }
          // Still processing - continue polling
        } catch (e) {
          console.error('Analysis poll error:', e);
        }
      }, 2000);
    } catch (e) {
      console.error('Analysis failed:', e);
      setScanStatus('error');
      setAnalysisProgress(null);
      alert('Failed to start analysis. Check console for details.');
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

  // ── Dissection Controls ───────────────────────────────────
  const handlePresetApply = useCallback((presetKey, filterFn) => {
    if (presetKey === 'all') {
      setLayers(prev => {
        const updated = { ...prev };
        Object.keys(updated).forEach(k => { updated[k] = { ...updated[k], visible: true }; });
        return updated;
      });
      setIsolatedSegment(null);
      return;
    }
    setLayers(prev => {
      const updated = { ...prev };
      Object.keys(updated).forEach(k => {
        if (k === 'primary') {
          updated[k] = { ...updated[k], visible: false };
        } else {
          updated[k] = { ...updated[k], visible: filterFn({ name: k }) };
        }
      });
      return updated;
    });
    setIsolatedSegment(null);
  }, []);

  const handleIsolateSegment = useCallback((segmentName) => {
    setIsolatedSegment(segmentName);
    if (segmentName) {
      setLayers(prev => {
        const updated = { ...prev };
        Object.keys(updated).forEach(k => {
          if (k === segmentName) {
            updated[k] = { ...updated[k], visible: true, opacity: 0.9 };
          } else {
            updated[k] = { ...updated[k], opacity: 0.08 };
          }
        });
        return updated;
      });
    } else {
      // Reset all opacities
      setLayers(prev => {
        const updated = { ...prev };
        Object.keys(updated).forEach(k => {
          const defaults = DEFAULT_LAYERS[k] || { opacity: 0.6 };
          updated[k] = { ...updated[k], opacity: defaults.opacity };
        });
        return updated;
      });
    }
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
        {showLeftPanel && (
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
          </div>
        )}

        {/* ── Center: 3D Viewer ─────────────────────────────── */}
        <div className="viewer-area">
          {analysisProgress && (
            <div className="analysis-progress-card" role="status" aria-live="polite">
              <div className="analysis-progress-header">
                <strong>Analyzing study</strong>
                <span>{Math.round(analysisProgress.progress || 0)}%</span>
              </div>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${analysisProgress.progress || 0}%` }} />
              </div>
              <div className="analysis-progress-meta">
                <span>{(analysisProgress.stage || 'working').replaceAll('_', ' ')}</span>
                <span>
                  {analysisProgress.eta_seconds === null || analysisProgress.eta_seconds === undefined
                    ? 'Estimating time...'
                    : analysisProgress.eta_seconds < 60
                      ? `About ${Math.max(1, analysisProgress.eta_seconds)} sec remaining`
                      : `About ${Math.max(1, Math.ceil(analysisProgress.eta_seconds / 60))} min remaining`}
                </span>
              </div>
            </div>
          )}
          {activeScan && meshUrl ? (
            <>
              <Viewer3D
                meshUrl={meshUrl}
                layerUrls={layerUrls}
                layers={layers}
                showHeatmap={showHeatmap}
                findings={findings}
                segments={segments}
                peelDepth={peelDepth}
                isExploded={isExploded}
                isolatedSegment={isolatedSegment}
                clipAxis={clipAxis}
                anatomyLabels={anatomyLabels}
                showLabels={showLabels}
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
              <div className="empty-state-icon">Medical Imaging</div>
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
                  Build 3D Model
                </button>
              )}
            </div>
          )}

          {/* Toolbar */}
          {activeScan && (
            <div className="viewer-toolbar">
              {bodyRegion && <BodyRegionBadge bodyRegion={bodyRegion} />}

              {activeScan.status === 'uploaded' && (
                <button className="btn btn-primary btn-sm" onClick={handleReconstruct}>Reconstruct</button>
              )}
              {(activeScan.status === 'reconstructed' || activeScan.status === 'analyzed') && (
                <>
                  <button className="btn btn-primary btn-sm" onClick={handleAnalyze}>Analyze</button>
                  <button
                    className={`btn btn-sm ${showHeatmap ? 'btn-danger' : 'btn-secondary'}`}
                    onClick={() => setShowHeatmap(!showHeatmap)}
                    disabled={!analysisResults}
                  >
                    {showHeatmap ? 'Hide' : 'Show'} Heatmap
                  </button>
                  <button
                    className={`btn btn-sm ${showLabels ? 'btn-info' : 'btn-secondary'}`}
                    onClick={() => setShowLabels(!showLabels)}
                    style={{ fontSize: 11 }}
                  >
                    Labels
                  </button>
                  <ReportButton scanId={activeScan?.id} />
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

        {/* ── Right Panel ─────────────────────────────── */}
        {showRightPanel && (
          <div className="right-panel">
            {activeScan && (
              <div className="right-panel-controls">
                {segments.length > 0 && showDissection && (
                  <DissectionControls
                    segments={segments}
                    onPeelChange={setPeelDepth}
                    onExplodedToggle={() => setIsExploded(prev => !prev)}
                    onIsolateSegment={handleIsolateSegment}
                    onPresetApply={handlePresetApply}
                    onClipPreset={setClipAxis}
                    peelDepth={peelDepth}
                    isExploded={isExploded}
                    isolatedSegment={isolatedSegment}
                    activeClipAxis={clipAxis}
                  />
                )}

                {showLayer && (
                  <LayerControls
                    layers={layers}
                    layerUrls={layerUrls}
                    onToggle={toggleLayer}
                    onOpacityChange={setLayerOpacity}
                  />
                )}

                {segments.length > 0 && showSegmentStats && (
                  <SegmentStats segments={segments} bodyRegion={bodyRegion} />
                )}

                {/* XAI Panel */}
                {findings.length > 0 && (
                  <ExplainPanel scanId={activeScan?.id} findings={findings} />
                )}

                {awm && (
                  <div className="awm-card">
                    <div className="sidebar-section-title">World Model Status</div>
                    <div className="awm-badges">
                      <ProvenanceBadge grounding={
                        awm.meshes?.[0]?.grounding ||
                        awm.volumes?.[0]?.grounding ||
                        (awm.study?.source_format === 'synthetic' ? 'synthetic' : 'scan_derived')
                      } />
                      <span className="badge badge-info">{awm.architecture}</span>
                    </div>
                    <UncertaintyOverlay uncertainty={awm.latent_state?.uncertainty ?? 1} />
                    {['XR', 'DX', 'CR'].includes(awm.study?.modality) && (
                      <p className="awm-muted">
                        3D anatomy is probabilistic, atlas-aligned, estimated, and not patient-specific ground truth.
                      </p>
                    )}
                    {awm.study?.source_format === 'synthetic' && (
                      <p className="awm-muted">Synthetic educational/demo content. Non-diagnostic and non-patient-specific.</p>
                    )}
                  </div>
                )}
                <SceneGraphPanel graph={sceneGraph} />
              </div>
            )}

            <div className="right-panel-bottom">
              {showChat && (
                activeScan ? (
                  <GroundedAssistantPanel studyId={activeScan.id} />
                ) : (
                  <ChatPanel
                    scanId={activeScan?.id}
                    sessionId={chatSessionId}
                    onSessionChange={setChatSessionId}
                    findings={findings}
                  />
                )
              )}
            </div>
          </div>
        )}

        {/* ── Floating Slice Viewer ─────────────────────────── */}
        {showSlice && activeScan && (
          <div
            ref={sliceRef}
            className="floating-slice-viewer"
            style={{
              width: sliceSize.width,
              height: sliceSize.height,
              left: slicePosition.x,
              top: slicePosition.y,
            }}
            onMouseDown={handleMouseDown}
          >
            <SliceViewer scanId={activeScan?.id} volumeDimensions={volumeDimensions} />
          </div>
        )}
      </div>

      {/* ── Toggle Buttons ─────────────────────────────────── */}
      <div className="toggle-buttons">
        <button
          className="toggle-btn"
          onClick={() => setShowLeftPanel(!showLeftPanel)}
          title={showLeftPanel ? 'Hide Left Panel' : 'Show Left Panel'}
        >
          {showLeftPanel ? 'Hide Left' : 'Show Left'}
        </button>
        <button
          className="toggle-btn"
          onClick={() => setShowRightPanel(!showRightPanel)}
          title={showRightPanel ? 'Hide Right Panel' : 'Show Right Panel'}
        >
          {showRightPanel ? 'Hide Right' : 'Show Right'}
        </button>
        {activeScan && segments.length > 0 && (
          <button
            className="toggle-btn"
            onClick={() => setShowDissection(!showDissection)}
            title={showDissection ? 'Minimize Dissection' : 'Maximize Dissection'}
          >
            {showDissection ? 'Min Dissect' : 'Max Dissect'}
          </button>
        )}
        {activeScan && segments.length > 0 && (
          <button
            className="toggle-btn"
            onClick={() => setShowSegmentStats(!showSegmentStats)}
            title={showSegmentStats ? 'Minimize Stats' : 'Maximize Stats'}
          >
            {showSegmentStats ? 'Min Stats' : 'Max Stats'}
          </button>
        )}
        {activeScan && (
          <button
            className="toggle-btn"
            onClick={() => setShowLayer(!showLayer)}
            title={showLayer ? 'Minimize Layers' : 'Maximize Layers'}
          >
            {showLayer ? 'Min Layers' : 'Max Layers'}
          </button>
        )}
        {activeScan && (
          <button
            className="toggle-btn"
            onClick={() => setShowSlice(!showSlice)}
            title={showSlice ? 'Hide Slice Viewer' : 'Show Slice Viewer'}
          >
            {showSlice ? 'Hide Slice' : 'Show Slice'}
          </button>
        )}
        {activeScan && (
          <button
            className="toggle-btn"
            onClick={() => setShowChat(!showChat)}
            title={showChat ? 'Hide Chat' : 'Show Chat'}
          >
            {showChat ? 'Hide Chat' : 'Show Chat'}
          </button>
        )}
      </div>
    </div>
  );
}
