/**
 * MedSpatial AI — Global State Stores (Zustand)
 * Scan store, UI store, and viewer store — per program.md Section execution order.
 */

import { create } from 'zustand';

// ── Scan Store ───────────────────────────────────────────────────
export const useScanStore = create((set, get) => ({
  // State
  scans: [],
  activeScan: null,
  scanStatus: 'idle',          // 'idle' | 'uploading' | 'processing' | 'error'
  meshUrl: null,
  layerUrls: {},
  volumeDimensions: null,
  analysisResults: null,
  findings: [],
  chatSessionId: null,

  // Actions
  setScans: (scans) => set({ scans }),
  setActiveScan: (scan) => set({
    activeScan: scan,
    meshUrl: null,
    layerUrls: {},
    volumeDimensions: null,
    analysisResults: null,
    findings: [],
    chatSessionId: null,
  }),
  setScanStatus: (status) => set({ scanStatus: status }),
  setMeshUrl: (url) => set({ meshUrl: url }),
  setLayerUrls: (urls) => set({ layerUrls: urls }),
  setVolumeDimensions: (dims) => set({ volumeDimensions: dims }),
  setAnalysisResults: (results) => set({ analysisResults: results }),
  setFindings: (findings) => set({ findings }),
  setChatSessionId: (id) => set({ chatSessionId: id }),
}));

// ── UI / Viewer Store ────────────────────────────────────────────
export const useViewerStore = create((set) => ({
  // Layer controls
  layers: {
    primary:     { visible: true,  opacity: 0.8, color: '#d4d8e0', label: 'Complete Model', icon: '🫁' },
    bone:        { visible: true,  opacity: 0.9, color: '#f0ecd8', label: 'Bone',           icon: '🦴' },
    soft_tissue: { visible: true,  opacity: 0.5, color: '#e6b8a2', label: 'Soft Tissue',    icon: '🫀' },
    air:         { visible: false, opacity: 0.2, color: '#4466cc', label: 'Air / Lung',     icon: '💨' },
    vessel:      { visible: true,  opacity: 0.7, color: '#cc4444', label: 'Vessels',        icon: '🩸' },
  },

  // Viewer controls
  dissectionDepth: 1.0,        // 0.0–1.0
  showHeatmap:     false,
  showMIP:         false,
  showGrid:        true,
  measureMode:     false,

  // Slice viewer
  activeAxis:  'axial',         // 'axial' | 'coronal' | 'sagittal'
  sliceIndex:  64,

  // Camera
  cameraTarget: [0, 0, 0],

  // Actions
  toggleLayer: (name) => set((state) => ({
    layers: {
      ...state.layers,
      [name]: { ...state.layers[name], visible: !state.layers[name].visible },
    },
  })),
  setLayerOpacity: (name, opacity) => set((state) => ({
    layers: { ...state.layers, [name]: { ...state.layers[name], opacity } },
  })),
  setDissectionDepth: (depth) => set({ dissectionDepth: depth }),
  setShowHeatmap:     (show)  => set({ showHeatmap: show }),
  setShowMIP:         (show)  => set({ showMIP: show }),
  setActiveAxis:      (axis)  => set({ activeAxis: axis }),
  setSliceIndex:      (idx)   => set({ sliceIndex: idx }),
  setMeasureMode:     (on)    => set({ measureMode: on }),
  setCameraTarget:    (pos)   => set({ cameraTarget: pos }),
}));
