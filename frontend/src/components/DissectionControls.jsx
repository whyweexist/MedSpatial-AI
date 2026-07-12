/**
 * MedSpatial AI — Dissection Controls
 * Full interactive dissection panel: peeling slider, exploded view,
 * isolate mode, preset views, and cross-section controls.
 */

import React, { useState, useCallback } from 'react';

// Dissection order (outside-in for slider peeling)
const DISSECTION_ORDER = {
  skin: 8, soft_tissue: 7, bone: 6,
  left_lung: 4, right_lung: 4, vessels: 3,
  heart: 2, pathology: 1,
};

// Preset layer configurations
const PRESETS = {
  all: { label: '🔄 All Layers', description: 'Show everything', filter: () => true },
  bones: { label: '🦴 Bones Only', description: 'Skeletal structures', filter: (s) => s.name === 'bone' },
  organs: { label: '🫀 Organs', description: 'Heart, lungs, vessels', filter: (s) => ['heart', 'left_lung', 'right_lung', 'vessels'].includes(s.name) },
  lungs: { label: '🫁 Lungs', description: 'Left and right lungs', filter: (s) => ['left_lung', 'right_lung'].includes(s.name) },
  cardio: { label: '❤️ Cardiothoracic', description: 'Heart + vessels + lungs', filter: (s) => ['heart', 'vessels', 'left_lung', 'right_lung'].includes(s.name) },
  deep: { label: '🔬 Deep Tissue', description: 'Heart + vessels + pathology', filter: (s) => ['heart', 'vessels', 'pathology'].includes(s.name) },
};

const CLIP_PRESETS = {
  none: { label: 'No Clip', axis: null },
  axial: { label: 'Axial', axis: 'y' },
  coronal: { label: 'Coronal', axis: 'z' },
  sagittal: { label: 'Sagittal', axis: 'x' },
};

export default function DissectionControls({
  segments,
  onPeelChange,
  onExplodedToggle,
  onIsolateSegment,
  onPresetApply,
  onClipPreset,
  peelDepth,
  isExploded,
  isolatedSegment,
  activeClipAxis,
}) {
  const [activePreset, setActivePreset] = useState('all');

  const handlePresetClick = useCallback((presetKey) => {
    setActivePreset(presetKey);
    const preset = PRESETS[presetKey];
    if (preset && onPresetApply) {
      onPresetApply(presetKey, preset.filter);
    }
  }, [onPresetApply]);

  const handleClipClick = useCallback((clipKey) => {
    if (onClipPreset) {
      onClipPreset(CLIP_PRESETS[clipKey].axis);
    }
  }, [onClipPreset]);

  return (
    <div className="dissection-controls">
      <div className="sidebar-section-title">🔪 Dissection Controls</div>

      {/* Peeling Slider */}
      <div className="dissection-control-group">
        <div className="dissection-label">
          <span>Layer Peeling</span>
          <span className="dissection-value">{Math.round(peelDepth * 100)}%</span>
        </div>
        <div className="dissection-slider-wrap">
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={peelDepth}
            onChange={(e) => onPeelChange && onPeelChange(parseFloat(e.target.value))}
            className="dissection-slider"
          />
          <div className="dissection-slider-labels">
            <span>Skin</span>
            <span>Core</span>
          </div>
        </div>
      </div>

      {/* Exploded View Toggle */}
      <div className="dissection-control-group">
        <button
          className={`dissection-btn ${isExploded ? 'dissection-btn-active' : ''}`}
          onClick={() => onExplodedToggle && onExplodedToggle()}
        >
          💥 {isExploded ? 'Collapse View' : 'Exploded View'}
        </button>
      </div>

      {/* Preset Views */}
      <div className="dissection-control-group">
        <div className="dissection-label">Preset Views</div>
        <div className="dissection-presets">
          {Object.entries(PRESETS).map(([key, preset]) => (
            <button
              key={key}
              className={`dissection-preset-btn ${activePreset === key ? 'dissection-preset-active' : ''}`}
              onClick={() => handlePresetClick(key)}
              title={preset.description}
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>

      {/* Cross-Section Presets */}
      <div className="dissection-control-group">
        <div className="dissection-label">Cross-Section</div>
        <div className="dissection-presets">
          {Object.entries(CLIP_PRESETS).map(([key, preset]) => (
            <button
              key={key}
              className={`dissection-preset-btn ${activeClipAxis === preset.axis ? 'dissection-preset-active' : ''}`}
              onClick={() => handleClipClick(key)}
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>

      {/* Segment List with Isolate */}
      {segments && segments.length > 0 && (
        <div className="dissection-control-group">
          <div className="dissection-label">Segments ({segments.length})</div>
          <div className="dissection-segments-list">
            {segments.map((segment) => {
              const isIsolated = isolatedSegment === segment.name;
              const shouldDim = isolatedSegment && !isIsolated;
              const order = DISSECTION_ORDER[segment.name] || 5;
              const isPeeled = (order / 8) > (1 - peelDepth);

              return (
                <div
                  key={segment.name}
                  className={`dissection-segment-item ${isIsolated ? 'dissection-segment-isolated' : ''} ${isPeeled ? 'dissection-segment-peeled' : ''}`}
                  style={{ opacity: shouldDim ? 0.4 : 1 }}
                  onClick={() => onIsolateSegment && onIsolateSegment(isIsolated ? null : segment.name)}
                >
                  <div
                    className="dissection-segment-color"
                    style={{ background: segment.color || '#808080' }}
                  />
                  <div className="dissection-segment-info">
                    <div className="dissection-segment-name">
                      {formatSegmentName(segment.name)}
                    </div>
                    <div className="dissection-segment-stats">
                      {segment.volume_cm3 > 0 ? `${segment.volume_cm3.toFixed(1)} cm³` : '—'}
                      {segment.mean_hu ? ` · ${segment.mean_hu.toFixed(0)} HU` : ''}
                    </div>
                  </div>
                  <div className="dissection-segment-badge">
                    {isIsolated ? '👁️' : isPeeled ? '✂️' : ''}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <style>{`
        .dissection-controls {
          margin-bottom: 12px;
        }
        .dissection-control-group {
          margin-bottom: 14px;
          padding: 0 4px;
        }
        .dissection-label {
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: 11px;
          font-weight: 600;
          color: var(--text-secondary, #94a3b8);
          margin-bottom: 6px;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }
        .dissection-value {
          font-family: monospace;
          color: var(--accent, #6366f1);
          font-weight: 700;
        }
        .dissection-slider-wrap {
          position: relative;
        }
        .dissection-slider {
          width: 100%;
          height: 6px;
          -webkit-appearance: none;
          appearance: none;
          background: linear-gradient(90deg, #6366f1, #06b6d4);
          border-radius: 3px;
          outline: none;
          cursor: pointer;
        }
        .dissection-slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 16px;
          height: 16px;
          border-radius: 50%;
          background: #6366f1;
          border: 2px solid white;
          cursor: pointer;
          box-shadow: 0 0 6px rgba(99,102,241,0.5);
        }
        .dissection-slider-labels {
          display: flex;
          justify-content: space-between;
          font-size: 9px;
          color: var(--text-muted, #64748b);
          margin-top: 2px;
        }
        .dissection-btn {
          width: 100%;
          padding: 8px 12px;
          border: 1px solid var(--border, #334155);
          border-radius: var(--radius-sm, 6px);
          background: var(--bg-tertiary, #1e293b);
          color: var(--text-primary, #f1f5f9);
          font-size: 12px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
          text-align: center;
        }
        .dissection-btn:hover {
          background: var(--bg-secondary, #2a3a50);
          border-color: var(--accent, #6366f1);
        }
        .dissection-btn-active {
          background: rgba(99, 102, 241, 0.2);
          border-color: #6366f1;
          color: #818cf8;
        }
        .dissection-presets {
          display: flex;
          flex-wrap: wrap;
          gap: 4px;
        }
        .dissection-preset-btn {
          flex: 0 0 auto;
          padding: 4px 8px;
          border: 1px solid var(--border, #334155);
          border-radius: 12px;
          background: var(--bg-tertiary, #1e293b);
          color: var(--text-secondary, #94a3b8);
          font-size: 10px;
          cursor: pointer;
          transition: all 0.2s;
          white-space: nowrap;
        }
        .dissection-preset-btn:hover {
          border-color: var(--accent, #6366f1);
          color: var(--text-primary, #f1f5f9);
        }
        .dissection-preset-active {
          background: rgba(99, 102, 241, 0.2);
          border-color: #6366f1;
          color: #818cf8;
        }
        .dissection-segments-list {
          max-height: 240px;
          overflow-y: auto;
          display: flex;
          flex-direction: column;
          gap: 2px;
        }
        .dissection-segment-item {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 6px 8px;
          border-radius: var(--radius-sm, 6px);
          cursor: pointer;
          transition: all 0.2s;
        }
        .dissection-segment-item:hover {
          background: var(--bg-tertiary, #1e293b);
        }
        .dissection-segment-isolated {
          background: rgba(99, 102, 241, 0.15) !important;
          border-left: 3px solid #6366f1;
        }
        .dissection-segment-peeled {
          opacity: 0.5;
          text-decoration: line-through;
        }
        .dissection-segment-color {
          width: 10px;
          height: 10px;
          border-radius: 50%;
          flex-shrink: 0;
          border: 1px solid rgba(255,255,255,0.2);
        }
        .dissection-segment-info {
          flex: 1;
          min-width: 0;
        }
        .dissection-segment-name {
          font-size: 11px;
          font-weight: 600;
          color: var(--text-primary, #f1f5f9);
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }
        .dissection-segment-stats {
          font-size: 9px;
          color: var(--text-muted, #64748b);
        }
        .dissection-segment-badge {
          font-size: 12px;
          width: 18px;
          text-align: center;
        }
      `}</style>
    </div>
  );
}

function formatSegmentName(name) {
  const names = {
    skin: '🧍 Skin',
    bone: '🦴 Bone',
    left_lung: '🫁 Left Lung',
    right_lung: '🫁 Right Lung',
    heart: '❤️ Heart',
    vessels: '🩸 Vessels',
    soft_tissue: '🫀 Soft Tissue',
    pathology: '⚠️ Pathology',
    brain: '🧠 Brain',
    liver: '🫘 Liver',
    kidneys: '🫘 Kidneys',
  };
  return names[name] || name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}
