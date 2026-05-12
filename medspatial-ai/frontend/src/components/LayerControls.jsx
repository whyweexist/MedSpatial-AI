/**
 * MedSpatial AI — Layer Controls Component
 * Toggle visibility, adjust opacity, and manage tissue layers in the 3D view.
 */

import React from 'react';

const LAYER_INFO = {
  primary: { icon: '📦', label: 'Complete Model', description: 'Full reconstructed volume' },
  skin: { icon: '🧍', label: 'Skin', description: 'Skin & outline' },
  bone: { icon: '🦴', label: 'Bones', description: 'Skeletal structures' },
  left_lung: { icon: '🫁', label: 'Left Lung', description: 'Left lung parenchyma' },
  right_lung: { icon: '🫁', label: 'Right Lung', description: 'Right lung parenchyma' },
  heart: { icon: '❤️', label: 'Heart', description: 'Cardiac silhouette' },
  vessels: { icon: '🩸', label: 'Vessels', description: 'Vasculature' },
  soft_tissue: { icon: '🫀', label: 'Soft Tissue', description: 'Muscles & organs' },
  pathology: { icon: '⚠️', label: 'Pathology', description: 'Anomalies' },
  brain: { icon: '🧠', label: 'Brain', description: 'Brain parenchyma' },
  liver: { icon: '🫘', label: 'Liver', description: 'Hepatic tissue' },
  kidneys: { icon: '🫘', label: 'Kidneys', description: 'Renal tissue' },
};

export default function LayerControls({ layers, layerUrls, onToggle, onOpacityChange }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div className="sidebar-section-title">
        🔬 Layer Dissection
      </div>

      {Object.entries(layers).map(([name, config]) => {
        const info = LAYER_INFO[name] || { icon: '📦', label: name, description: '' };
        const hasUrl = name === 'primary' || !!layerUrls[name];

        return (
          <div key={name} className="layer-control" style={{ opacity: hasUrl ? 1 : 0.4 }}>
            {/* Color indicator */}
            <div
              className="layer-color-dot"
              style={{ background: config.color }}
            ></div>

            {/* Toggle */}
            <label className="toggle">
              <input
                type="checkbox"
                checked={config.visible}
                onChange={() => onToggle(name)}
                disabled={!hasUrl}
              />
              <span className="toggle-track"></span>
            </label>

            {/* Label */}
            <div style={{ flex: 1, minWidth: 0 }}>
              <div className="layer-name">
                {info.icon} {info.label}
              </div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                {info.description}
              </div>
            </div>

            {/* Opacity slider */}
            <div className="layer-opacity">
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={config.opacity}
                onChange={(e) => onOpacityChange(name, parseFloat(e.target.value))}
                disabled={!hasUrl || !config.visible}
                title={`Opacity: ${Math.round(config.opacity * 100)}%`}
              />
            </div>
          </div>
        );
      })}

      <div style={{
        marginTop: 12,
        padding: '8px 12px',
        background: 'var(--bg-tertiary)',
        borderRadius: 'var(--radius-sm)',
        fontSize: 11,
        color: 'var(--text-muted)',
        lineHeight: 1.5,
      }}>
        💡 Toggle layers to dissect the 3D model. Adjust opacity to see through tissue layers.
        Use the clip slider on the viewer to cut through the volume.
      </div>
    </div>
  );
}
