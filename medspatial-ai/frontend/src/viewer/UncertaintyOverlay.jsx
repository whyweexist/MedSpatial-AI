import React from 'react';

export default function UncertaintyOverlay({ uncertainty = 1, label = 'Model uncertainty' }) {
  const percent = Math.max(0, Math.min(100, uncertainty * 100));
  return (
    <div className="awm-uncertainty" aria-label={`${label}: ${percent.toFixed(0)} percent`}>
      <div className="awm-panel-row">
        <span>{label}</span>
        <strong>{percent.toFixed(0)}%</strong>
      </div>
      <div className="progress-bar">
        <div className="progress-fill uncertainty-fill" style={{ width: `${percent}%` }} />
      </div>
    </div>
  );
}
