/**
 * MedSpatial AI — Body Region Badge
 * Displays detected body region with icon and confidence badge.
 */

import React from 'react';

export default function BodyRegionBadge({ bodyRegion }) {
  if (!bodyRegion) return null;

  const confidenceColor =
    bodyRegion.confidence > 0.8 ? '#10b981' :
    bodyRegion.confidence > 0.5 ? '#f59e0b' : '#ef4444';

  return (
    <div className="body-region-badge" title={`Detection method: ${bodyRegion.method || 'auto'}`}>
      <span className="body-region-icon">{bodyRegion.icon || '📍'}</span>
      <span className="body-region-text">
        {bodyRegion.display_name || bodyRegion.region}
      </span>
      <span className="body-region-modality">{bodyRegion.modality || ''}</span>
      <span
        className="body-region-conf"
        style={{ color: confidenceColor }}
      >
        {(bodyRegion.confidence * 100).toFixed(0)}%
      </span>

      <style>{`
        .body-region-badge {
          display: inline-flex;
          align-items: center;
          gap: 5px;
          padding: 4px 10px;
          background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(6,182,212,0.15));
          border: 1px solid rgba(99,102,241,0.3);
          border-radius: 20px;
          font-size: 11px;
          font-weight: 600;
          color: var(--text-primary, #f1f5f9);
          white-space: nowrap;
          backdrop-filter: blur(4px);
        }
        .body-region-icon { font-size: 14px; }
        .body-region-text { text-transform: uppercase; letter-spacing: 0.5px; }
        .body-region-modality {
          padding: 1px 5px;
          background: rgba(255,255,255,0.1);
          border-radius: 8px;
          font-size: 9px;
          color: var(--text-muted, #94a3b8);
        }
        .body-region-conf {
          font-family: monospace;
          font-size: 10px;
          font-weight: 700;
        }
      `}</style>
    </div>
  );
}
