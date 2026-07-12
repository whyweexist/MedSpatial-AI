/**
 * MedSpatial AI — Segment Statistics Panel
 * Displays detailed per-segment statistics with mini bar chart.
 */

import React, { useMemo } from 'react';

export default function SegmentStats({ segments, bodyRegion }) {
  const sortedSegments = useMemo(() => {
    if (!segments) return [];
    return [...segments]
      .filter(s => s.volume_cm3 > 0)
      .sort((a, b) => b.volume_cm3 - a.volume_cm3);
  }, [segments]);

  const maxVolume = useMemo(() => {
    return Math.max(...sortedSegments.map(s => s.volume_cm3), 1);
  }, [sortedSegments]);

  const totalVolume = useMemo(() => {
    return sortedSegments.reduce((sum, s) => sum + s.volume_cm3, 0);
  }, [sortedSegments]);

  if (!sortedSegments.length) return null;

  return (
    <div className="segment-stats">
      <div className="sidebar-section-title">
        📊 Segment Statistics
      </div>

      {/* Body region badge */}
      {bodyRegion && (
        <div className="segment-stats-region">
          <span className="segment-stats-region-icon">{bodyRegion.icon || '📍'}</span>
          <span>{bodyRegion.display_name || bodyRegion.region}</span>
          <span className="segment-stats-region-conf">
            {(bodyRegion.confidence * 100).toFixed(0)}%
          </span>
        </div>
      )}

      {/* Total */}
      <div className="segment-stats-total">
        Total Volume: <strong>{totalVolume.toFixed(1)} cm³</strong>
        <span style={{ marginLeft: 'auto', fontSize: 10, color: 'var(--text-muted)' }}>
          {sortedSegments.length} segments
        </span>
      </div>

      {/* Bar chart + details */}
      <div className="segment-stats-bars">
        {sortedSegments.map((segment) => {
          const pct = (segment.volume_cm3 / maxVolume) * 100;
          const totalPct = totalVolume > 0 ? (segment.volume_cm3 / totalVolume * 100).toFixed(1) : 0;

          return (
            <div key={segment.name} className="segment-stats-row">
              <div className="segment-stats-row-header">
                <span
                  className="segment-stats-dot"
                  style={{ background: segment.color || '#808080' }}
                />
                <span className="segment-stats-name">{formatName(segment.name)}</span>
                <span className="segment-stats-vol">
                  {segment.volume_cm3.toFixed(1)} cm³
                </span>
              </div>
              <div className="segment-stats-bar-track">
                <div
                  className="segment-stats-bar-fill"
                  style={{
                    width: `${pct}%`,
                    background: segment.color || '#6366f1',
                  }}
                />
              </div>
              <div className="segment-stats-detail">
                <span>{totalPct}% of total</span>
                {segment.mean_hu !== 0 && (
                  <span>Mean: {segment.mean_hu.toFixed(0)} HU</span>
                )}
                {segment.voxel_count > 0 && (
                  <span>{(segment.voxel_count / 1000).toFixed(1)}k voxels</span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      <style>{`
        .segment-stats {
          margin-bottom: 12px;
        }
        .segment-stats-region {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 6px 10px;
          margin-bottom: 8px;
          background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(6,182,212,0.1));
          border: 1px solid rgba(99,102,241,0.25);
          border-radius: var(--radius-sm, 6px);
          font-size: 12px;
          font-weight: 600;
          color: var(--text-primary, #f1f5f9);
        }
        .segment-stats-region-icon {
          font-size: 16px;
        }
        .segment-stats-region-conf {
          margin-left: auto;
          font-size: 10px;
          color: #06b6d4;
          font-weight: 700;
        }
        .segment-stats-total {
          display: flex;
          align-items: center;
          padding: 6px 10px;
          margin-bottom: 8px;
          background: var(--bg-tertiary, #1e293b);
          border-radius: var(--radius-sm, 6px);
          font-size: 11px;
          color: var(--text-secondary, #94a3b8);
        }
        .segment-stats-total strong {
          color: var(--text-primary, #f1f5f9);
          margin-left: 4px;
        }
        .segment-stats-bars {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        .segment-stats-row {
          padding: 4px 0;
        }
        .segment-stats-row-header {
          display: flex;
          align-items: center;
          gap: 6px;
          margin-bottom: 3px;
        }
        .segment-stats-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          flex-shrink: 0;
        }
        .segment-stats-name {
          font-size: 11px;
          font-weight: 600;
          color: var(--text-primary, #f1f5f9);
          flex: 1;
        }
        .segment-stats-vol {
          font-size: 10px;
          font-weight: 700;
          color: var(--accent, #6366f1);
          font-family: monospace;
        }
        .segment-stats-bar-track {
          height: 4px;
          background: var(--bg-tertiary, #1e293b);
          border-radius: 2px;
          overflow: hidden;
          margin-bottom: 2px;
        }
        .segment-stats-bar-fill {
          height: 100%;
          border-radius: 2px;
          transition: width 0.5s ease;
          min-width: 2px;
        }
        .segment-stats-detail {
          display: flex;
          gap: 8px;
          font-size: 9px;
          color: var(--text-muted, #64748b);
        }
      `}</style>
    </div>
  );
}

function formatName(name) {
  const names = {
    skin: 'Skin', bone: 'Bone',
    left_lung: 'Left Lung', right_lung: 'Right Lung',
    heart: 'Heart', vessels: 'Vessels',
    soft_tissue: 'Soft Tissue', pathology: 'Pathology',
    brain: 'Brain', liver: 'Liver', kidneys: 'Kidneys',
  };
  return names[name] || name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}
