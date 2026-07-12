/**
 * MedSpatial AI — Anomaly Overlay Component
 * Displays findings panel with severity indicators overlaid on the viewer.
 */

import React from 'react';

export default function AnomalyOverlay({ findings, analysisResults }) {
  if (!findings || findings.length === 0) {
    return (
      <div style={{
        position: 'absolute',
        top: 16,
        left: 16,
        background: 'var(--bg-glass)',
        backdropFilter: 'blur(12px)',
        borderRadius: 'var(--radius-md)',
        border: '1px solid var(--border-subtle)',
        padding: '12px 16px',
        maxWidth: 300,
        zIndex: 20,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
          <span>✅</span>
          <span style={{ fontWeight: 600, fontSize: 13 }}>No Anomalies Detected</span>
        </div>
        <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
          The AI analysis did not detect any significant abnormalities in this scan.
          {analysisResults?.confidence !== undefined && (
            <span> Confidence: {((1 - analysisResults.confidence) * 100).toFixed(0)}% normal.</span>
          )}
        </div>
      </div>
    );
  }

  return (
    <div style={{
      position: 'absolute',
      top: 16,
      left: 16,
      background: 'var(--bg-glass)',
      backdropFilter: 'blur(12px)',
      borderRadius: 'var(--radius-md)',
      border: '1px solid var(--border-subtle)',
      padding: '12px 16px',
      maxWidth: 320,
      maxHeight: 'calc(100% - 80px)',
      overflowY: 'auto',
      zIndex: 20,
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: 12,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span>🔬</span>
          <span style={{ fontWeight: 600, fontSize: 13 }}>
            AI Findings ({findings.length})
          </span>
        </div>
        {analysisResults?.confidence !== undefined && (
          <span className="badge badge-warning" style={{ fontSize: 9 }}>
            {(analysisResults.confidence * 100).toFixed(0)}% anomaly
          </span>
        )}
      </div>

      {/* Summary */}
      {analysisResults?.summary && (
        <div style={{
          fontSize: 12,
          color: 'var(--text-secondary)',
          lineHeight: 1.5,
          marginBottom: 12,
          padding: '8px 10px',
          background: 'var(--bg-tertiary)',
          borderRadius: 'var(--radius-sm)',
        }}>
          {analysisResults.summary.length > 200
            ? analysisResults.summary.slice(0, 200) + '...'
            : analysisResults.summary}
        </div>
      )}

      {/* Findings list */}
      {findings.map((finding, i) => (
        <div key={i} className={`finding-item ${finding.severity || 'mild'}`}>
          <div className="finding-header">
            <span className={`finding-severity ${finding.severity || 'mild'}`}>
              {finding.severity || 'unknown'}
            </span>
            <span className="finding-confidence">
              {finding.confidence !== undefined
                ? `${(finding.confidence * 100).toFixed(0)}%`
                : '—'}
            </span>
          </div>
          <div style={{ fontSize: 12, fontWeight: 500, color: 'var(--text-primary)', marginBottom: 2 }}>
            📍 {finding.region || 'Unknown region'}
          </div>
          <div className="finding-description">
            {finding.description || 'No description available.'}
          </div>
        </div>
      ))}

      {/* Disclaimer */}
      <div style={{
        marginTop: 12,
        padding: '6px 8px',
        background: 'rgba(245,158,11,0.08)',
        borderRadius: 4,
        fontSize: 10,
        color: 'var(--warning)',
        lineHeight: 1.4,
      }}>
        ⚠️ AI-assisted analysis. Should be reviewed by a qualified radiologist.
      </div>
    </div>
  );
}
