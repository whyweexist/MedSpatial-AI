/**
 * C2Three — Header Component
 * Top navigation bar with branding and system status.
 */

import React from 'react';

export default function Header({ scanStatus, activeScan }) {
  const statusClass = scanStatus === 'processing' ? 'processing' :
                      scanStatus === 'error' ? 'error' : 'active';

  const statusText = scanStatus === 'processing' ? 'Processing...' :
                     scanStatus === 'error' ? 'Error' :
                     activeScan ? 'Connected' : 'Ready';

  return (
    <header className="header">
      <div className="header-brand">
        <div className="header-logo">C2</div>
        <div className="header-text">
          <h1 className="header-title">C2Three</h1>
          <span className="header-subtitle">3D Medical Imaging Platform</span>
        </div>
      </div>
      <div className="header-status">
        {activeScan && (
          <div className="status-indicator">
            <span className="status-label">Scan</span>
            <span>{activeScan.modality || 'SCAN'} — {activeScan.body_part || 'Unknown'}</span>
          </div>
        )}
        <div className="status-indicator">
          <span className={`status-dot ${statusClass}`}></span>
          <span>{statusText}</span>
        </div>
      </div>
    </header>
  );
}
