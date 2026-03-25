/**
 * MedSpatial AI — Header Component
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
        <div className="header-logo">M</div>
        <h1 className="header-title">MedSpatial AI</h1>
        <span className="header-subtitle">3D Medical Imaging Platform</span>
      </div>
      <div className="header-status">
        {activeScan && (
          <div className="status-indicator">
            <span>📋</span>
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
