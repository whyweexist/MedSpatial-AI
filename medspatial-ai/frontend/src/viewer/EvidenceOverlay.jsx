import React from 'react';

export default function EvidenceOverlay({ anchors = [], onSelect }) {
  return (
    <section className="awm-card" aria-label="Evidence anchors">
      <div className="sidebar-section-title">Evidence</div>
      {anchors.length === 0 && <p className="awm-muted">No supporting evidence is available.</p>}
      {anchors.map(anchor => (
        <button key={anchor.id} className="awm-evidence" onClick={() => onSelect?.(anchor)}>
          <span>{anchor.label}</span>
          <small>{anchor.evidence_type.replaceAll('_', ' ')}</small>
          {anchor.slice_index !== null && anchor.slice_index !== undefined && (
            <small>Slice {anchor.slice_index}</small>
          )}
        </button>
      ))}
    </section>
  );
}
