import React from 'react';

const LABELS = {
  scan_derived: ['Scan-derived', 'badge-success'],
  estimated: ['Estimated / atlas-aligned', 'badge-warning'],
  synthetic: ['Synthetic / educational', 'badge-info'],
  user_authored: ['User annotation', 'badge-info'],
};

export default function ProvenanceBadge({ grounding = 'scan_derived' }) {
  const [label, className] = LABELS[grounding] || [grounding, 'badge-info'];
  return <span className={`badge ${className}`} title="Data provenance">{label}</span>;
}
