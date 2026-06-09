import React from 'react';

export default function SceneGraphPanel({ graph }) {
  if (!graph) return null;
  const labels = Object.fromEntries((graph.nodes || []).map(node => [node.id, node.label]));
  return (
    <section className="awm-card">
      <div className="sidebar-section-title">Anatomical Scene Graph</div>
      <div className="awm-graph-stats">
        <span>{graph.nodes?.length || 0} nodes</span>
        <span>{graph.edges?.length || 0} relations</span>
      </div>
      {(graph.edges || []).slice(0, 12).map(edge => (
        <div className="awm-relation" key={edge.id}>
          <strong>{labels[edge.source_id] || edge.source_id}</strong>
          <span>{edge.relation.replaceAll('_', ' ')}</span>
          <strong>{labels[edge.target_id] || edge.target_id}</strong>
        </div>
      ))}
    </section>
  );
}
