import React from 'react';

export default function PolicyDecisionPanel({ decision }) {
  if (!decision) return null;
  return (
    <section className={`awm-card policy-${decision.allowed ? 'allow' : 'deny'}`}>
      <div className="sidebar-section-title">Policy Decision</div>
      <span className={`badge ${decision.allowed ? 'badge-success' : 'badge-danger'}`}>
        {decision.decision}
      </span>
      <p>{decision.user_explanation}</p>
      <small>Trace: {decision.trace_id}</small>
    </section>
  );
}
