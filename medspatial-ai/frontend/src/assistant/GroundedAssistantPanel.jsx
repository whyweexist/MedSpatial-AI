import React, { useState } from 'react';
import { askGroundedQuestion } from '../world/AWMClient';
import EvidenceOverlay from '../viewer/EvidenceOverlay';
import ProvenanceBadge from '../viewer/ProvenanceBadge';
import UncertaintyOverlay from '../viewer/UncertaintyOverlay';

export default function GroundedAssistantPanel({ studyId, onEvidenceSelect }) {
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  async function submit(event) {
    event.preventDefault();
    if (!studyId || !question.trim()) return;
    setLoading(true);
    try {
      setResult(await askGroundedQuestion(studyId, question));
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="awm-card">
      <div className="sidebar-section-title">Grounded Assistant</div>
      <form onSubmit={submit} className="awm-question">
        <textarea value={question} onChange={event => setQuestion(event.target.value)}
          placeholder="Ask about evidence in this study" />
        <button className="btn btn-primary btn-sm" disabled={loading || !question.trim()}>
          {loading ? 'Checking evidence...' : 'Ask'}
        </button>
      </form>
      {result && (
        <>
          <p className="awm-answer">{result.answer}</p>
          <div className="awm-badges">
            <ProvenanceBadge grounding={result.grounding} />
            <span className="badge badge-info">{Math.round(result.confidence * 100)}% confidence</span>
          </div>
          <UncertaintyOverlay uncertainty={result.uncertainty} />
          <EvidenceOverlay anchors={result.evidence_anchors} onSelect={onEvidenceSelect} />
        </>
      )}
    </section>
  );
}
