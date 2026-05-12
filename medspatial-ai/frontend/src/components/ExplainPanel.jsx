/**
 * MedSpatial AI — Explain Panel
 * Displays XAI reasoning chains for AI findings.
 */

import React, { useState, useCallback } from 'react';
import { explainScan, getReasoning } from '../services/api';

const CATEGORY_ICONS = {
  anomaly_evidence: '🔥',
  density_analysis: '📊',
  classification_evidence: '🏷️',
  anatomical_context: '📍',
};

const CATEGORY_COLORS = {
  anomaly_evidence: '#ef4444',
  density_analysis: '#06b6d4',
  classification_evidence: '#f59e0b',
  anatomical_context: '#10b981',
};

export default function ExplainPanel({ scanId, findings }) {
  const [reasoning, setReasoning] = useState(null);
  const [loading, setLoading] = useState(false);
  const [expandedChain, setExpandedChain] = useState(null);
  const [error, setError] = useState(null);

  const handleExplain = useCallback(async () => {
    if (!scanId) return;
    setLoading(true);
    setError(null);
    try {
      await explainScan(scanId);
      const data = await getReasoning(scanId);
      setReasoning(data.reasoning_chains || []);
    } catch (err) {
      console.error('Explain failed:', err);
      setError('Failed to generate explanations');
    } finally {
      setLoading(false);
    }
  }, [scanId]);

  if (!scanId || !findings || findings.length === 0) return null;

  return (
    <div className="explain-panel">
      <div className="explain-header">
        <span>🧠 AI Explainability</span>
        <button
          className="btn btn-sm btn-primary"
          onClick={handleExplain}
          disabled={loading}
          style={{ fontSize: 10, padding: '3px 8px' }}
        >
          {loading ? '⏳ Computing...' : '🔍 Explain Findings'}
        </button>
      </div>

      {error && (
        <div style={{ color: '#ef4444', fontSize: 11, padding: '6px 0' }}>{error}</div>
      )}

      {reasoning && reasoning.length > 0 && (
        <div className="explain-chains">
          {reasoning.map((chain, idx) => (
            <div key={idx} className="explain-chain">
              <div
                className="explain-chain-header"
                onClick={() => setExpandedChain(expandedChain === idx ? null : idx)}
              >
                <span className="explain-chain-toggle">
                  {expandedChain === idx ? '▼' : '▶'}
                </span>
                <span className="explain-chain-finding">
                  {chain.finding?.substring(0, 80)}
                  {chain.finding?.length > 80 ? '...' : ''}
                </span>
                <span className="explain-chain-conf">
                  {(chain.confidence * 100).toFixed(0)}%
                </span>
              </div>

              {expandedChain === idx && (
                <div className="explain-chain-body">
                  {/* Steps */}
                  {chain.steps && chain.steps.map((step, sIdx) => (
                    <div key={sIdx} className="explain-step">
                      <span className="explain-step-icon">
                        {CATEGORY_ICONS[step.category] || '📋'}
                      </span>
                      <div className="explain-step-content">
                        <div
                          className="explain-step-category"
                          style={{ color: CATEGORY_COLORS[step.category] || '#94a3b8' }}
                        >
                          {step.category?.replace(/_/g, ' ')}
                        </div>
                        <div className="explain-step-desc">{step.description}</div>
                      </div>
                    </div>
                  ))}

                  {/* Differential */}
                  {chain.differential && chain.differential.length > 0 && (
                    <div className="explain-differential">
                      <div className="explain-diff-title">Differential Diagnosis:</div>
                      {chain.differential.map((dx, dIdx) => (
                        <span key={dIdx} className="explain-diff-tag">{dx}</span>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      <style>{`
        .explain-panel { margin-top: 12px; }
        .explain-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: 12px;
          font-weight: 700;
          color: var(--text-primary, #f1f5f9);
          margin-bottom: 8px;
        }
        .explain-chains { display: flex; flex-direction: column; gap: 6px; }
        .explain-chain {
          border: 1px solid var(--border, #334155);
          border-radius: var(--radius-sm, 6px);
          overflow: hidden;
        }
        .explain-chain-header {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 6px 8px;
          cursor: pointer;
          background: var(--bg-tertiary, #1e293b);
          transition: background 0.2s;
        }
        .explain-chain-header:hover { background: var(--bg-secondary, #2a3a50); }
        .explain-chain-toggle { font-size: 8px; color: var(--text-muted); width: 10px; }
        .explain-chain-finding {
          flex: 1;
          font-size: 10px;
          color: var(--text-secondary, #94a3b8);
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        .explain-chain-conf {
          font-size: 10px;
          font-weight: 700;
          color: var(--accent, #6366f1);
          font-family: monospace;
        }
        .explain-chain-body { padding: 8px; }
        .explain-step {
          display: flex;
          gap: 8px;
          padding: 4px 0;
          border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .explain-step-icon { font-size: 14px; flex-shrink: 0; margin-top: 1px; }
        .explain-step-content { flex: 1; }
        .explain-step-category {
          font-size: 9px;
          text-transform: uppercase;
          font-weight: 700;
          letter-spacing: 0.5px;
        }
        .explain-step-desc {
          font-size: 10px;
          color: var(--text-secondary, #94a3b8);
          line-height: 1.4;
          margin-top: 1px;
        }
        .explain-differential {
          margin-top: 8px;
          padding-top: 6px;
          border-top: 1px solid rgba(255,255,255,0.08);
        }
        .explain-diff-title {
          font-size: 9px;
          font-weight: 700;
          color: var(--text-muted, #64748b);
          text-transform: uppercase;
          margin-bottom: 4px;
        }
        .explain-diff-tag {
          display: inline-block;
          padding: 2px 6px;
          margin: 2px;
          background: rgba(99,102,241,0.15);
          border: 1px solid rgba(99,102,241,0.25);
          border-radius: 10px;
          font-size: 9px;
          color: #818cf8;
        }
      `}</style>
    </div>
  );
}
