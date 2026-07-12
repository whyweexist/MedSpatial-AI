/**
 * MedSpatial AI — Report Button
 * Dropdown button for PDF / DOCX report download.
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { downloadReport } from '../services/api';

export default function ReportButton({ scanId }) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(null);
  const ref = useRef(null);

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const handleDownload = useCallback(async (format) => {
    if (!scanId) return;
    setLoading(format);
    try {
      await downloadReport(scanId, format);
    } catch (err) {
      console.error('Report download failed:', err);
      alert(`Failed to generate ${format.toUpperCase()} report. Ensure reportlab/python-docx are installed.`);
    } finally {
      setLoading(null);
      setOpen(false);
    }
  }, [scanId]);

  if (!scanId) return null;

  return (
    <div className="report-button-wrap" ref={ref}>
      <button
        className="btn btn-sm btn-secondary"
        onClick={() => setOpen(!open)}
      >
        📄 Report {open ? '▲' : '▼'}
      </button>

      {open && (
        <div className="report-dropdown">
          <button
            className="report-dropdown-item"
            onClick={() => handleDownload('pdf')}
            disabled={loading === 'pdf'}
          >
            {loading === 'pdf' ? '⏳' : '📕'} Download PDF
          </button>
          <button
            className="report-dropdown-item"
            onClick={() => handleDownload('docx')}
            disabled={loading === 'docx'}
          >
            {loading === 'docx' ? '⏳' : '📘'} Download DOCX
          </button>
        </div>
      )}

      <style>{`
        .report-button-wrap { position: relative; display: inline-block; }
        .report-dropdown {
          position: absolute;
          top: calc(100% + 4px);
          right: 0;
          min-width: 160px;
          background: var(--bg-secondary, #1e293b);
          border: 1px solid var(--border, #334155);
          border-radius: var(--radius-sm, 6px);
          box-shadow: 0 8px 24px rgba(0,0,0,0.4);
          z-index: 100;
          overflow: hidden;
        }
        .report-dropdown-item {
          display: block;
          width: 100%;
          padding: 8px 12px;
          border: none;
          background: transparent;
          color: var(--text-primary, #f1f5f9);
          font-size: 12px;
          cursor: pointer;
          text-align: left;
          transition: background 0.15s;
        }
        .report-dropdown-item:hover {
          background: rgba(99,102,241,0.15);
        }
        .report-dropdown-item:disabled {
          opacity: 0.5;
          cursor: wait;
        }
      `}</style>
    </div>
  );
}
