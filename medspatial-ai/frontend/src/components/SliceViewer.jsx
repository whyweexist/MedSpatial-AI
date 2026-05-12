/**
 * MedSpatial AI — Slice Viewer Component
 * 2D slice navigation (axial/coronal/sagittal) with scrubber.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { getSlice } from '../services/api';

export default function SliceViewer({ scanId, volumeDimensions }) {
  const [axis, setAxis] = useState('axial');
  const [sliceIndex, setSliceIndex] = useState(0);
  const [sliceData, setSliceData] = useState(null);
  const [totalSlices, setTotalSlices] = useState(0);
  const [loading, setLoading] = useState(false);

  const maxSlice = volumeDimensions
    ? (axis === 'axial' ? volumeDimensions.x :
       axis === 'coronal' ? volumeDimensions.y :
       volumeDimensions.z) - 1
    : 0;

  // Fetch slice when axis or index changes
  const fetchSlice = useCallback(async () => {
    if (!scanId) return;
    setLoading(true);
    try {
      const data = await getSlice(scanId, axis, sliceIndex);
      setSliceData(data.image_data);
      setTotalSlices(data.total_slices);
    } catch (e) {
      console.error('Slice fetch error:', e);
    }
    setLoading(false);
  }, [scanId, axis, sliceIndex]);

  useEffect(() => {
    fetchSlice();
  }, [fetchSlice]);

  // Reset index when axis changes
  useEffect(() => {
    setSliceIndex(Math.floor(maxSlice / 2));
  }, [axis, maxSlice]);

  if (!scanId || !volumeDimensions) {
    return (
      <div className="slice-viewer-floating">
        <div className="slice-viewer-header sidebar-section-title">Slice Viewer</div>
        <div style={{ textAlign: 'center', padding: 16, color: 'var(--text-muted)', fontSize: 12 }}>
          Reconstruct a scan to view slices
        </div>
      </div>
    );
  }

  return (
    <div className="slice-viewer-floating">
      <div className="slice-viewer-header sidebar-section-title">Slice Viewer</div>

      <div className="slice-viewer">
        {/* Slice image */}
        <div className="slice-canvas">
          {sliceData ? (
            <img
              src={`data:image/png;base64,${sliceData}`}
              alt={`${axis} slice ${sliceIndex}`}
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'contain',
                display: 'block'
              }}
            />
          ) : (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              color: 'var(--text-muted)',
              fontSize: 12,
            }}>
              {loading ? (
                <div className="spinner"></div>
              ) : (
                'No slice data'
              )}
            </div>
          )}

          {/* Axis label overlay */}
          <div style={{
            position: 'absolute',
            top: 8,
            left: 8,
            background: 'rgba(0,0,0,0.7)',
            padding: '2px 8px',
            borderRadius: 4,
            fontSize: 10,
            fontWeight: 600,
            color: '#818cf8',
            fontFamily: 'var(--font-mono)',
          }}>
            {axis.toUpperCase()} [{sliceIndex}/{totalSlices}]
          </div>
        </div>

        {/* Axis selection */}
        <div className="slice-controls">
          <div className="slice-axis-buttons">
            {['axial', 'coronal', 'sagittal'].map(ax => (
              <button
                key={ax}
                className={`slice-axis-btn ${axis === ax ? 'active' : ''}`}
                onClick={() => setAxis(ax)}
              >
                {ax.charAt(0).toUpperCase()}
              </button>
            ))}
          </div>

          {/* Slice slider */}
          <input
            type="range"
            className="slice-slider"
            min={0}
            max={maxSlice || 100}
            value={sliceIndex}
            onChange={(e) => setSliceIndex(parseInt(e.target.value))}
          />

          <span className="slice-number">
            {sliceIndex}
          </span>
        </div>
      </div>
    </div>
  );
}
