/**
 * MedSpatial AI — Upload Panel Component
 * Drag-and-drop DICOM upload with progress tracking.
 */

import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { uploadDicomFiles, getScan } from '../services/api';

export default function UploadPanel({ onUploadComplete }) {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);

  const onDrop = useCallback(async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;

    setUploading(true);
    setProgress(0);
    setError(null);

    try {
      const result = await uploadDicomFiles(acceptedFiles, (pct) => setProgress(pct));
      
      // Fetch full scan data
      const scan = await getScan(result.scan_id);
      onUploadComplete(scan);
      setProgress(100);

      // Reset after delay
      setTimeout(() => {
        setUploading(false);
        setProgress(0);
      }, 1500);
    } catch (err) {
      console.error('Upload failed:', err);
      setError(err.response?.data?.detail || 'Upload failed. Please try again.');
      setUploading(false);
    }
  }, [onUploadComplete]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/dicom': ['.dcm', '.dicom'],
      'application/octet-stream': ['.dcm'],
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/png': ['.png'],
      'image/tiff': ['.tif', '.tiff'],
      'image/bmp': ['.bmp'],
    },
    multiple: true,
    disabled: uploading,
  });

  return (
    <div className="sidebar-section">
      <div className="sidebar-section-title">
        📤 Upload Scan
      </div>

      <div
        {...getRootProps()}
        className={`upload-zone ${isDragActive ? 'drag-active' : ''}`}
        id="upload-zone"
        style={{ padding: uploading ? '16px' : '24px' }}
      >
        <input {...getInputProps()} id="dicom-upload-input" />

        {uploading ? (
          <div>
            <div style={{ fontSize: 28, marginBottom: 8 }}>
              {progress >= 100 ? '✅' : '⏳'}
            </div>
            <div className="upload-zone-text">
              {progress >= 100 ? 'Upload Complete!' : `Uploading... ${progress}%`}
            </div>
            <div className="upload-progress">
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${progress}%` }}></div>
              </div>
            </div>
          </div>
        ) : (
          <>
            <div className="upload-zone-icon">📁</div>
            <div className="upload-zone-text">
              {isDragActive ? 'Drop files here' : 'Drag & drop DICOM or Image files'}
            </div>
            <div className="upload-zone-hint">
              or click to select · .dcm, .png, .jpg, .tiff
            </div>
          </>
        )}
      </div>

      {error && (
        <div style={{
          marginTop: 8,
          padding: '8px 12px',
          background: 'rgba(239,68,68,0.1)',
          border: '1px solid rgba(239,68,68,0.2)',
          borderRadius: 6,
          fontSize: 12,
          color: '#ef4444',
        }}>
          {error}
        </div>
      )}
    </div>
  );
}
