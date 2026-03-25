/**
 * MedSpatial AI — Volume Renderer (Three.js)
 * Integrates the GLSL volume shader with THREE.Data3DTexture
 * for GPU-accelerated raycasting directly in the browser.
 */

import * as THREE from 'three';
import {
  VolumeVertexShader,
  VolumeFragmentShader,
  DEFAULT_LAYER_COLORS,
  DEFAULT_LAYER_OPACITIES,
  LAYER_DISSECTION_ORDER,
} from './VolumeShader';

export class VolumeRenderer {
  constructor(scene, renderer) {
    this.scene = scene;
    this.renderer = renderer;

    // Enable local clipping
    renderer.localClippingEnabled = true;

    this.volumeTexture    = null;
    this.segTexture       = null;
    this.anomalyTexture   = null;
    this.mesh             = null;
    this.material         = null;
    this.clock            = new THREE.Clock();

    // State
    this.dissectionDepth  = 1.0;     // 0=skin only, 1=full
    this.mipMode          = false;
    this.showAnomaly      = false;
    this.anomalyWeight    = 0.8;
    this.clipPlane        = new THREE.Vector4(0, 0, 0, 1e6); // disabled

    this.layerColors      = DEFAULT_LAYER_COLORS.map(c => new THREE.Color(...c));
    this.layerOpacities   = [...DEFAULT_LAYER_OPACITIES];
    this.layerVisible     = new Array(12).fill(true);

    this._buildMesh();
  }

  _buildMesh() {
    // Placeholder 1×1×1 Data3DTexture until volume data arrives
    const placeholder = new THREE.Data3DTexture(new Float32Array([0]), 1, 1, 1);
    placeholder.format = THREE.RedFormat;
    placeholder.type   = THREE.FloatType;
    placeholder.needsUpdate = true;

    this.material = new THREE.ShaderMaterial({
      vertexShader:   VolumeVertexShader,
      fragmentShader: VolumeFragmentShader,
      transparent:    true,
      depthWrite:     false,
      side:           THREE.BackSide,
      uniforms: {
        u_volume:         { value: placeholder },
        u_segmentation:   { value: placeholder },
        u_anomaly:        { value: placeholder },
        u_dissectionDepth:{ value: 1.0 },
        u_clipPlane:      { value: this.clipPlane },
        u_time:           { value: 0.0 },
        u_mipMode:        { value: 0 },
        u_stepSize:       { value: 1.0 / 128.0 },
        u_globalOpacity:  { value: 1.0 },
        u_layerColors:    { value: this.layerColors },
        u_layerOpacity:   { value: this.layerOpacities },
        u_layerVisible:   { value: this.layerVisible },
        u_layerOrder:     { value: LAYER_DISSECTION_ORDER },
        u_anomalyWeight:  { value: this.anomalyWeight },
        u_showAnomaly:    { value: false },
      },
    });

    const geometry = new THREE.BoxGeometry(1, 1, 1);
    this.mesh = new THREE.Mesh(geometry, this.material);
    this.scene.add(this.mesh);
  }

  /**
   * Load volume data from a Float32Array into a 3D texture.
   * @param {Float32Array} data - flat volumetric data (density)
   * @param {number} D - depth
   * @param {number} H - height
   * @param {number} W - width
   */
  loadVolume(data, D, H, W) {
    if (this.volumeTexture) this.volumeTexture.dispose();

    this.volumeTexture = new THREE.Data3DTexture(data, W, H, D);
    this.volumeTexture.format     = THREE.RedFormat;
    this.volumeTexture.type       = THREE.FloatType;
    this.volumeTexture.minFilter  = THREE.LinearFilter;
    this.volumeTexture.magFilter  = THREE.LinearFilter;
    this.volumeTexture.wrapS      = THREE.ClampToEdgeWrapping;
    this.volumeTexture.wrapT      = THREE.ClampToEdgeWrapping;
    this.volumeTexture.wrapR      = THREE.ClampToEdgeWrapping;
    this.volumeTexture.needsUpdate = true;

    this.material.uniforms.u_volume.value = this.volumeTexture;

    // Scale mesh preserving aspect ratio
    const maxDim = Math.max(D, H, W);
    this.mesh.scale.set(W / maxDim, H / maxDim, D / maxDim);
  }

  /** Load segmentation label volume. */
  loadSegmentation(data, D, H, W) {
    if (this.segTexture) this.segTexture.dispose();
    this.segTexture = new THREE.Data3DTexture(data, W, H, D);
    this.segTexture.format = THREE.RedFormat;
    this.segTexture.type   = THREE.FloatType;
    this.segTexture.minFilter = THREE.NearestFilter;
    this.segTexture.magFilter = THREE.NearestFilter;
    this.segTexture.needsUpdate = true;
    this.material.uniforms.u_segmentation.value = this.segTexture;
  }

  /** Load anomaly heatmap volume. */
  loadAnomaly(data, D, H, W) {
    if (this.anomalyTexture) this.anomalyTexture.dispose();
    this.anomalyTexture = new THREE.Data3DTexture(data, W, H, D);
    this.anomalyTexture.format    = THREE.RedFormat;
    this.anomalyTexture.type      = THREE.FloatType;
    this.anomalyTexture.minFilter = THREE.LinearFilter;
    this.anomalyTexture.magFilter = THREE.LinearFilter;
    this.anomalyTexture.needsUpdate = true;
    this.material.uniforms.u_anomaly.value = this.anomalyTexture;
  }

  /** Set dissection depth [0..1] — progressive outside-in peel. */
  setDissectionDepth(depth) {
    this.dissectionDepth = depth;
    this.material.uniforms.u_dissectionDepth.value = depth;
  }

  /** Toggle MIP (Maximum Intensity Projection) rendering. */
  setMIPMode(enabled) {
    this.mipMode = enabled;
    this.material.uniforms.u_mipMode.value = enabled ? 1 : 0;
  }

  /** Set anomaly overlay visibility. */
  setShowAnomaly(show) {
    this.showAnomaly = show;
    this.material.uniforms.u_showAnomaly.value = show;
  }

  /** Set clip plane in world space as {nx,ny,nz,d}. */
  setClipPlane(nx, ny, nz, d) {
    this.clipPlane.set(nx, ny, nz, d);
    this.material.uniforms.u_clipPlane.value = this.clipPlane;
  }

  /** Disable the clip plane. */
  clearClipPlane() {
    this.setClipPlane(0, 0, 0, 1e6);
  }

  /** Toggle/set individual layer visibility and opacity. */
  setLayerVisible(layerIdx, visible) {
    if (layerIdx < 0 || layerIdx >= 12) return;
    this.layerVisible[layerIdx] = visible;
    this.material.uniforms.u_layerVisible.value = this.layerVisible;
    this.material.needsUpdate = true;
  }

  setLayerOpacity(layerIdx, opacity) {
    if (layerIdx < 0 || layerIdx >= 12) return;
    this.layerOpacities[layerIdx] = opacity;
    this.material.uniforms.u_layerOpacity.value = this.layerOpacities;
    this.material.needsUpdate = true;
  }

  /** Adapt step size: 'quality' = 1/128, 'performance' = 1/64. */
  setQuality(mode) {
    this.material.uniforms.u_stepSize.value = mode === 'quality' ? 1/128 : 1/64;
  }

  /** Call every frame to update time uniform (pulsing anomaly). */
  update() {
    if (this.material) {
      this.material.uniforms.u_time.value = this.clock.getElapsedTime();
    }
  }

  dispose() {
    this.volumeTexture?.dispose();
    this.segTexture?.dispose();
    this.anomalyTexture?.dispose();
    this.material?.dispose();
    this.mesh?.geometry.dispose();
    this.scene.remove(this.mesh);
  }
}
