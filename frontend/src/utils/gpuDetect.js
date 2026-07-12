/**
 * MedSpatial AI — GPU + Platform Detector
 * Client-side detection of WebGL2 capabilities and GPU vendor
 * for adaptive quality selection.
 */

export function detectGPU() {
  const result = {
    webgl2: false,
    vendor: 'unknown',
    renderer: 'unknown',
    isIntegrated: false,
    quality: 'low',                  // 'low' | 'medium' | 'high'
    recommendedVolumeSize: 64,
    frameTarget: 30,
  };

  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2');
    if (!gl) return result;

    result.webgl2 = true;

    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
    if (debugInfo) {
      result.vendor   = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) || 'unknown';
      result.renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) || 'unknown';
    }

    const renderer = result.renderer.toLowerCase();

    // Detect integrated GPU patterns
    const integratedPatterns = [
      'intel', 'iris', 'uhd', 'hd graphics',
      'adreno', 'mali', 'apple m1', 'apple m2',
      'llvmpipe', 'swiftshader',
    ];
    result.isIntegrated = integratedPatterns.some(p => renderer.includes(p));

    // Detect discrete GPU patterns
    const highEndPatterns = [
      'rtx', 'gtx 10', 'gtx 16', 'gtx 20',
      'rx 5', 'rx 6', 'rx 7', 'vega',
      'quadro', 'tesla',
    ];
    const isHighEnd = highEndPatterns.some(p => renderer.includes(p));

    if (isHighEnd) {
      result.quality = 'high';
      result.recommendedVolumeSize = 128;
      result.frameTarget = 60;
    } else if (!result.isIntegrated) {
      result.quality = 'medium';
      result.recommendedVolumeSize = 128;
      result.frameTarget = 60;
    } else {
      result.quality = 'low';
      result.recommendedVolumeSize = 64;
      result.frameTarget = 30;
    }

    // Extra check: max 3D texture size
    const max3D = gl.getParameter(gl.MAX_3D_TEXTURE_SIZE);
    if (max3D < 128) {
      result.recommendedVolumeSize = Math.min(result.recommendedVolumeSize, max3D);
    }

    canvas.remove();
  } catch (e) {
    console.warn('GPU detection error:', e);
  }

  return result;
}

/** Return adaptive step size for ray marcher. */
export function getAdaptiveStepSize(gpuInfo) {
  return gpuInfo.quality === 'high' ? 1 / 128 : 1 / 64;
}
