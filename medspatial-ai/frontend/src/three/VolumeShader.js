/**
 * MedSpatial AI — GPU Volume Raycasting Shader
 * Complete GLSL implementation per program.md Section 3.
 *
 * Features:
 *  - Ray marching through THREE.Data3DTexture
 *  - Per-layer transfer functions (density → color + opacity)
 *  - Dissection slider: layers peeled from outside in
 *  - Clip plane: arbitrary cutting plane
 *  - Anomaly overlay: pulsing red heatmap
 *  - Phong shading via gradient-estimated normals (central differences)
 *  - Maximum Intensity Projection (MIP) mode
 *  - Adaptive step size (quality vs performance)
 */

export const VolumeVertexShader = /* glsl */`
varying vec3 vRayOrigin;
varying vec3 vModelPos;

void main() {
  vModelPos = position;
  vec4 worldPos = modelMatrix * vec4(position, 1.0);
  vRayOrigin = (inverse(modelMatrix) * vec4(cameraPosition, 1.0)).xyz;
  gl_Position = projectionMatrix * viewMatrix * worldPos;
}
`;

export const VolumeFragmentShader = /* glsl */`
precision highp float;
precision highp sampler3D;

// ── Volume textures ──────────────────────────────────────────────
uniform sampler3D u_volume;       // Raw density (HU normalized 0-1)
uniform sampler3D u_segmentation; // Label (0-11) as float/12
uniform sampler3D u_anomaly;      // Anomaly heatmap [0,1]

// ── Rendering control ────────────────────────────────────────────
uniform float u_dissectionDepth;  // 0=full anatomy, 1=everything visible
uniform vec4  u_clipPlane;        // (nx,ny,nz, d) world-space clip plane
uniform float u_time;             // Elapsed seconds for pulse animation
uniform int   u_mipMode;          // 0 = compositing, 1 = MIP
uniform float u_stepSize;         // Ray step size (adaptive)
uniform float u_globalOpacity;    // Master opacity multiplier

// ── Layer transfer functions ─────────────────────────────────────
// Layer order (outside-in): 1(skin) 2(muscle) 3(bone) 4(pleura)
//                            5(left lung) 6(right lung) 7(airways)
//                            8(vessels) 9(heart) 10(mediastinum) 11(pathology)
uniform vec3  u_layerColors[12];     // RGB per layer
uniform float u_layerOpacity[12];    // Opacity per layer
uniform bool  u_layerVisible[12];    // Visibility flags
uniform float u_layerOrder[12];      // Layer depth order for dissection

// ── Anomaly overlay ──────────────────────────────────────────────
uniform float u_anomalyWeight;    // How strongly to show anomaly
uniform bool  u_showAnomaly;

varying vec3 vRayOrigin;
varying vec3 vModelPos;

// ── AABB ray intersection ────────────────────────────────────────
vec2 intersectAABB(vec3 ro, vec3 rd) {
  vec3 invRd = 1.0 / rd;
  vec3 t0 = (-0.5 - ro) * invRd;
  vec3 t1 = ( 0.5 - ro) * invRd;
  vec3 tmin = min(t0, t1);
  vec3 tmax = max(t0, t1);
  float tNear = max(max(tmin.x, tmin.y), tmin.z);
  float tFar  = min(min(tmax.x, tmax.y), tmax.z);
  return vec2(tNear, tFar);
}

// ── Sample volume at position ────────────────────────────────────
float sampleDensity(vec3 pos) {
  vec3 uv = pos + 0.5; // [-0.5,0.5] → [0,1]
  return texture(u_volume, uv).r;
}

int sampleLabel(vec3 pos) {
  vec3 uv = pos + 0.5;
  return int(texture(u_segmentation, uv).r * 11.0 + 0.5);
}

float sampleAnomaly(vec3 pos) {
  vec3 uv = pos + 0.5;
  return texture(u_anomaly, uv).r;
}

// ── Gradient-estimated normal via central differences ────────────
vec3 computeNormal(vec3 pos) {
  float eps = 0.005;
  float dx = sampleDensity(pos + vec3(eps,0,0)) - sampleDensity(pos - vec3(eps,0,0));
  float dy = sampleDensity(pos + vec3(0,eps,0)) - sampleDensity(pos - vec3(0,eps,0));
  float dz = sampleDensity(pos + vec3(0,0,eps)) - sampleDensity(pos - vec3(0,0,eps));
  return normalize(vec3(dx, dy, dz));
}

// ── Phong shading ────────────────────────────────────────────────
vec3 phongShading(vec3 pos, vec3 normal, vec3 baseColor, vec3 lightDir) {
  vec3 N = normal;
  if (dot(N, -lightDir) < 0.0) N = -N;
  float diffuse  = max(dot(N, -lightDir), 0.0);
  vec3 viewDir   = normalize(cameraPosition - pos);
  vec3 halfVec   = normalize(-lightDir + viewDir);
  float specular = pow(max(dot(N, halfVec), 0.0), 32.0);
  vec3 ambient   = 0.2 * baseColor;
  return ambient + 0.7 * diffuse * baseColor + 0.1 * specular * vec3(1.0);
}

// ── Main fragment program ────────────────────────────────────────
void main() {
  vec3 ro = vRayOrigin;
  vec3 rd = normalize(vModelPos - vRayOrigin);

  // Clip plane
  // (we test each sample position against the clip plane)

  vec2 tHit = intersectAABB(ro, rd);
  if (tHit.x >= tHit.y) discard;

  float tStart = max(tHit.x, 0.001);
  float tEnd   = tHit.y;

  float dt        = u_stepSize;
  vec4  accColor  = vec4(0.0);
  float mipMax    = 0.0;
  vec3  mipColor  = vec3(0.0);
  vec3  lightDir  = normalize(vec3(-1.0, -2.0, -1.5));

  float t = tStart;
  for (int i = 0; i < 512; i++) {
    if (t >= tEnd) break;

    vec3 pos = ro + rd * t;

    // ── Clip plane test ────────────────────────────────────────
    vec3 worldPos = (modelMatrix * vec4(pos, 1.0)).xyz;
    if (dot(worldPos, u_clipPlane.xyz) + u_clipPlane.w < 0.0) {
      t += dt;
      continue;
    }

    // ── Sample data ─────────────────────────────────────────
    float density = sampleDensity(pos);
    int   label   = sampleLabel(pos);
    float anom    = u_showAnomaly ? sampleAnomaly(pos) : 0.0;

    // ── MIP mode ───────────────────────────────────────────
    if (u_mipMode == 1) {
      if (density > mipMax) {
        mipMax   = density;
        mipColor = mix(vec3(0.0, 0.0, 0.5), vec3(1.0, 1.0, 0.0), density);
      }
      t += dt;
      continue;
    }

    // ── Dissection depth culling ───────────────────────────
    // Discard if this layer's depth order > dissection depth
    if (label >= 0 && label < 12) {
      float layerDepthOrder = u_layerOrder[label] / 11.0;
      if (layerDepthOrder > u_dissectionDepth) {
        t += dt;
        continue;
      }
      if (!u_layerVisible[label]) {
        t += dt;
        continue;
      }
    }

    // ── Transfer function lookup ───────────────────────────
    vec3  layerColor   = (label >= 0 && label < 12) ? u_layerColors[label]  : vec3(0.5);
    float layerOpacity = (label >= 0 && label < 12) ? u_layerOpacity[label] : 0.0;

    // Density-modulated opacity (empty voxels transparent)
    float alpha = layerOpacity * u_globalOpacity * clamp(density * 2.0, 0.0, 1.0);

    if (alpha < 0.001) {
      t += dt;
      continue;
    }

    // ── Phong shading ─────────────────────────────────────
    vec3 normal   = computeNormal(pos);
    vec3 shadedColor = phongShading(worldPos, normal, layerColor, lightDir);

    // ── Anomaly overlay (pulsing red) ─────────────────────
    if (u_showAnomaly && anom > 0.3) {
      float pulse = sin(u_time * 3.0) * 0.5 + 0.5;
      vec4 anomColor = vec4(1.0, 0.15, 0.0, 0.5 * pulse);
      float anomWeight = anom * u_anomalyWeight;
      shadedColor = mix(shadedColor, anomColor.rgb, anomWeight);
      alpha = max(alpha, anomColor.a * anomWeight);
    }

    // ── Front-to-back compositing ─────────────────────────
    vec4 sample4 = vec4(shadedColor, alpha);
    accColor.rgb += (1.0 - accColor.a) * sample4.a * sample4.rgb;
    accColor.a   += (1.0 - accColor.a) * sample4.a;

    if (accColor.a >= 0.98) break;

    t += dt;
  }

  // ── MIP final output ───────────────────────────────────────
  if (u_mipMode == 1) {
    gl_FragColor = vec4(mipColor, mipMax > 0.0 ? 1.0 : 0.0);
    return;
  }

  if (accColor.a < 0.01) discard;
  gl_FragColor = accColor;
}
`;

// ── Default transfer function ────────────────────────────────────
export const DEFAULT_LAYER_COLORS = [
  [0.05, 0.05, 0.08],  //  0 Background
  [0.90, 0.75, 0.65],  //  1 Skin
  [0.75, 0.50, 0.40],  //  2 Musculature
  [0.95, 0.92, 0.80],  //  3 Bone
  [0.70, 0.85, 0.90],  //  4 Pleura
  [0.40, 0.65, 0.85],  //  5 Left Lung
  [0.30, 0.55, 0.80],  //  6 Right Lung
  [0.80, 0.85, 0.90],  //  7 Airways
  [0.85, 0.20, 0.20],  //  8 Vasculature
  [0.90, 0.60, 0.60],  //  9 Heart
  [0.70, 0.70, 0.85],  // 10 Mediastinum
  [1.00, 0.15, 0.00],  // 11 Pathology
];

export const DEFAULT_LAYER_OPACITIES = [
  0.0,  // 0 Background
  0.4,  // 1 Skin
  0.5,  // 2 Musculature
  0.8,  // 3 Bone
  0.3,  // 4 Pleura
  0.35, // 5 Left Lung
  0.35, // 6 Right Lung
  0.6,  // 7 Airways
  0.7,  // 8 Vasculature
  0.7,  // 9 Heart
  0.5,  // 10 Mediastinum
  0.9,  // 11 Pathology
];

// Outside-in dissection order (0 = innermost, 11 = outermost)
export const LAYER_DISSECTION_ORDER = [0, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
