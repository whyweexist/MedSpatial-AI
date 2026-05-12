/**
 * MedSpatial AI — 3D Viewer Component (Enhanced)
 * Interactive Three.js volumetric renderer using React Three Fiber.
 * Supports complete dissection features: peeling, exploded views, multi-axis clipping,
 * floating anatomy labels, and XAI heatmap overlays.
 */

import React, { useRef, useState, useEffect, Suspense, useMemo } from 'react';
import { Canvas, useFrame, useLoader, useThree } from '@react-three/fiber';
import { OrbitControls, GizmoHelper, GizmoViewport, Grid, Html } from '@react-three/drei';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import AnatomyLabels from '../three/AnatomyLabels';

/* ── Layer Mesh Component with Dissection Support ───────────────── */

function LayerMesh({
  name,
  url,
  visible,
  opacity,
  color,
  clippingPlanes,
  dissectionOrder,
  peelDepth,
  isExploded,
  isIsolated,
  anyIsolated
}) {
  const gltf = useLoader(GLTFLoader, url);
  const meshRef = useRef();

  // Determine if this layer should be peeled away 
  // Outside layers vanish first as peelDepth 0 -> 1
  const orderThreshold = (dissectionOrder || 5) / 8; // normalize to 0-1
  const isPeeled = orderThreshold > (1 - peelDepth);
  
  // Actually render if visible (from toggle) AND not peeled AND (either nothing isolated or THIS is isolated)
  const shouldRender = visible && !isPeeled && (!anyIsolated || isIsolated);

  // Calculate exploded offset
  const explodedVector = useMemo(() => {
    if (!isExploded || name === 'primary') return new THREE.Vector3(0, 0, 0);
    // Displace outward based on name
    const offsets = {
      skin: [2, 0, 0],
      bone: [-1, 0, 1],
      left_lung: [-1, 0, -1],
      right_lung: [1, 0, -1],
      heart: [0, 0.5, 0.5],
      vessels: [0, -0.5, 0.5],
      soft_tissue: [1.5, 0, 0],
      pathology: [0, 1, 0],
      brain: [0, 1, 0],
      liver: [0, -1, 0],
      kidneys: [0, -1, 1],
    };
    const [x, y, z] = offsets[name] || [0, 0, 0];
    return new THREE.Vector3(x, y, z).multiplyScalar(0.5); // Spread factor
  }, [isExploded, name]);

  const targetOpacity = isIsolated ? 0.9 : opacity;

  const material = useMemo(() => {
    return new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(color),
      transparent: true,
      opacity: targetOpacity,
      roughness: 0.6,
      metalness: 0.05,
      side: THREE.DoubleSide,
      depthWrite: targetOpacity > 0.5,
      clippingPlanes: clippingPlanes || [],
      clipShadows: true,
    });
  }, [color, targetOpacity, clippingPlanes]);

  useFrame(() => {
    if (meshRef.current) {
      if (isExploded) {
        meshRef.current.position.lerp(explodedVector, 0.1);
      } else {
        meshRef.current.position.lerp(new THREE.Vector3(0, 0, 0), 0.1);
      }
    }
  });

  useEffect(() => {
    if (meshRef.current) {
      meshRef.current.traverse((child) => {
        if (child.isMesh) {
          child.material = material;
          child.castShadow = true;
          child.receiveShadow = true;
        }
      });
    }
  }, [material, gltf]);

  if (!shouldRender) return null;

  return (
    <primitive
      ref={meshRef}
      object={gltf.scene.clone()}
      scale={0.01}
    />
  );
}

/* ── Animated Scan Lines Effect ───────────────────────────────── */

function ScanEffect() {
  const meshRef = useRef();
  
  useFrame(({ clock }) => {
    if (meshRef.current) {
      meshRef.current.material.uniforms.time.value = clock.elapsedTime;
    }
  });

  const scanMaterial = useMemo(() => {
    return new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      uniforms: {
        time: { value: 0 },
        color: { value: new THREE.Color('#6366f1') },
      },
      vertexShader: `
        varying vec3 vPosition;
        void main() {
          vPosition = position;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform float time;
        uniform vec3 color;
        varying vec3 vPosition;
        void main() {
          float scan = sin(vPosition.y * 20.0 - time * 3.0) * 0.5 + 0.5;
          scan = smoothstep(0.4, 0.6, scan);
          float alpha = scan * 0.08;
          gl_FragColor = vec4(color, alpha);
        }
      `,
      side: THREE.BackSide,
    });
  }, []);

  return (
    <mesh ref={meshRef} material={scanMaterial}>
      <boxGeometry args={[4, 4, 4]} />
    </mesh>
  );
}

/* ── Findings Markers ─────────────────────────────────────────── */

function FindingMarker({ finding, index }) {
  const meshRef = useRef();
  const [hovered, setHovered] = useState(false);

  useFrame(({ clock }) => {
    if (meshRef.current) {
      meshRef.current.scale.setScalar(1 + Math.sin(clock.elapsedTime * 3 + index) * 0.15);
    }
  });

  if (!finding.location) return null;

  // Assume location is in bounded 0-100 or already normalized
  // Default mapping assuming normalization -50 to 50
  const position = [
    (finding.location.x / 100 - 0.5) * 2,
    (finding.location.z / 100 - 0.5) * 2,
    (finding.location.y / 100 - 0.5) * 2,
  ];

  const severityColor = {
    normal: '#10b981',
    mild: '#f59e0b',
    moderate: '#f97316',
    severe: '#ef4444',
    critical: '#dc2626',
  }[finding.severity] || '#f59e0b';

  return (
    <group position={position}>
      <mesh
        ref={meshRef}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <sphereGeometry args={[0.05, 16, 16]} />
        <meshBasicMaterial color={severityColor} transparent opacity={0.8} />
      </mesh>
      {/* Outer ring */}
      <mesh>
        <ringGeometry args={[0.06, 0.08, 32]} />
        <meshBasicMaterial color={severityColor} transparent opacity={0.4} side={THREE.DoubleSide} />
      </mesh>
      {hovered && (
        <Html distanceFactor={5} position={[0, 0.15, 0]} zIndexRange={[100, 0]}>
          <div style={{
            background: 'rgba(17,24,39,0.95)',
            padding: '8px 12px',
            borderRadius: '8px',
            border: `1px solid ${severityColor}`,
            fontSize: '11px',
            color: '#f1f5f9',
            width: '200px',
            pointerEvents: 'none',
            backdropFilter: 'blur(4px)',
          }}>
            <div style={{ fontWeight: 600, color: severityColor, marginBottom: 4 }}>
              {finding.severity?.toUpperCase()} ({(finding.confidence * 100).toFixed(0)}%)
            </div>
            <div>{finding.region}</div>
          </div>
        </Html>
      )}
    </group>
  );
}

/* ── Scene Setup ──────────────────────────────────────────────── */

function SceneSetup() {
  const { gl } = useThree();
  
  useEffect(() => {
    gl.localClippingEnabled = true;
    gl.toneMapping = THREE.ACESFilmicToneMapping;
    gl.toneMappingExposure = 1.2;
  }, [gl]);

  return null;
}

/* ── Loading Fallback ─────────────────────────────────────────── */

function LoadingFallback() {
  return (
    <Html center>
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '12px',
      }}>
        <div className="spinner" style={{ width: 32, height: 32, borderWidth: 3 }}></div>
        <div style={{ color: '#94a3b8', fontSize: 13, background: 'rgba(15,23,42,0.8)', padding: '4px 12px', borderRadius: 16 }}>Loading 3D Meshes...</div>
      </div>
    </Html>
  );
}

/* ── Main Viewer Component ────────────────────────────────────── */

export default function Viewer3D({ 
  meshUrl, 
  layerUrls, 
  layers, 
  showHeatmap, 
  findings,
  segments,
  peelDepth = 0,
  isExploded = false,
  isolatedSegment = null,
  clipAxis = null,
  anatomyLabels = [],
  showLabels = true,
}) {
  const [clipVal, setClipVal] = useState(5);
  
  const clippingPlanes = useMemo(() => {
    if (!clipAxis || clipVal >= 4.9) return [];
    
    // Create plane based on selected axis
    if (clipAxis === 'x') return [new THREE.Plane(new THREE.Vector3(-1, 0, 0), clipVal)];
    if (clipAxis === 'y') return [new THREE.Plane(new THREE.Vector3(0, -1, 0), clipVal)];
    if (clipAxis === 'z') return [new THREE.Plane(new THREE.Vector3(0, 0, -1), clipVal)];
    
    return [new THREE.Plane(new THREE.Vector3(0, -1, 0), clipVal)];
  }, [clipAxis, clipVal]);

  return (
    <div className="viewer-canvas" id="viewer-3d">
      <Canvas
        camera={{ position: [2.5, 2, 2.5], fov: 50, near: 0.01, far: 100 }}
        shadows
        gl={{ antialias: true, alpha: false }}
        style={{ background: 'linear-gradient(to bottom, #0a0e1a, #0f172a)' }}
      >
        <SceneSetup />

        {/* Lighting */}
        <ambientLight intensity={0.4} />
        <directionalLight position={[5, 8, 5]} intensity={1.2} castShadow shadow-mapSize={1024} />
        <directionalLight position={[-3, 4, -3]} intensity={0.5} color="#818cf8" />
        <pointLight position={[0, -3, 0]} intensity={0.2} color="#06b6d4" />

        {/* Grid */}
        <Grid
          args={[10, 10]}
          position={[0, -1.5, 0]}
          cellSize={0.5}
          cellThickness={0.5}
          cellColor="#1e293b"
          sectionSize={2}
          sectionThickness={1}
          sectionColor="#334155"
          fadeDistance={8}
          fadeStrength={1}
          infiniteGrid
        />

        {/* Scan effect */}
        <ScanEffect />

        {/* Meshes */}
        <Suspense fallback={<LoadingFallback />}>
          {/* Primary mesh (if layers not heavily used or for fallback) */}
          {meshUrl && layers.primary?.visible && !isExploded && peelDepth === 0 && (
            <LayerMesh
              name="primary"
              url={meshUrl}
              visible={true}
              opacity={layers.primary.opacity}
              color={layers.primary.color}
              clippingPlanes={clippingPlanes}
            />
          )}

          {/* Layer meshes */}
          {Object.entries(layerUrls).map(([name, url]) => {
            const layerConfig = layers[name];
            // Find segment config for ordering
            const segmentInfo = (segments || []).find(s => s.name === name);
            
            if (!layerConfig || !url) return null;
            return (
              <LayerMesh
                key={name}
                name={name}
                url={url}
                visible={layerConfig.visible}
                opacity={layerConfig.opacity}
                color={layerConfig.color}
                clippingPlanes={clippingPlanes}
                dissectionOrder={segmentInfo?.dissection_order || 5}
                peelDepth={peelDepth}
                isExploded={isExploded}
                isIsolated={name === isolatedSegment}
                anyIsolated={isolatedSegment !== null}
              />
            );
          })}

          {/* Anatomy Labels */}
          {showLabels && <AnatomyLabels labels={anatomyLabels} visible={!isExploded} />}

          {/* Finding markers */}
          {showHeatmap && findings && findings.map((finding, i) => (
            <FindingMarker key={i} finding={finding} index={i} />
          ))}
        </Suspense>

        {/* Controls */}
        <OrbitControls
          makeDefault
          enableDamping
          dampingFactor={0.08}
          minDistance={0.5}
          maxDistance={10}
          enablePan
        />

        {/* Gizmo */}
        <GizmoHelper alignment="bottom-right" margin={[60, 60]}>
          <GizmoViewport
            axisColors={['#ef4444', '#10b981', '#3b82f6']}
            labelColor="white"
          />
        </GizmoHelper>
      </Canvas>

      {/* Cross Section Slider */}
      {clipAxis && (
        <div style={{
          position: 'absolute',
          right: 16,
          top: '50%',
          transform: 'translateY(-50%)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 8,
          background: 'rgba(15,23,42,0.8)',
          padding: '16px 8px',
          borderRadius: 20,
          backdropFilter: 'blur(8px)',
          border: '1px solid rgba(255,255,255,0.1)',
        }}>
          <span style={{ fontSize: 10, color: '#94a3b8', writingMode: 'vertical-rl', fontWeight: 700, letterSpacing: 1 }}>
            {clipAxis.toUpperCase()} CROSS-SECTION
          </span>
          <input
            type="range"
            min={-2}
            max={5}
            step={0.05}
            value={clipVal}
            onChange={(e) => setClipVal(parseFloat(e.target.value))}
            className="dissection-slider"
            style={{
              writingMode: 'vertical-lr',
              direction: 'rtl',
              height: 150,
              width: 6,
            }}
          />
        </div>
      )}
    </div>
  );
}
