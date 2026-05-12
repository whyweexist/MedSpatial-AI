/**
 * MedSpatial AI — Anatomy Labels Component
 * Renders 3D floating labels pointing to anatomical centroids.
 */

import React, { useRef, useState } from 'react';
import { Html, Line } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

export default function AnatomyLabels({ labels, visible }) {
  if (!visible || !labels || labels.length === 0) return null;

  return (
    <group>
      {labels.map((label, idx) => (
        <AnatomyLabel key={idx} data={label} />
      ))}
    </group>
  );
}

function AnatomyLabel({ data }) {
  const [hovered, setHovered] = useState(false);
  const targetRef = useRef();

  // Position is {x, y, z} in viewer coords mapped from volume coords
  const position = [data.position.x, data.position.y, data.position.z];

  // Colors
  const [r, g, b] = data.color || [0.5, 0.5, 0.5];
  const colorHex = `#${Math.round(r*255).toString(16).padStart(2,'0')}${Math.round(g*255).toString(16).padStart(2,'0')}${Math.round(b*255).toString(16).padStart(2,'0')}`;

  // Calculate an offset for the label to float away from the actual body part
  // Usually offset outward based on position from center
  const offsetDistance = 0.5;
  const dir = new THREE.Vector3(...position).normalize();
  if (dir.lengthSq() < 0.1) dir.set(0, 1, 0); // fallback if exact center
  const labelPos = [
    position[0] + dir.x * offsetDistance,
    position[1] + dir.y * offsetDistance + 0.2, // slightly up
    position[2] + dir.z * offsetDistance,
  ];

  return (
    <group>
      {/* Target Dot */}
      <mesh position={position} ref={targetRef}>
        <sphereGeometry args={[0.02, 16, 16]} />
        <meshBasicMaterial color={colorHex} transparent opacity={0.6} depthTest={false} />
      </mesh>

      {/* Leader Line */}
      <Line
        points={[position, labelPos]}
        color={colorHex}
        lineWidth={1.5}
        transparent
        opacity={0.4}
        depthTest={false}
      />

      {/* Floating HTML Label */}
      <Html position={labelPos} center distanceFactor={10} zIndexRange={[100, 0]}>
        <div
          className="anatomy-label"
          onMouseEnter={() => setHovered(true)}
          onMouseLeave={() => setHovered(false)}
          style={{
            background: hovered ? 'rgba(30, 41, 59, 0.95)' : 'rgba(15, 23, 42, 0.75)',
            border: `1px solid ${colorHex}`,
            borderRadius: '4px',
            padding: '4px 8px',
            color: '#f8fafc',
            fontSize: '10px',
            fontWeight: 600,
            whiteSpace: 'nowrap',
            pointerEvents: 'auto',
            cursor: 'default',
            boxShadow: hovered ? `0 4px 12px rgba(0,0,0,0.5), 0 0 8px ${colorHex}55` : 'none',
            transform: hovered ? 'scale(1.1)' : 'scale(1)',
            transition: 'all 0.2s',
            backdropFilter: 'blur(4px)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: colorHex, display: 'inline-block' }}></span>
            {data.name}
          </div>
          {hovered && data.description && (
            <div style={{ 
              marginTop: '4px', 
              fontSize: '8px', 
              color: '#94a3b8', 
              fontWeight: 400,
              maxWidth: '120px',
              whiteSpace: 'normal',
              lineHeight: '1.2'
            }}>
              {data.description}
            </div>
          )}
        </div>
      </Html>
    </group>
  );
}
