import { useGLTF } from "@react-three/drei";
import { useRef } from "react";
import { useFrame } from "@react-three/fiber";

export default function BrainModel({ scrollY }) {
  const { scene } = useGLTF("/brain.glb");
  const ref = useRef();

  useFrame(() => {
    if (ref.current) {
      // scroll-based rotation
      ref.current.rotation.y = scrollY.current * 0.005;
    //   ref.current.rotation.x = scrollY.current * 0.001;
    }
  });

  return (
    <primitive
      ref={ref}
      object={scene}
      scale={0.023}   // 🔥 adjust if model looks too big/small
    />
  );
}