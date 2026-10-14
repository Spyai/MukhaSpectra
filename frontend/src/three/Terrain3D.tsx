/**
 * 3D relative-AF terrain (lazy chunk). Height = AF loss = 1 − relative AF, so dark (FVL) areas
 * rise as peaks above a translucent plane drawn at the FVL threshold. Colours run from the green
 * glow of healthy mucosa to rose for deep glow loss; height carries the same information, so
 * colour is never the only cue.
 */
import { OrbitControls } from '@react-three/drei';
import { Canvas } from '@react-three/fiber';
import { useMemo } from 'react';
import * as THREE from 'three';
import type { RelativeAfGrid } from '../api/types';
import { buildTerrainGeometry, HEIGHT_SCALE, WORLD_W } from './terrainGeometry';

export interface Terrain3DProps {
  grid: RelativeAfGrid;
  threshold: number;
  reducedMotion: boolean;
  dark: boolean;
  label: string;
}

export default function Terrain3D({ grid, threshold, reducedMotion, dark, label }: Terrain3DProps) {
  const geometry = useMemo(() => buildTerrainGeometry(grid), [grid]);
  const depth = (WORLD_W * grid.height) / grid.width;
  const planeY = Math.max(0, 1 - threshold) * HEIGHT_SCALE;
  const planeEdges = useMemo(() => {
    const g = new THREE.EdgesGeometry(new THREE.PlaneGeometry(WORLD_W * 1.04, depth * 1.04));
    return g;
  }, [depth]);

  return (
    <div className="h-full w-full" role="img" aria-label={label}>
      <Canvas
        camera={{ position: [0.15, 1.15, 1.55], fov: 38, near: 0.01, far: 20 }}
        dpr={[1, 2]}
        frameloop={reducedMotion ? 'demand' : 'always'}
        gl={{ antialias: true, alpha: true }}
      >
        <ambientLight intensity={dark ? 0.55 : 0.75} />
        <hemisphereLight
          args={[dark ? '#a8a2ff' : '#ffffff', dark ? '#0b0e1a' : '#d9dbe8', 0.55]}
        />
        <directionalLight position={[1.6, 2.4, 1.2]} intensity={1.15} />
        <directionalLight position={[-1.8, 1.2, -1.5]} intensity={0.35} color="#8f7bff" />
        <mesh geometry={geometry}>
          <meshStandardMaterial
            vertexColors
            roughness={0.5}
            metalness={0.05}
            side={THREE.DoubleSide}
            flatShading={false}
          />
        </mesh>
        <group position={[0, planeY, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <mesh>
            <planeGeometry args={[WORLD_W * 1.04, depth * 1.04]} />
            <meshBasicMaterial
              color="#7c5cff"
              transparent
              opacity={0.16}
              side={THREE.DoubleSide}
              depthWrite={false}
            />
          </mesh>
          <lineSegments geometry={planeEdges}>
            <lineBasicMaterial color="#a48bff" transparent opacity={0.8} />
          </lineSegments>
        </group>
        <OrbitControls
          makeDefault
          enableDamping
          autoRotate={!reducedMotion}
          autoRotateSpeed={0.55}
          minDistance={0.9}
          maxDistance={4}
          maxPolarAngle={Math.PI / 2.05}
          target={[0, 0.08, 0]}
        />
      </Canvas>
    </div>
  );
}
