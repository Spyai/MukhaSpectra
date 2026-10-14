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
import { LOSS_STOPS } from '../lib/colormap';

const HEIGHT_SCALE = 0.55;
const WORLD_W = 1.6;

const STOPS: Array<[number, THREE.Color]> = LOSS_STOPS.map(([p, hex]) => [p, new THREE.Color(hex)]);

export function lossColor(loss: number, out = new THREE.Color()): THREE.Color {
  const v = Math.min(1, Math.max(0, loss));
  for (let i = 1; i < STOPS.length; i++) {
    const [p1, c1] = STOPS[i]!;
    const [p0, c0] = STOPS[i - 1]!;
    if (v <= p1) return out.copy(c0).lerp(c1, (v - p0) / (p1 - p0));
  }
  return out.copy(STOPS[STOPS.length - 1]![1]);
}

export function buildTerrainGeometry(grid: RelativeAfGrid): THREE.BufferGeometry {
  const { width: W, height: H, values } = grid;
  const depth = (WORLD_W * H) / W;
  const positions = new Float32Array(W * H * 3);
  const colors = new Float32Array(W * H * 3);
  const valid = new Uint8Array(W * H);
  const c = new THREE.Color();
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      const k = j * W + i;
      const v = values[j]?.[i];
      const ok = typeof v === 'number' && Number.isFinite(v);
      valid[k] = ok ? 1 : 0;
      const loss = ok ? Math.max(0, 1 - v) : 0;
      positions[k * 3] = (i / (W - 1) - 0.5) * WORLD_W;
      positions[k * 3 + 1] = loss * HEIGHT_SCALE;
      positions[k * 3 + 2] = (j / (H - 1) - 0.5) * depth;
      lossColor(loss, c);
      colors[k * 3] = c.r;
      colors[k * 3 + 1] = c.g;
      colors[k * 3 + 2] = c.b;
    }
  }
  const index: number[] = [];
  for (let j = 0; j < H - 1; j++) {
    for (let i = 0; i < W - 1; i++) {
      const a = j * W + i;
      const b = a + 1;
      const d = a + W;
      const e = d + 1;
      if (valid[a] && valid[d] && valid[b]) index.push(a, d, b);
      if (valid[b] && valid[d] && valid[e]) index.push(b, d, e);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geo.setIndex(index);
  geo.computeVertexNormals();
  return geo;
}

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
        <hemisphereLight args={[dark ? '#a8a2ff' : '#ffffff', dark ? '#0b0e1a' : '#d9dbe8', 0.55]} />
        <directionalLight position={[1.6, 2.4, 1.2]} intensity={1.15} />
        <directionalLight position={[-1.8, 1.2, -1.5]} intensity={0.35} color="#8f7bff" />
        <mesh geometry={geometry}>
          <meshStandardMaterial vertexColors roughness={0.5} metalness={0.05} side={THREE.DoubleSide} flatShading={false} />
        </mesh>
        <group position={[0, planeY, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <mesh>
            <planeGeometry args={[WORLD_W * 1.04, depth * 1.04]} />
            <meshBasicMaterial color="#7c5cff" transparent opacity={0.16} side={THREE.DoubleSide} depthWrite={false} />
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
