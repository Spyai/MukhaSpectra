/**
 * 3D relative-AF terrain (lazy chunk). Height = AF loss = 1 − relative AF, so dark (FVL) areas
 * rise as peaks above a translucent plane drawn at the FVL threshold. Colours run from the green
 * glow of healthy mucosa to rose for deep glow loss; height carries the same information, so
 * colour is never the only cue. Region markers (r1…) and a "top of photo" label keep the
 * terrain anchored to the 2D images.
 */
import { OrbitControls } from '@react-three/drei';
import { Canvas, useFrame } from '@react-three/fiber';
import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import type { FVLRegion, RelativeAfGrid } from '../api/types';
import { buildTerrainGeometry, heightAt, HEIGHT_SCALE, WORLD_W } from './terrainGeometry';

export interface Terrain3DProps {
  grid: RelativeAfGrid;
  threshold: number;
  reducedMotion: boolean;
  dark: boolean;
  label: string;
  regions?: FVLRegion[];
  topLabel?: string;
}

/** True while the element is on screen and the tab is visible (pauses the render loop). */
function useActive(ref: React.RefObject<HTMLElement | null>): boolean {
  const [inView, setInView] = useState(true);
  const [visible, setVisible] = useState(
    typeof document === 'undefined' || document.visibilityState !== 'hidden',
  );
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === 'undefined') return;
    const io = new IntersectionObserver(([entry]) => setInView(Boolean(entry?.isIntersecting)));
    io.observe(el);
    return () => io.disconnect();
  }, [ref]);
  useEffect(() => {
    const onVis = () => setVisible(document.visibilityState !== 'hidden');
    document.addEventListener('visibilitychange', onVis);
    return () => document.removeEventListener('visibilitychange', onVis);
  }, []);
  return inView && visible;
}

const labelClass =
  'pointer-events-none absolute top-0 left-0 select-none whitespace-nowrap rounded-full border border-white/80 bg-[oklch(0.25_0.05_290/0.85)] px-2 py-0.5 text-[11px] font-bold text-white shadow';

interface Anchor {
  key: string;
  text: string;
  pos: THREE.Vector3;
}

/**
 * Projects 3D anchor points to the canvas every rendered frame and moves plain DOM labels that
 * live OUTSIDE the canvas (no extra React root inside the scene, so unmounting stays clean).
 */
function LabelProjector({
  anchors,
  refs,
}: {
  anchors: Anchor[];
  refs: React.RefObject<Array<HTMLSpanElement | null>>;
}) {
  const v = useMemo(() => new THREE.Vector3(), []);
  useFrame(({ camera, size }) => {
    anchors.forEach((a, i) => {
      const el = refs.current?.[i];
      if (!el) return;
      v.copy(a.pos).project(camera);
      const x = (v.x * 0.5 + 0.5) * size.width;
      const y = (-v.y * 0.5 + 0.5) * size.height;
      el.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px) translate(-50%, -50%)`;
      el.style.opacity = v.z > -1 && v.z < 1 ? '1' : '0';
    });
  });
  return null;
}

export default function Terrain3D({
  grid,
  threshold,
  reducedMotion,
  dark,
  label,
  regions = [],
  topLabel,
}: Terrain3DProps) {
  const box = useRef<HTMLDivElement>(null);
  const active = useActive(box);
  const animate = active && !reducedMotion;
  const geometry = useMemo(() => buildTerrainGeometry(grid), [grid]);
  const depth = (WORLD_W * grid.height) / grid.width;
  const planeY = Math.max(0, 1 - threshold) * HEIGHT_SCALE;
  const planeEdges = useMemo(
    () => new THREE.EdgesGeometry(new THREE.PlaneGeometry(WORLD_W * 1.04, depth * 1.04)),
    [depth],
  );
  const anchors = useMemo<Anchor[]>(
    () => [
      ...(topLabel
        ? [
            {
              key: 'top',
              text: `↑ ${topLabel}`,
              pos: new THREE.Vector3(-WORLD_W * 0.32, 0.02, -depth / 2 - 0.1),
            },
          ]
        : []),
      ...regions.map((r) => ({
        key: r.region_id,
        text: r.region_id,
        pos: new THREE.Vector3(
          (r.centroid.x - 0.5) * WORLD_W,
          heightAt(grid, r.centroid.x, r.centroid.y) + 0.07,
          (r.centroid.y - 0.5) * depth,
        ),
      })),
    ],
    [topLabel, regions, grid, depth],
  );
  const labelRefs = useRef<Array<HTMLSpanElement | null>>([]);

  return (
    <div ref={box} className="relative h-full w-full" role="img" aria-label={label}>
      <Canvas
        camera={{ position: [0.15, 1.15, 1.55], fov: 38, near: 0.01, far: 20 }}
        dpr={[1, 2]}
        frameloop={animate ? 'always' : 'demand'}
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
        <LabelProjector anchors={anchors} refs={labelRefs} />
        <OrbitControls
          makeDefault
          enableDamping
          autoRotate={animate}
          autoRotateSpeed={0.55}
          minDistance={0.9}
          maxDistance={4}
          maxPolarAngle={Math.PI / 2.05}
          target={[0, 0.08, 0]}
        />
      </Canvas>
      {anchors.map((a, i) => (
        <span
          key={a.key}
          ref={(el) => {
            labelRefs.current[i] = el;
          }}
          className={labelClass}
          style={{ opacity: 0 }}
          aria-hidden="true"
          data-testid={`terrain-label-${a.key}`}
        >
          {a.text}
        </span>
      ))}
    </div>
  );
}
