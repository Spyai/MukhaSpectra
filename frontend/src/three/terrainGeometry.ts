import * as THREE from 'three';
import type { RelativeAfGrid } from '../api/types';
import { LOSS_STOPS } from '../lib/colormap';

export const HEIGHT_SCALE = 0.55;
export const WORLD_W = 1.6;

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
