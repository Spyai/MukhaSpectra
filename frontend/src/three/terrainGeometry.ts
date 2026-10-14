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

/**
 * Terrain mesh from the relative-AF grid. The tissue outline is traced with marching squares
 * on a 3 × 3-smoothed tissue mask (iso 0.5), so the edge is a smooth contour instead of the
 * grid's staircase. Height = AF loss = 1 − relative AF.
 */
export function buildTerrainGeometry(grid: RelativeAfGrid): THREE.BufferGeometry {
  const { width: W, height: H, values } = grid;
  const depth = (WORLD_W * H) / W;
  const N = W * H;
  const valid = new Uint8Array(N);
  const loss = new Float32Array(N);
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      const v = values[j]?.[i];
      const ok = typeof v === 'number' && Number.isFinite(v);
      valid[j * W + i] = ok ? 1 : 0;
      loss[j * W + i] = ok ? Math.max(0, 1 - v) : 0;
    }
  }
  // Smoothed coverage (0..1) and a height for every vertex (holes borrow from neighbours).
  const cover = new Float32Array(N);
  const height = new Float32Array(N);
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      let n = 0;
      let inside = 0;
      let sum = 0;
      for (let dj = -1; dj <= 1; dj++) {
        for (let di = -1; di <= 1; di++) {
          const ii = Math.min(W - 1, Math.max(0, i + di));
          const jj = Math.min(H - 1, Math.max(0, j + dj));
          const k = jj * W + ii;
          n += 1;
          if (valid[k]) {
            inside += 1;
            sum += loss[k]!;
          }
        }
      }
      const k = j * W + i;
      cover[k] = inside / n;
      height[k] = valid[k] ? loss[k]! : inside > 0 ? sum / inside : 0;
    }
  }

  const positions: number[] = [];
  const colors: number[] = [];
  const c = new THREE.Color();
  const addVertex = (gx: number, gz: number, h: number): number => {
    positions.push((gx / (W - 1) - 0.5) * WORLD_W, h * HEIGHT_SCALE, (gz / (H - 1) - 0.5) * depth);
    lossColor(h, c);
    colors.push(c.r, c.g, c.b);
    return positions.length / 3 - 1;
  };
  const gridIndex = new Int32Array(N).fill(-1);
  const cornerVertex = (k: number) => {
    if (gridIndex[k]! < 0) gridIndex[k] = addVertex(k % W, Math.floor(k / W), height[k]!);
    return gridIndex[k]!;
  };
  const edgeVertices = new Map<string, number>();
  const edgeVertex = (p: number, q: number) => {
    const key = p < q ? `${p}:${q}` : `${q}:${p}`;
    const hit = edgeVertices.get(key);
    if (hit !== undefined) return hit;
    const cp = cover[p]!;
    const cq = cover[q]!;
    const t = cq === cp ? 0.5 : Math.min(1, Math.max(0, (0.5 - cp) / (cq - cp)));
    const px = p % W;
    const pz = Math.floor(p / W);
    const qx = q % W;
    const qz = Math.floor(q / W);
    const id = addVertex(
      px + (qx - px) * t,
      pz + (qz - pz) * t,
      height[p]! + (height[q]! - height[p]!) * t,
    );
    edgeVertices.set(key, id);
    return id;
  };

  const index: number[] = [];
  const inside = (k: number) => cover[k]! >= 0.5;
  for (let j = 0; j < H - 1; j++) {
    for (let i = 0; i < W - 1; i++) {
      const a = j * W + i;
      const b = a + 1;
      const d = a + W;
      const e = d + 1;
      // Corners in a consistent winding (normals point up): a → d → e → b.
      const ring = [a, d, e, b];
      const flags = ring.map(inside);
      const count = flags.filter(Boolean).length;
      if (count === 0) continue;
      const poly: number[] = [];
      for (let r = 0; r < 4; r++) {
        const p = ring[r]!;
        const q = ring[(r + 1) % 4]!;
        if (flags[r]) poly.push(cornerVertex(p));
        if (flags[r] !== flags[(r + 1) % 4]) poly.push(edgeVertex(p, q));
      }
      for (let f = 1; f < poly.length - 1; f++) index.push(poly[0]!, poly[f]!, poly[f + 1]!);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geo.setIndex(index);
  geo.computeVertexNormals();
  return geo;
}

/** Height (world units) of the terrain at a normalised image point, for 3D labels. */
export function heightAt(grid: RelativeAfGrid, x: number, y: number): number {
  const i = Math.min(grid.width - 1, Math.max(0, Math.round(x * (grid.width - 1))));
  const j = Math.min(grid.height - 1, Math.max(0, Math.round(y * (grid.height - 1))));
  const v = grid.values[j]?.[i];
  return typeof v === 'number' && Number.isFinite(v) ? Math.max(0, 1 - v) * HEIGHT_SCALE : 0;
}
