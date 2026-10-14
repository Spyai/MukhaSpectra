/** Small chart helpers shared by SVG visualisations. */

export const SPECTRAL_CHANNELS: Array<[string, number]> = [
  ['F1_415', 415],
  ['F2_445', 445],
  ['F3_480', 480],
  ['F4_515', 515],
  ['F5_555', 555],
  ['F6_590', 590],
  ['F7_630', 630],
  ['F8_680', 680],
];

export type Pt = { x: number; y: number };

/** Monotone cubic (Fritsch–Carlson) path: smooth, never overshoots the measured points. */
export function monotonePath(pts: Pt[]): string {
  const n = pts.length;
  if (n === 0) return '';
  if (n === 1) return `M${pts[0]!.x},${pts[0]!.y}`;
  const dx: number[] = [];
  const slope: number[] = [];
  for (let i = 0; i < n - 1; i++) {
    dx.push(pts[i + 1]!.x - pts[i]!.x);
    slope.push((pts[i + 1]!.y - pts[i]!.y) / dx[i]!);
  }
  const tan: number[] = [slope[0]!];
  for (let i = 1; i < n - 1; i++) {
    const a = slope[i - 1]!;
    const b = slope[i]!;
    tan.push(
      a * b <= 0
        ? 0
        : (3 * (dx[i - 1]! + dx[i]!)) /
            ((2 * dx[i]! + dx[i - 1]!) / a + (dx[i]! + 2 * dx[i - 1]!) / b),
    );
  }
  tan.push(slope[n - 2]!);
  let d = `M${pts[0]!.x.toFixed(2)},${pts[0]!.y.toFixed(2)}`;
  for (let i = 0; i < n - 1; i++) {
    const p0 = pts[i]!;
    const p1 = pts[i + 1]!;
    const h = dx[i]! / 3;
    d += ` C${(p0.x + h).toFixed(2)},${(p0.y + h * tan[i]!).toFixed(2)} ${(p1.x - h).toFixed(2)},${(p1.y - h * tan[i + 1]!).toFixed(2)} ${p1.x.toFixed(2)},${p1.y.toFixed(2)}`;
  }
  return d;
}
