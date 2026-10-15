/** AF-loss colour map shared by the 3D terrain (lazy chunk) and its HTML legend. */
export const LOSS_STOPS: ReadonlyArray<readonly [number, string]> = [
  [0.0, '#19c37d'],
  [0.18, '#7fd65a'],
  [0.3, '#e7cf4a'],
  [0.45, '#f39a3d'],
  [0.62, '#e9506e'],
  [1.0, '#7b1f63'],
];

export const LOSS_GRADIENT_CSS = `linear-gradient(90deg, ${LOSS_STOPS.map(([p, c]) => `${c} ${p * 100}%`).join(', ')})`;
