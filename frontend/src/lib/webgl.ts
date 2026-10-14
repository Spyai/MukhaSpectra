let cached: boolean | null = null;

/** True when a WebGL context can be created (cached). */
export function hasWebGL(): boolean {
  if (cached !== null) return cached;
  try {
    const canvas = document.createElement('canvas');
    const ctx =
      canvas.getContext('webgl2') ??
      canvas.getContext('webgl') ??
      canvas.getContext('experimental-webgl');
    cached = Boolean(ctx);
  } catch {
    cached = false;
  }
  return cached;
}
