export const IMAGE_ACCEPT =
  'image/jpeg,image/png,image/heic,image/heif,image/tiff,.jpg,.jpeg,.png,.heic,.heif,.tif,.tiff,.dng';
export const SPECTRUM_ACCEPT = '.csv,.json,text/csv,application/json';
export const PREVIEWABLE = /\.(jpe?g|png|webp|gif|bmp)$/i;

export function isImageFile(f: File): boolean {
  return /\.(jpe?g|png|heic|heif|tiff?|dng|webp)$/i.test(f.name) || f.type.startsWith('image/');
}

export function isSpectrumFile(f: File): boolean {
  return /\.(csv|json)$/i.test(f.name) || f.type === 'text/csv' || f.type === 'application/json';
}

/** Keep the start and the informative end of long file names ("…tongue_right_wl.jpg"). */
export function middleTruncate(name: string, max = 44): string {
  if (name.length <= max) return name;
  const tail = Math.ceil(max * 0.55);
  const head = max - tail - 1;
  return `${name.slice(0, head)}…${name.slice(-tail)}`;
}
