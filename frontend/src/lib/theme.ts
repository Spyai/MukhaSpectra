import { useCallback, useSyncExternalStore } from 'react';

export type Theme = 'light' | 'dark';
const KEY = 'mukha-theme';
const listeners = new Set<() => void>();

function systemTheme(): Theme {
  if (typeof window === 'undefined' || !window.matchMedia) return 'dark';
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

function storedTheme(): Theme | null {
  try {
    const v = localStorage.getItem(KEY);
    return v === 'light' || v === 'dark' ? v : null;
  } catch {
    return null;
  }
}

export function currentTheme(): Theme {
  return storedTheme() ?? systemTheme();
}

export function applyTheme(theme: Theme = currentTheme()): void {
  document.documentElement.dataset.theme = theme;
  const meta = document.querySelector('meta[name="theme-color"]');
  meta?.setAttribute('content', theme === 'dark' ? '#0b0e1a' : '#f7f8fc');
}

/** Follows the OS preference until the person picks a theme with the toggle. */
export function initTheme(): void {
  applyTheme();
  if (typeof window === 'undefined' || !window.matchMedia) return;
  window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', () => {
    if (!storedTheme()) {
      applyTheme();
      listeners.forEach((l) => l());
    }
  });
}

export function useTheme(): [Theme, () => void] {
  const theme = useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    currentTheme,
    () => 'dark' as Theme,
  );
  const toggle = useCallback(() => {
    const next: Theme = currentTheme() === 'dark' ? 'light' : 'dark';
    try {
      localStorage.setItem(KEY, next);
    } catch {
      /* private mode: theme still applies for this page */
    }
    applyTheme(next);
    listeners.forEach((l) => l());
  }, []);
  return [theme, toggle];
}
