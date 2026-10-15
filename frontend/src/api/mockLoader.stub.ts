import type { ApiClient } from './types';

/**
 * Live build: the mock API is not bundled. vite.config.ts aliases `@mock-loader` to this stub
 * unless VITE_MOCK=1, so no mock code or SYNTHETIC asset ends up in a live build.
 */
export function loadMockClient(): Promise<ApiClient> {
  return Promise.reject(new Error('Mock mode is not enabled in this build (set VITE_MOCK=1).'));
}
