import type { ApiClient } from './types';

/** Mock build: load the in-memory SYNTHETIC mock API (see src/mocks/). */
export function loadMockClient(): Promise<ApiClient> {
  return import('../mocks/mockClient').then((m) => m.createMockClient());
}
