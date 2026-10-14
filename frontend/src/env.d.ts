/// <reference types="vite/client" />

/** True when built or served with VITE_MOCK=1 (see vite.config.ts `define`). */
declare const __MUKHA_MOCK__: boolean;

interface ImportMetaEnv {
  readonly VITE_MOCK?: string;
}

declare module '@mock-loader' {
  import type { ApiClient } from './api/types';
  export function loadMockClient(): Promise<ApiClient>;
}
