/// <reference types="vitest/config" />
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { fileURLToPath } from 'node:url';

const API_TARGET = process.env.MUKHA_API_TARGET ?? 'http://localhost:8000';

const apiProxy = {
  '/api': {
    target: API_TARGET,
    changeOrigin: true,
  },
};

export default defineConfig(({ mode }) => {
  const env = { ...loadEnv(mode, process.cwd(), 'VITE_'), ...process.env };
  const mock = env.VITE_MOCK === '1' || env.VITE_MOCK === 'true';
  return {
    plugins: [react(), tailwindcss()],
    // Compile-time flag: in a live build the mock client (and its SYNTHETIC assets) is dead code
    // and is dropped from the bundle entirely.
    define: {
      __MUKHA_MOCK__: JSON.stringify(mock),
    },
    resolve: {
      alias: {
        '@mock-loader': fileURLToPath(
          new URL(
            mock ? './src/api/mockLoader.ts' : './src/api/mockLoader.stub.ts',
            import.meta.url,
          ),
        ),
      },
    },
    server: {
      port: 5173,
      proxy: apiProxy,
    },
    preview: {
      port: 4173,
      proxy: apiProxy,
    },
    build: {
      target: 'es2022',
      sourcemap: false,
      chunkSizeWarningLimit: 1200,
      // Never inline PDFs as data: URLs (browsers block opening them in a new tab).
      assetsInlineLimit: (file: string) => (file.endsWith('.pdf') ? false : undefined),
    },
    test: {
      globals: true,
      environment: 'jsdom',
      setupFiles: ['./src/test/setup.ts'],
      css: false,
      include: ['src/**/*.test.{ts,tsx}'],
      restoreMocks: true,
    },
  };
});
