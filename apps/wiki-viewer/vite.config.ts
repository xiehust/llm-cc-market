/// <reference types="vitest" />

import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:5174',
    },
  },
  // Vitest reads this key, but Vite's config type does not include it.
  // @ts-expect-error Vitest-only config property.
  test: {
    environment: 'jsdom',
    globals: true,
  },
});
