import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const apiTarget = loadEnv(mode, '.', '').VITE_API_PROXY ?? 'http://localhost:8080';
  const wsTarget = apiTarget.replace(/^http/, 'ws');
  return {
    plugins: [react()],
    server: { proxy: { '/api': apiTarget, '/ws': { target: wsTarget, ws: true } } },
  };
});
