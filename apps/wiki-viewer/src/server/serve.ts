import express from 'express';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createApp } from './app.js';

const host = process.env.WIKI_VIEWER_HOST ?? '0.0.0.0';
const port = Number(process.env.WIKI_VIEWER_PORT ?? 5175);

// dist/ is produced by `vite build` (apps/wiki-viewer/dist).
const distDir = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'dist');
const indexHtml = join(distDir, 'index.html');

if (!existsSync(indexHtml)) {
  console.error(`Built frontend not found at ${distDir}. Run "npm run build" first.`);
  process.exit(1);
}

const app = createApp();

// createApp() already owns /api/* (including its own 404). Everything else is
// served from the built SPA, with an index.html fallback for client-side routing.
app.use(express.static(distDir));
app.get('*', (_req, res) => {
  res.sendFile(indexHtml);
});

app.listen(port, host, () => {
  console.log(`wiki viewer listening on http://${host}:${port} (pid ${process.pid})`);
});
