import { createApp } from './app.js';

const port = Number(process.env.WIKI_VIEWER_API_PORT ?? 5174);

createApp().listen(port, '127.0.0.1', () => {
  console.log(`wiki viewer api listening on http://127.0.0.1:${port}`);
});
