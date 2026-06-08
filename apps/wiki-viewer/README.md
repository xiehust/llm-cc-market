# Wiki Viewer

A local web viewer for an [llm-wiki](../../llm-wiki) knowledge hub. It reads
your wiki directory at request time and serves:

- a **shelf** of topics with per-kind document groups,
- a **reader** that renders Markdown (relative wiki links navigate in-app),
- a **search** panel, and
- an interactive **knowledge graph** (pan / zoom, document focus).

The stack is a React + Vite frontend backed by a small Express API that
indexes the wiki on the fly.

## Requirements

- Node.js 20+ (developed on 22)
- An llm-wiki hub on disk (see [Wiki hub resolution](#wiki-hub-resolution))

```bash
npm install
```

## Run it in the background (single port)

`start.sh` builds the frontend (if needed) and serves the SPA **and** the API
from one process on `0.0.0.0:5175`.

```bash
./start.sh            # start (builds dist/ only if missing)
./start.sh --build    # force a fresh build, then start
./stop.sh             # stop
```

- PID is written to `.wiki-viewer.pid`, logs to `.wiki-viewer.log` (both gitignored).
- After changing frontend code, re-run `./start.sh --build` so the rebuilt
  `dist/` is served.

Override host/port with environment variables:

```bash
WIKI_VIEWER_HOST=0.0.0.0 WIKI_VIEWER_PORT=8080 ./start.sh
```

## Develop

Hot-reloading dev mode runs the Vite frontend and the API as two processes:

```bash
npm run dev
```

- Frontend: <http://127.0.0.1:5173> (Vite, with HMR)
- API: <http://127.0.0.1:5174> (proxied at `/api` by Vite; override with `WIKI_VIEWER_API_PORT`)

## Build & test

```bash
npm run build      # tsc typecheck + vite build -> dist/
npm test           # vitest (watch)
npm run test:run   # vitest (single run)
```

## Wiki hub resolution

The API locates the wiki hub in this order (first match wins):

1. `WIKI_HUB_PATH` environment variable
2. `hub_path` in `~/.config/llm-wiki/config.json`
3. `~/wiki` (default)

```bash
WIKI_HUB_PATH=/path/to/wiki ./start.sh
```

If the hub is missing the viewer renders a setup screen explaining which paths
were checked.

## Environment variables

| Variable | Default | Used by | Purpose |
| --- | --- | --- | --- |
| `WIKI_VIEWER_HOST` | `0.0.0.0` | `serve.ts` (start.sh) | Bind address for the combined server |
| `WIKI_VIEWER_PORT` | `5175` | `serve.ts` (start.sh) | Port for the combined server |
| `WIKI_VIEWER_API_PORT` | `5174` | `dev.ts` (`npm run dev`) | API port in dev mode |
| `WIKI_HUB_PATH` | — | API | Explicit wiki hub path |

## Layout

```
src/
  client/         React app (App, ShelfHome, TopicView, ReaderView, GraphView, SearchPanel, ...)
  server/
    app.ts        Express app factory (/api routes)
    dev.ts        Dev API entry (api only, 127.0.0.1:5174)
    serve.ts      Production entry (SPA + api on one port)
    wiki-index.ts Builds the in-memory wiki index
    graph.ts      Knowledge-graph construction
    search.ts     Search ranking
    hub-resolver.ts  Wiki hub path resolution
start.sh / stop.sh  Background process management
```
