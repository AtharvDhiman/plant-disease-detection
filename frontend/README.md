# PlantGuard AI — frontend

React + Vite + Tailwind CSS single-page app for the plant disease detection
system. See the [project README](../README.md) for the whole system.

## Development

```bash
npm install
npm run dev
```

The dev server runs on <http://localhost:5173> and proxies `/api`, `/docs` and
`/redoc` to the FastAPI backend on port 8000, so the browser sees a single
origin and no CORS preflight is needed. Start the backend first:

```bash
uvicorn app.main:app --app-dir ../backend --reload
```

## Scripts

| Command | Purpose |
|---|---|
| `npm run dev` | Dev server with hot module replacement |
| `npm run build` | Production bundle into `dist/` |
| `npm run preview` | Serve the built bundle locally |
| `npm run lint` | oxlint over `src/` |

## Configuration

Copy `.env.example` to `.env.local` to override:

* `VITE_API_BASE_URL` — set only when the frontend is deployed separately from
  the API. Leave empty in development so requests stay same-origin.
* `VITE_API_TARGET` — where the dev-server proxy forwards `/api`.

## Structure

```
src/
├── components/       uploader, result, explainability, disease panel
│   ├── charts/       Recharts wrappers sharing one visual language
│   └── ui/           buttons, cards, chips, skeletons, tabs, toasts
├── pages/            Home, Dashboard, Benchmark, Research, Library, History
├── layouts/          AppLayout — navigation, theming, backend status
├── hooks/            useApi (load + error + reload), useTheme
├── services/api.js   API client; normalises the backend error envelope
├── utils/format.js   number/percent/date formatting and the shared palette
└── index.css         design tokens for both themes
```

## Design notes

* **Theme** — light and dark are defined as CSS variables on `:root` and
  `html.dark`. The stored preference is applied by an inline script in
  `index.html` before first paint, so the page never flashes the wrong theme.
* **Charts** read their colours from the same CSS variables, so they follow the
  theme without a second palette.
* **Confidence is never rounded up to 100%** — `percent()` renders `>99.99%`
  instead, because a classifier is never certain and displaying certainty it
  does not have is misleading.
* **No `backdrop-filter` on the sticky header** — it forces a full-page
  compositing layer on every scroll frame, which is a measurable cost on
  low-end mobile hardware.
