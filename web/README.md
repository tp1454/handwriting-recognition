# Web Client (Static)

This frontend is a separate static app that calls the FastAPI backend.

It supports two workflows:

1. Score an uploaded handwriting sheet (`POST /sheet/score`)
2. Create a new handwriting sheet with configurable layout (`GET /sheet/options`, `POST /sheet/create`)

## Run

1. Start backend:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

2. In another terminal, serve the web folder:

```bash
cd web
python3 -m http.server 5173
```

3. Open:

- http://localhost:5173

## Configure API URL

Edit `web/config.js` and set `apiBaseUrl`.

## Sheet Creation Notes

- Server fonts are discovered from `sheet.fonts_dir` in `config/default.yaml`.
- You can also upload a `.ttf` or `.otf` font directly from the web form.
- Generated files are served by the API at `/sheet/files/{filename}`.

## Run With Docker

From project root:

```bash
docker compose up -d
```

Open:

- http://localhost:5173

Useful commands:

```bash
docker compose logs -f web
docker compose logs -f api
docker compose down
```
