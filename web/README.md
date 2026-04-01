# Web Client (Static)

This frontend is a separate static app that calls the FastAPI backend.

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
