# Deployment (CPU-first)

## Local production mode (Windows)
```powershell
cd D:\Projects\StyleFindAI
docker compose up -d                      # PostgreSQL + pgvector (healthcheck included)
powershell -ExecutionPolicy Bypass -File scripts\start-prod.ps1
```
One Node process serves the built React app **and** the API on port 4000; the AI service listens only on
127.0.0.1:8001 (never expose it). Health: `http://localhost:4000/api/health` (DB + AI, `outdated_features`).

## Before going online - checklist
| item | how |
|---|---|
| Secrets | `.env`: new random `JWT_SECRET`, `AI_INTERNAL_TOKEN`, `POSTGRES_PASSWORD` (24+ chars). With `NODE_ENV=production` the backend refuses to start with defaults. |
| HTTPS | put a reverse proxy in front of port 4000 (example below). Cookies get `Secure` automatically in production. |
| CORS | `CORS_ORIGINS=https://your-domain` (only needed if the site is served from another origin). |
| Database port | do not publish 5432 to the internet (remove `ports:` in docker-compose or bind to `127.0.0.1:5432:5432`). |
| Backups | `scripts\backup-db.ps1` daily (Task Scheduler) + copy `dataset\products`; test `restore-db.ps1` once. |
| Logs | backend logs method/url/status/time/request-id (no bodies, no cookies); AI service logs errors with details server-side only. Correlate with the `X-Request-Id` response header. |
| Monitoring | poll `/api/health` (uptime checker); watch `avgSearchLatencyMs` and "Products that are not searchable" in the admin dashboard. |
| Models | first start downloads FashionCLIP (~600 MB) into `.model-cache`; pre-download by running one search before opening to users. |
| Memory | FashionCLIP on CPU ~1.5 GB RSS; keep `AI_MAX_LOADED_MODELS=1`. 4 GB RAM total is a practical minimum for one instance. |

## HTTPS reverse proxy example (Caddy, automatic certificates)
```
your-domain.com {
    encode gzip
    reverse_proxy 127.0.0.1:4000
}
```
Set `app.set("trust proxy", ...)` is already "loopback", which is correct when the proxy runs on the same machine.

## Rate limits
search 30/min/IP, login 10/15 min/IP, API 300/min/IP (`backend/src/middleware/rateLimit.ts`).
