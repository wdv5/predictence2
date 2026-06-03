# Predictive Maintenance Agent — Phase 2 (ML)

Phase 2 replaces every in-memory, simulated, and rule-based component with
production-grade counterparts while keeping the **API contract identical** to
Phase 1 — no dashboard changes required.

---

## What changed

| Component | Phase 1 | Phase 2 |
|-----------|---------|---------|
| **Anomaly detection** | Hard-coded IF/THEN thresholds (`rules_engine.py`) | Isolation Forest (`ml/anomaly_detector.py`) |
| **Metrics source** | Synthetic simulator (`prometheus_sim.py`) | Live Prometheus scrape (`prometheus_scraper.py`) |
| **State store** | In-memory deques (`state_store.py`) | TimescaleDB + Redis (`state_store.py`) |
| **Dashboard** | Phase-1 panels | + ML training progress + Prometheus scrape button |

All three Phase-1 components include **graceful fallbacks**:
- Isolation Forest falls back to rule-based evaluation until 50 samples collected.
- `prometheus_scraper.scrape()` falls back to `generate()` when Prometheus is unreachable.
- `state_store` falls back to in-memory deques when TimescaleDB / Redis are down.

---

## Directory structure (Phase 2 additions)

```
backend/
  ml/
    __init__.py
    anomaly_detector.py       ← Isolation Forest singleton
  core/
    rules_engine.py           ← evaluate() now delegates to detector
    prometheus_scraper.py     ← live scrape + sim fallback (replaces prometheus_sim.py)
    state_store.py            ← TimescaleDB + Redis (replaces in-memory store)
  api/
    metrics.py                ← adds /metrics/scrape, /metrics/ml/status
docker/
  docker-compose.yml          ← TimescaleDB, Redis, Prometheus, Node Exporter, n8n
  prometheus.yml              ← Prometheus scrape config
  Dockerfile                  ← FastAPI container
dashboard/
  index.html                  ← adds ML status panel + Scrape Prometheus button
```

---

## Quick start

### Option A — Docker Compose (recommended)

```bash
# From the project root
docker compose -f docker/docker-compose.yml up -d --build

# Optional: with n8n
docker compose -f docker/docker-compose.yml --profile n8n up -d --build
```

If the backend container keeps restarting with `Could not import module "backend.main"`
after pulling code changes, rebuild the image so Docker does not reuse the old
copy that lacked `backend/main.py`:

```bash
docker compose -f docker/docker-compose.yml down
docker compose -f docker/docker-compose.yml build --no-cache backend
docker compose -f docker/docker-compose.yml up -d
```

Services:
| Service | Port | URL |
|---------|------|-----|
| FastAPI backend | 8000 | http://localhost:8000 |
| Dashboard | 3000 | http://localhost:3000 |
| TimescaleDB | 5432 | postgresql://localhost:5432/predictence |
| Redis | 6379 | redis://localhost:6379 |
| Prometheus | 9090 | http://localhost:9090 |
| Node Exporter | 9100 | http://localhost:9100/metrics |
| n8n (optional) | 5678 | http://localhost:5678 |

### Option B — Local (no Docker)

```bash
# 1. Install dependencies (adds scikit-learn, prometheus-api-client, psycopg2, redis)
pip install -r backend/requirements.txt

# 2. Start the backend (backends gracefully fall back when DBs aren't running)
uvicorn backend.main:app --reload --port 8000

# 3. Serve dashboard
cd dashboard && python -m http.server 3000
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/predictence` | TimescaleDB DSN |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `PROMETHEUS_URL` | `http://localhost:9090` | Prometheus base URL |
| `PROMETHEUS_TIMEOUT` | `10` | Per-request timeout (seconds) |
| `PROMETHEUS_QUERY_CPU` | node_cpu PromQL | Override CPU query |
| `PROMETHEUS_QUERY_RAM` | node_memory PromQL | Override RAM query |
| `PROMETHEUS_QUERY_LATENCY` | http_request_duration PromQL | Override latency query |
| `PROMETHEUS_QUERY_ERROR_RATE` | http_requests_total PromQL | Override error rate query |
| `N8N_ENABLED` | `false` | Enable n8n webhook delegation |
| `N8N_WEBHOOK_URL` | `http://localhost:5678/webhook/agent-alert` | Alert handler webhook |
| `N8N_PREDICTIVE_WEBHOOK_URL` | `http://localhost:5678/webhook/cpu-threshold-forecast` | Forecast webhook |

---

## New API endpoints

All Phase-1 endpoints remain unchanged. Phase 2 adds:

### `GET /metrics/scrape`
Pull current metrics from live Prometheus. Falls back to a normal simulation
sample when Prometheus is unreachable.

```bash
curl http://localhost:8000/metrics/scrape | python -m json.tool
```

### `GET /metrics/ml/status`
Inspect the Isolation Forest model's training progress.

```bash
curl.exe http://localhost:8000/metrics/ml/status -o status.json
py -m json.tool status.json
```

Response:
```json
{
  "trained": false,
  "training_samples": 23,
  "min_samples_needed": 50,
  "mode": "rule_fallback"
}
```

Once `trained` becomes `true`, `mode` switches to `"isolation_forest"` and
anomaly detection uses the ML model instead of hard-coded thresholds.

---

## How the Isolation Forest detector works

```
                        ┌──────────────────────────────────┐
   MetricPayload ──────►│  anomaly_detector.ingest(payload)│
                        │  • appends to rolling history     │
                        │  • retrains every 25 new samples  │
                        └───────────────┬──────────────────┘
                                        │
                        ┌───────────────▼──────────────────┐
   alerts ◄─────────────│  anomaly_detector.score(payload) │
                        │  IF trained:                      │
                        │    score_samples() → IF score     │
                        │    score > -0.10  → OK            │
                        │    score > -0.25  → WARNING       │
                        │    score ≤ -0.25  → CRITICAL      │
                        │  ELSE: rule_evaluate() fallback   │
                        └──────────────────────────────────┘
```

**Key parameters** (in `ml/anomaly_detector.py`):

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `MIN_SAMPLES_TO_TRAIN` | 50 | Observations needed before IF activates |
| `RETRAIN_EVERY` | 25 | Retrain frequency after initial training |
| `CONTAMINATION` | 0.05 | Expected anomaly fraction (~5%) |
| `N_ESTIMATORS` | 200 | Number of isolation trees |
| `WARNING_SCORE_THRESHOLD` | -0.10 | IF score below this → WARNING |
| `CRITICAL_SCORE_THRESHOLD` | -0.25 | IF score below this → CRITICAL |

Tune `CONTAMINATION` to match your system's actual anomaly rate.
Lower values (e.g. 0.01) make the model less sensitive; higher values (0.10)
flag more data points as anomalous.

---

## Bootstrapping the ML model

The fastest way to get the Isolation Forest trained is to run the existing
bootstrap script and then hit `/metrics/ingest` with each row:

```bash
# 1. Generate 240 synthetic historical rows
python backend/scripts/bootstrap_prophet_data.py \
  --out backend/data/metrics.csv \
  --points 240

# 2. Re-ingest them so the detector accumulates training data
python - <<'EOF'
import csv, httpx
with open("backend/data/metrics.csv") as f:
    for row in list(csv.DictReader(f))[:100]:
        httpx.post("http://localhost:8000/metrics/ingest", json={
            "cpu_percent": float(row["cpu_percent"]),
            "ram_percent": float(row["ram_percent"]),
            "latency_ms":  float(row["latency_ms"]),
            "error_rate":  float(row["error_rate"]),
        })
print("Done — check /metrics/ml/status")
EOF
```

Or simply run the **Auto** simulation for ~30 seconds (sends ~15 samples at
2 s/sample) and the model trains live.

---

## TimescaleDB schema (auto-created)

```sql
-- Continuous metrics — partitioned by time
CREATE TABLE metrics (
    id          BIGSERIAL,
    ts          TIMESTAMPTZ NOT NULL,
    cpu_percent DOUBLE PRECISION,
    ram_percent DOUBLE PRECISION,
    latency_ms  DOUBLE PRECISION,
    error_rate  DOUBLE PRECISION,
    source      TEXT
);
SELECT create_hypertable('metrics', 'ts');

-- Alerts with resolve support
CREATE TABLE alerts (
    id        TEXT PRIMARY KEY,
    severity  TEXT,
    metric    TEXT,
    value     DOUBLE PRECISION,
    threshold DOUBLE PRECISION,
    message   TEXT,
    ts        TIMESTAMPTZ NOT NULL,
    resolved  BOOLEAN DEFAULT FALSE
);
```

---

## Phase 3 ideas

- Replace Isolation Forest with a streaming OCSVM or Autoencoder for
  better sensitivity to concept drift.
- Add continuous TimescaleDB aggregate views for multi-resolution dashboards.
- Introduce a feedback loop: resolved alerts feed a label store that
  reweights the model's contamination estimate automatically.
- Expose `/metrics` (Prometheus format) from FastAPI for self-monitoring.
