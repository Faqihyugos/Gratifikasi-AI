# Gratifikasi-AI

Production-ready AI system for automated gratification case classification, built on IndoBERT with similarity-based retrieval (Qdrant) and MLflow model management.

## Architecture

```
┌─────────────┐     HTTP      ┌──────────────────┐
│  Django Web │ ──────────── ▶│  FastAPI AI Svc  │
│  (REST API) │               │  /predict        │
│             │               │  /cases/upsert   │
└──────┬──────┘               └───────┬──────────┘
       │ Celery tasks                  │
       ▼                        ┌──────┴──────┐   ┌──────────┐
  ┌─────────┐                   │   Qdrant    │   │  MLflow  │
  │  Redis  │                   │ (vector DB) │   │ Registry │
  └─────────┘                   └─────────────┘   └──────────┘
       │
  ┌────▼────┐   ┌──────────┐
  │ Celery  │   │Celery    │
  │ Worker  │   │ Beat     │
  └─────────┘   └──────────┘
```

### Services

| Service   | Description                                     | Port  |
|-----------|-------------------------------------------------|-------|
| `web`     | Django REST API — submit & approve records      | 8000  |
| `ai`      | FastAPI inference — similarity + classifier     | 8001  |
| `worker`  | Celery worker — async AI calls, Qdrant upserts  | —     |
| `beat`    | Celery beat — scheduled data-drift check        | —     |
| `trainer` | Fine-tuning pipeline (on-demand, profile-gated) | —     |
| `db`      | PostgreSQL 16                                   | 5432  |
| `redis`   | Redis 7                                         | 6379  |
| `qdrant`  | Qdrant vector store                             | 6333  |
| `mlflow`  | MLflow tracking + model registry               | 5000  |
| `minio`   | S3-compatible artifact store                    | 9000  |

## Project Structure

```
.
├── apps/
│   ├── web/                    # Django application
│   │   ├── gratifikasi/        # Django project settings, celery, urls
│   │   └── records/            # Records app (models, views, tasks)
│   ├── ai_service/             # FastAPI inference service
│   │   └── routers/            # predict, model_info, cases, health
│   └── trainer/                # Fine-tuning pipeline
├── libs/
│   └── common/                 # Shared logging & settings utilities
├── scripts/
│   └── promote_to_production.py
├── infra/docker/               # Per-service Dockerfiles
├── docker-compose.yml
├── pyproject.toml              # uv-managed dependencies
└── .env.example
```

## Quick Start

### Prerequisites
- Docker + Docker Compose v2
- [uv](https://docs.astral.sh/uv/) (for local development)

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env — at minimum set DJANGO_SECRET_KEY to a random string
```

### 2. Start all services

```bash
docker compose up -d
```

### 3. Run Django migrations

```bash
docker compose exec web uv run python apps/web/manage.py migrate --run-syncdb
```

### 4. Create a superuser (optional)

```bash
docker compose exec web uv run python apps/web/manage.py createsuperuser
```

## API Reference

### Submit a record

```http
POST /api/records/
Content-Type: application/json

{
  "text": "Penerimaan hadiah berupa uang tunai sebesar Rp 500.000 dari rekanan",
  "value_estimation": 500000
}
```

Response: the created record (status `PROCESSING`). A Celery task calls the AI service asynchronously. Poll the detail endpoint until `status` becomes `WAITING_APPROVAL`.

### Get record details

```http
GET /api/records/{id}/
```

### Approve / set final label

```http
POST /api/records/{id}/approve/
Content-Type: application/json

{
  "final_label": "Milik Negara",
  "note": "Confirmed by compliance officer"
}
```

Valid labels: `"Milik Negara"` or `"Bukan Milik Negara"`.

### Get audit log

```http
GET /api/records/{id}/audit/
```

### AI Service endpoints

| Method | Path            | Description                  |
|--------|-----------------|------------------------------|
| GET    | `/healthz`      | Health check                 |
| POST   | `/predict`      | Run inference on text        |
| GET    | `/model`        | Current model info           |
| POST   | `/cases/upsert` | Embed and store in Qdrant    |

## Inference Pipeline

1. **Similarity search** — embed text with `paraphrase-multilingual-MiniLM-L12-v2`, search Qdrant for nearest neighbours
2. **Threshold gate** — if best cosine similarity ≥ `SIMILARITY_THRESHOLD` (default 0.85), return the matched label
3. **Classifier fallback** — otherwise run the fine-tuned IndoBERT model loaded from MLflow

## Training

Run on-demand (profile-gated so it does not start with `docker compose up`):

```bash
docker compose run --rm trainer
```

The trainer will:
1. Load labelled records from Postgres (or a CSV/JSONL file via `TRAIN_DATA_PATH`)
2. Fine-tune `BASE_MODEL` with HuggingFace Trainer
3. Log params, metrics, and the model artifact to MLflow
4. Auto-promote to **Staging** if `eval_f1 >= PROMOTE_F1_THRESHOLD`

### Promote to Production

```bash
python scripts/promote_to_production.py --version <VERSION>
```

Then restart the AI service to pick up the new model:

```bash
docker compose restart ai
```

## Key Environment Variables

| Variable               | Default                               | Description                         |
|------------------------|---------------------------------------|-------------------------------------|
| `DJANGO_SECRET_KEY`    | `change-me-in-production`             | Django secret key                   |
| `DATABASE_URL`         | `postgresql://gratifikasi:...@db/...` | PostgreSQL connection URL           |
| `REDIS_URL`            | `redis://redis:6379/0`                | Redis / Celery broker URL           |
| `AI_SERVICE_URL`       | `http://ai:8001`                      | FastAPI service URL (from worker)   |
| `MLFLOW_TRACKING_URI`  | `http://mlflow:5000`                  | MLflow server URL                   |
| `MODEL_URI`            | `models:/gratifikasi_classifier/Production` | MLflow model URI            |
| `SIMILARITY_THRESHOLD` | `0.85`                                | Qdrant similarity gate              |
| `BASE_MODEL`           | `indobenchmark/indobert-base-p1`      | HuggingFace base model for training |
| `PROMOTE_F1_THRESHOLD` | `0.85`                                | Auto-promote to Staging threshold   |

## Development

### Install dependencies locally

```bash
uv sync
```

### Run Django dev server

```bash
cd apps/web
uv run python manage.py migrate
uv run python manage.py runserver
```

### Run AI service locally

```bash
uv run uvicorn apps.ai_service.main:app --reload --port 8001
```

### Run Celery worker locally

```bash
cd apps/web
uv run celery -A gratifikasi worker -l info
```

## License

MIT

## django and fast api