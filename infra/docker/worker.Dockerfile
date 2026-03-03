FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
RUN uv sync --no-dev

COPY apps/web ./apps/web
COPY libs ./libs

ENV DJANGO_SETTINGS_MODULE=gratifikasi.settings
ENV PYTHONPATH=/app:/app/apps/web

CMD ["uv", "run", "celery", "-A", "gratifikasi", "worker", "-l", "info", "--concurrency", "2"]
