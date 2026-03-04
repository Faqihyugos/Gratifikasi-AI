FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv
ENV UV_SYSTEM_PYTHON=1

COPY pyproject.toml ./
RUN uv pip install --system ".[web]"

COPY apps/web ./apps/web
COPY libs ./libs

ENV DJANGO_SETTINGS_MODULE=gratifikasi.settings
ENV PYTHONPATH=/app:/app/apps/web

CMD ["celery", "-A", "gratifikasi", "worker", "-l", "info", "--concurrency", "2"]
