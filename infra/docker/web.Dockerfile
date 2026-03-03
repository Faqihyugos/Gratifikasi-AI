FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
RUN uv pip install --system .

COPY apps/web ./apps/web
COPY libs ./libs

ENV DJANGO_SETTINGS_MODULE=gratifikasi.settings
ENV PYTHONPATH=/app:/app/apps/web

EXPOSE 8000

CMD ["gunicorn", "gratifikasi.wsgi:application", \
     "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", \
     "--chdir", "/app/apps/web"]
