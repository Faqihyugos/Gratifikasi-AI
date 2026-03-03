FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
RUN uv sync --no-dev

COPY apps/trainer ./apps/trainer
COPY libs ./libs

ENV PYTHONPATH=/app

CMD ["uv", "run", "python", "apps/trainer/train.py"]
