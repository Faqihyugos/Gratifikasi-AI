FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
RUN uv pip install --system .

COPY apps/trainer ./apps/trainer
COPY libs ./libs

ENV PYTHONPATH=/app

CMD ["python", "apps/trainer/train.py"]
