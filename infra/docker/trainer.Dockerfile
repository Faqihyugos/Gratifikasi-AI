FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv
ENV UV_SYSTEM_PYTHON=1

COPY pyproject.toml ./
# torch CPU-only to skip NVIDIA CUDA packages (~4GB)
RUN uv pip install --system ".[trainer]" \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY apps/trainer ./apps/trainer
COPY libs ./libs

ENV PYTHONPATH=/app

CMD ["python", "apps/trainer/train.py"]
