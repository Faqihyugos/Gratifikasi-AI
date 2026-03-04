FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv
ENV UV_SYSTEM_PYTHON=1

COPY pyproject.toml ./
# No PyTorch — uses fastembed (ONNX embeddings) + onnxruntime (ONNX inference)
RUN uv pip install --system ".[ai]"

COPY apps/ai_service ./apps/ai_service
COPY libs ./libs

ENV PYTHONPATH=/app

EXPOSE 8001

CMD ["uvicorn", "apps.ai_service.main:app", "--host", "0.0.0.0", "--port", "8001"]
