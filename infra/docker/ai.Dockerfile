FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
RUN uv pip install --system .

COPY apps/ai_service ./apps/ai_service
COPY libs ./libs

ENV PYTHONPATH=/app

EXPOSE 8001

CMD ["uvicorn", "apps.ai_service.main:app", "--host", "0.0.0.0", "--port", "8001"]
