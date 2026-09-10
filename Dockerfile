FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
COPY models/ ./models/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app/src
ENV MODEL_BACKEND=sklearn
ENV MODELS_DIR=/app/models

EXPOSE 8000

CMD ["uvicorn", "triage_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
