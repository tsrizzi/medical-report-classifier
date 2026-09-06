FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

ENV PYTHONPATH=/app/src
ENV MODEL_BACKEND=sklearn
ENV MODELS_DIR=/app/models

EXPOSE 8000

CMD ["uvicorn", "triage_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
