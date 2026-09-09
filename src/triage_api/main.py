from functools import lru_cache

from fastapi import Depends, FastAPI

from triage_api.config import get_settings
from triage_api.metrics import metrics_endpoint, metrics_middleware
from triage_api.model import TriageModel
from triage_api.schemas import TriageRequest, TriageResponse

app = FastAPI(title="Triagem de Laudos Medicos")

app.middleware("http")(metrics_middleware)
app.add_api_route("/metrics", metrics_endpoint, methods=["GET"])


@lru_cache
def get_model() -> TriageModel:
    settings = get_settings()
    return TriageModel(backend=settings.model_backend, models_dir=settings.models_dir)


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    try:
        get_model()
        status = "ok"
    except Exception:
        status = "degraded"
    return {"status": status, "backend": settings.model_backend}


@app.post("/predict", response_model=TriageResponse)
def predict(request: TriageRequest, model: TriageModel = Depends(get_model)) -> TriageResponse:
    result = model.predict(request.text)
    return TriageResponse(**result)
