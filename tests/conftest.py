from pathlib import Path

import pytest
from starlette.testclient import TestClient

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_medical_tc_csv() -> Path:
    return FIXTURES_DIR / "medical_tc_sample.csv"


class FakeTriageModel:
    def predict(self, text: str) -> dict:
        return {"label": "urgente", "scores": {"normal": 0.1, "atenção": 0.2, "urgente": 0.7}}


@pytest.fixture
def client():
    from triage_api.main import app, get_model

    app.dependency_overrides[get_model] = lambda: FakeTriageModel()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
