def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_label_and_scores(client):
    response = client.post("/predict", json={"text": "dor toracica intensa"})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "urgente"
    assert body["scores"]["urgente"] == 0.7


def test_predict_rejects_empty_text(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_health_reports_degraded_when_model_fails_to_load(monkeypatch):
    from starlette.testclient import TestClient

    from triage_api import main

    def _raise():
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(main, "get_model", _raise)
    with TestClient(main.app) as test_client:
        response = test_client.get("/health")

    assert response.json()["status"] == "degraded"
