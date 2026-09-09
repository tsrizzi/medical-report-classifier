def test_metrics_endpoint_exposes_custom_metrics(client):
    client.post("/predict", json={"text": "dor toracica intensa"})

    response = client.get("/metrics")

    assert response.status_code == 200
    body = response.text
    assert "triage_requests_total" in body
    assert "triage_request_latency_seconds" in body


def test_metrics_records_500_on_unhandled_exception():
    from starlette.testclient import TestClient

    from triage_api.main import app, get_model

    class RaisingModel:
        def predict(self, text: str) -> dict:
            raise RuntimeError("boom")

    app.dependency_overrides[get_model] = lambda: RaisingModel()
    try:
        with TestClient(app, raise_server_exceptions=False) as test_client:
            response = test_client.post("/predict", json={"text": "any text"})
            assert response.status_code == 500
            metrics_response = test_client.get("/metrics")
            assert (
                'triage_errors_total{method="POST",path="/predict",status_code="500"}'
                in metrics_response.text
            )
    finally:
        app.dependency_overrides.clear()
