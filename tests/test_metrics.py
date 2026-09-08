def test_metrics_endpoint_exposes_custom_metrics(client):
    client.post("/predict", json={"text": "dor toracica intensa"})

    response = client.get("/metrics")

    assert response.status_code == 200
    body = response.text
    assert "triage_requests_total" in body
    assert "triage_request_latency_seconds" in body
