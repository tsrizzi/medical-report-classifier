from scripts.measure_latency import run_latency_probe


def test_run_latency_probe_returns_one_duration_per_request(client):
    durations = run_latency_probe(client, n_requests=5)

    assert len(durations) == 5
    assert all(d >= 0 for d in durations)
