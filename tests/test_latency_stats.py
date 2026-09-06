from ml.latency_stats import compute_percentiles


def test_compute_percentiles_on_known_distribution():
    durations = [i / 1000 for i in range(1, 101)]  # 0.001s .. 0.100s

    stats = compute_percentiles(durations)

    assert stats["count"] == 100
    assert 49 <= stats["p50_ms"] <= 51
    assert 94 <= stats["p95_ms"] <= 96
    assert stats["p99_ms"] >= stats["p95_ms"]


def test_compute_percentiles_raises_on_empty_input():
    import pytest

    with pytest.raises(ValueError):
        compute_percentiles([])
