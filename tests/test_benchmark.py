from ml.benchmark import benchmark_backend


class _ConstantModel:
    def predict(self, text: str) -> dict:
        return {"label": "normal", "scores": {"normal": 1.0}}


def test_benchmark_backend_counts_one_duration_per_call():
    stats = benchmark_backend(_ConstantModel(), texts=["a", "b"], n_repeats=3)

    assert stats["count"] == 6
    assert stats["p50_ms"] >= 0
