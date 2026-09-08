import time

from ml.latency_stats import compute_percentiles


def benchmark_backend(model, texts: list[str], n_repeats: int) -> dict:
    durations = []
    for _ in range(n_repeats):
        for text in texts:
            start = time.perf_counter()
            model.predict(text)
            durations.append(time.perf_counter() - start)
    return compute_percentiles(durations)
