import statistics
from collections.abc import Sequence


def compute_percentiles(durations_seconds: Sequence[float]) -> dict:
    if not durations_seconds:
        raise ValueError("durations_seconds must not be empty")
    sorted_durations = sorted(durations_seconds)
    quantiles = statistics.quantiles(sorted_durations, n=100, method="inclusive")
    return {
        "p50_ms": quantiles[49] * 1000,
        "p95_ms": quantiles[94] * 1000,
        "p99_ms": quantiles[98] * 1000,
        "mean_ms": statistics.mean(sorted_durations) * 1000,
        "count": len(sorted_durations),
    }
