import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx2 as httpx  # noqa: E402

from ml.latency_stats import compute_percentiles  # noqa: E402
from ml.samples import SAMPLE_MEDICAL_TEXTS as SAMPLE_TEXTS  # noqa: E402


def run_latency_probe(client: httpx.Client, n_requests: int) -> list[float]:
    durations = []
    for i in range(n_requests):
        text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]
        start = time.perf_counter()
        response = client.post("/predict", json={"text": text})
        response.raise_for_status()
        durations.append(time.perf_counter() - start)
    return durations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--n-requests", type=int, default=200)
    parser.add_argument("--output", default="reports/baseline_latency.md")
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=10.0) as client:
        durations = run_latency_probe(client, args.n_requests)

    stats = compute_percentiles(durations)
    report = (
        "# Latencia baseline da API (backend sklearn)\n\n"
        f"- Requisicoes: {stats['count']}\n"
        f"- p50: {stats['p50_ms']:.2f} ms\n"
        f"- p95: {stats['p95_ms']:.2f} ms\n"
        f"- p99: {stats['p99_ms']:.2f} ms\n"
        f"- media: {stats['mean_ms']:.2f} ms\n"
    )
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
