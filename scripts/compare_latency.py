import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ml.benchmark import benchmark_backend  # noqa: E402
from triage_api.model import TriageModel  # noqa: E402

SAMPLE_TEXTS = [
    "Paciente com dor toracica aguda e sudorese, suspeita de infarto.",
    "Exame de rotina sem alteracoes significativas.",
    "Achados moderados sugerem necessidade de acompanhamento em duas semanas.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--n-repeats", type=int, default=100)
    parser.add_argument("--output", default="reports/latency_comparison.md")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    sklearn_model = TriageModel(backend="sklearn", models_dir=models_dir)
    onnx_model = TriageModel(backend="onnx", models_dir=models_dir)

    sklearn_stats = benchmark_backend(sklearn_model, SAMPLE_TEXTS, args.n_repeats)
    onnx_stats = benchmark_backend(onnx_model, SAMPLE_TEXTS, args.n_repeats)

    report = (
        "# Comparativo de latencia: sklearn vs ONNX Runtime\n\n"
        "| Backend | p50 (ms) | p95 (ms) | p99 (ms) | media (ms) | amostras |\n"
        "|---|---|---|---|---|---|\n"
        f"| sklearn (RandomForest puro) | {sklearn_stats['p50_ms']:.2f} | "
        f"{sklearn_stats['p95_ms']:.2f} | {sklearn_stats['p99_ms']:.2f} | "
        f"{sklearn_stats['mean_ms']:.2f} | {sklearn_stats['count']} |\n"
        f"| onnx (classificador convertido) | {onnx_stats['p50_ms']:.2f} | "
        f"{onnx_stats['p95_ms']:.2f} | {onnx_stats['p99_ms']:.2f} | "
        f"{onnx_stats['mean_ms']:.2f} | {onnx_stats['count']} |\n"
    )
    Path(args.output).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
