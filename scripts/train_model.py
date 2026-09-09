import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from ml.train import evaluate_model, save_model, train_from_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="data/processed/triage_train.csv")
    parser.add_argument("--test-csv", default="data/processed/triage_test.csv")
    parser.add_argument("--output", default="models/triage_model.joblib")
    parser.add_argument("--metrics-output", default="reports/model_evaluation.md")
    args = parser.parse_args()

    pipeline = train_from_csv(Path(args.train_csv))
    save_model(pipeline, Path(args.output))
    print(f"Modelo salvo em {args.output}")

    test_df = pd.read_csv(args.test_csv)
    metrics = evaluate_model(pipeline, test_df)
    report_text = (
        "# Avaliação do modelo no conjunto de teste\n\n"
        f"Acurácia: {metrics['accuracy']:.4f}\n\n"
        f"```\n{metrics['report']}\n```\n"
    )
    Path(args.metrics_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_output).write_text(report_text, encoding="utf-8")
    print(report_text)


if __name__ == "__main__":
    main()
