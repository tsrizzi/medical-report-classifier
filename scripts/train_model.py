import argparse
from pathlib import Path

from ml.train import save_model, train_from_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="data/processed/triage_train.csv")
    parser.add_argument("--output", default="models/triage_model.joblib")
    args = parser.parse_args()

    pipeline = train_from_csv(Path(args.train_csv))
    save_model(pipeline, Path(args.output))
    print(f"Modelo salvo em {args.output}")


if __name__ == "__main__":
    main()
