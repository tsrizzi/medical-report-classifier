import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.export_onnx import convert_from_joblib  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sklearn-model", default="models/triage_model.joblib")
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    convert_from_joblib(Path(args.sklearn_model), Path(args.output_dir))
    print(f"Classificador ONNX exportado em {args.output_dir}")


if __name__ == "__main__":
    main()
