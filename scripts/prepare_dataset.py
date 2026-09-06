import argparse
from pathlib import Path

from ml.data import extract_raw_dataset, prepare_processed_datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", default="archive (1).zip")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    if not zip_path.exists():
        raise SystemExit(
            f"Dataset zip nao encontrado em '{zip_path}'. Baixe o 'Medical Abstracts TC "
            "Corpus' no Kaggle e coloque o arquivo .zip na raiz do projeto, ou informe "
            "--zip-path."
        )

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    extract_raw_dataset(zip_path, raw_dir)
    prepare_processed_datasets(raw_dir, processed_dir)
    print(f"Dataset processado em {processed_dir}")


if __name__ == "__main__":
    main()
