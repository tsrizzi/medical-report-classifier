from datetime import datetime
from pathlib import Path

from airflow.decorators import dag, task

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@dag(
    dag_id="triage_training_pipeline",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["triage", "training"],
)
def triage_training_pipeline():
    @task
    def load_data() -> str:
        from ml.data import extract_raw_dataset, prepare_processed_datasets

        zip_path = DATA_DIR / "archive (1).zip"
        raw_dir = DATA_DIR / "raw"
        processed_dir = DATA_DIR / "processed"
        if not (raw_dir / "medical_tc_train.csv").exists():
            extract_raw_dataset(zip_path, raw_dir)
        prepare_processed_datasets(raw_dir, processed_dir)
        return str(processed_dir / "triage_train.csv")

    @task
    def train_and_save(processed_train_csv: str) -> str:
        from ml.train import save_model, train_from_csv

        pipeline = train_from_csv(Path(processed_train_csv))
        output_path = MODELS_DIR / "triage_model.joblib"
        save_model(pipeline, output_path)
        return str(output_path)

    train_and_save(load_data())


triage_training_pipeline()
