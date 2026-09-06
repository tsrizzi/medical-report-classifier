import shutil

import pytest

pytest.importorskip("airflow")

pytestmark = pytest.mark.airflow


def test_training_pipeline_produces_model(tmp_path, monkeypatch, sample_medical_tc_csv):
    import dags.training_dag as dag_module

    raw_dir = tmp_path / "data" / "raw"
    models_dir = tmp_path / "models"
    raw_dir.mkdir(parents=True)

    shutil.copyfile(sample_medical_tc_csv, raw_dir / "medical_tc_train.csv")
    shutil.copyfile(sample_medical_tc_csv, raw_dir / "medical_tc_test.csv")

    monkeypatch.setattr(dag_module, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(dag_module, "MODELS_DIR", models_dir)

    dag_module.triage_training_pipeline().test()

    assert (models_dir / "triage_model.joblib").exists()
