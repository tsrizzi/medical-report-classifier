import pytest

from ml.data import build_labeled_dataset
from ml.train import save_model, train_model
from triage_api.model import TriageModel


@pytest.fixture
def trained_model_dir(sample_medical_tc_csv, tmp_path):
    df = build_labeled_dataset(sample_medical_tc_csv)
    pipeline = train_model(df)
    models_dir = tmp_path / "models"
    save_model(pipeline, models_dir / "triage_model.joblib")
    return models_dir


def test_predict_returns_label_and_scores(trained_model_dir):
    model = TriageModel(backend="sklearn", models_dir=trained_model_dir)
    result = model.predict("Acute abdominal pain required emergency surgical intervention.")

    assert result["label"] in {"normal", "atenção", "urgente"}
    assert abs(sum(result["scores"].values()) - 1.0) < 1e-6


def test_unsupported_backend_raises_value_error(trained_model_dir):
    with pytest.raises(ValueError):
        TriageModel(backend="bogus", models_dir=trained_model_dir)
