from ml.data import build_labeled_dataset
from ml.export_onnx import convert_classifier_to_onnx
from ml.train import save_model, train_model
from triage_api.model import TriageModel


def test_triage_model_onnx_backend_predicts_valid_label(sample_medical_tc_csv, tmp_path):
    df = build_labeled_dataset(sample_medical_tc_csv)
    pipeline = train_model(df)

    models_dir = tmp_path / "models"
    save_model(pipeline, models_dir / "triage_model.joblib")
    convert_classifier_to_onnx(pipeline, models_dir)

    model = TriageModel(backend="onnx", models_dir=models_dir)
    result = model.predict("Acute abdominal pain required emergency surgical intervention.")

    assert result["label"] in {"normal", "atenção", "urgente"}
    assert abs(sum(result["scores"].values()) - 1.0) < 1e-3
