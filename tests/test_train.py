import joblib

from ml.data import build_labeled_dataset
from ml.train import evaluate_model, save_model, train_model


def test_train_model_predicts_known_classes(sample_medical_tc_csv):
    df = build_labeled_dataset(sample_medical_tc_csv)
    pipeline = train_model(df)

    text = "Acute abdominal pain required emergency surgical intervention."
    prediction = pipeline.predict([text])[0]

    assert prediction in {"normal", "atenção", "urgente"}
    assert set(pipeline.classes_) == {"normal", "atenção", "urgente"}


def test_save_model_round_trips_predictions(sample_medical_tc_csv, tmp_path):
    df = build_labeled_dataset(sample_medical_tc_csv)
    pipeline = train_model(df)
    output_path = tmp_path / "model.joblib"

    save_model(pipeline, output_path)
    reloaded = joblib.load(output_path)

    text = "Chronic but stable presentation of the nervous system disorder."
    assert reloaded.predict([text])[0] == pipeline.predict([text])[0]


def test_evaluate_model_returns_accuracy_and_report(sample_medical_tc_csv):
    df = build_labeled_dataset(sample_medical_tc_csv)
    pipeline = train_model(df)

    metrics = evaluate_model(pipeline, df)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert "precision" in metrics["report"]
