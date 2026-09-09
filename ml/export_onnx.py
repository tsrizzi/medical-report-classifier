import json
from pathlib import Path

import joblib
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.pipeline import Pipeline


def convert_classifier_to_onnx(pipeline: Pipeline, output_dir: Path) -> None:
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, output_dir / "triage_vectorizer.joblib")

    n_features = len(vectorizer.get_feature_names_out())
    onnx_model = to_onnx(
        classifier,
        initial_types=[("input", FloatTensorType([None, n_features]))],
        options={id(classifier): {"zipmap": False}},
    )
    (output_dir / "triage_classifier.onnx").write_bytes(onnx_model.SerializeToString())
    (output_dir / "triage_model_classes.json").write_text(
        json.dumps(list(classifier.classes_)), encoding="utf-8"
    )


def convert_from_joblib(sklearn_pipeline_path: Path, output_dir: Path) -> None:
    pipeline = joblib.load(sklearn_pipeline_path)
    convert_classifier_to_onnx(pipeline, output_dir)
