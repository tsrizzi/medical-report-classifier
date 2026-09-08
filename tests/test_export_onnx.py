import json

import numpy as np
import onnxruntime as ort

from ml.data import build_labeled_dataset
from ml.export_onnx import convert_classifier_to_onnx
from ml.train import train_model


def test_onnx_classifier_matches_sklearn_predictions(sample_medical_tc_csv, tmp_path):
    df = build_labeled_dataset(sample_medical_tc_csv)
    pipeline = train_model(df)

    output_dir = tmp_path / "models"
    convert_classifier_to_onnx(pipeline, output_dir)

    vectorizer = pipeline.named_steps["tfidf"]
    classes = json.loads((output_dir / "triage_model_classes.json").read_text())
    session = ort.InferenceSession(str(output_dir / "triage_classifier.onnx"))
    input_name = session.get_inputs()[0].name

    for text in df["text"].tolist():
        expected_label = pipeline.predict([text])[0]

        features = vectorizer.transform([text]).toarray().astype(np.float32)
        _, proba_array = session.run(None, {input_name: features})
        onnx_label = classes[int(np.argmax(proba_array[0]))]

        assert onnx_label == expected_label
