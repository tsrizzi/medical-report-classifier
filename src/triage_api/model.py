import json
from pathlib import Path

import joblib
import numpy as np

DEFAULT_MODELS_DIR = Path("models")


class TriageModel:
    def __init__(self, backend: str, models_dir: Path = DEFAULT_MODELS_DIR):
        self.backend = backend
        if backend == "sklearn":
            self._pipeline = joblib.load(models_dir / "triage_model.joblib")
            self._classes = list(self._pipeline.classes_)
        elif backend == "onnx":
            import onnxruntime as ort

            self._vectorizer = joblib.load(models_dir / "triage_vectorizer.joblib")
            self._session = ort.InferenceSession(str(models_dir / "triage_classifier.onnx"))
            self._classes = json.loads((models_dir / "triage_model_classes.json").read_text())
        else:
            raise ValueError(f"unsupported MODEL_BACKEND: {backend}")

    def predict(self, text: str) -> dict:
        if self.backend == "sklearn":
            label = self._pipeline.predict([text])[0]
            proba = self._pipeline.predict_proba([text])[0]
        else:
            features = self._vectorizer.transform([text]).toarray().astype(np.float32)
            input_name = self._session.get_inputs()[0].name
            _, proba_array = self._session.run(None, {input_name: features})
            proba = np.asarray(proba_array[0])
            label = self._classes[int(np.argmax(proba))]
        return {
            "label": str(label),
            "scores": {c: float(p) for c, p in zip(self._classes, proba)},
        }
