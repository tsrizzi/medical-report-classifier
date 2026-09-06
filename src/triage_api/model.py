from pathlib import Path

import joblib

DEFAULT_MODELS_DIR = Path("models")


class TriageModel:
    def __init__(self, backend: str, models_dir: Path = DEFAULT_MODELS_DIR):
        self.backend = backend
        if backend == "sklearn":
            self._pipeline = joblib.load(models_dir / "triage_model.joblib")
            self._classes = list(self._pipeline.classes_)
        else:
            raise ValueError(f"unsupported MODEL_BACKEND: {backend}")

    def predict(self, text: str) -> dict:
        label = self._pipeline.predict([text])[0]
        proba = self._pipeline.predict_proba([text])[0]
        return {
            "label": str(label),
            "scores": {c: float(p) for c, p in zip(self._classes, proba)},
        }
