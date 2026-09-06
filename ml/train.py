from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 1))),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def train_model(df: pd.DataFrame) -> Pipeline:
    pipeline = build_pipeline()
    pipeline.fit(df["text"], df["urgency_label"])
    return pipeline


def save_model(pipeline: Pipeline, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)


def train_from_csv(csv_path: Path) -> Pipeline:
    df = pd.read_csv(csv_path)
    return train_model(df)
