import zipfile
from pathlib import Path

import pandas as pd

from ml.labeling import assign_urgency_label

RAW_ZIP_MEMBERS = ("medical_tc_train.csv", "medical_tc_test.csv", "medical_tc_labels.csv")


def extract_raw_dataset(zip_path: Path, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in RAW_ZIP_MEMBERS:
            archive.extract(member, path=raw_dir)


def build_labeled_dataset(raw_csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_csv_path)
    df = df.rename(columns={"medical_abstract": "text"})
    df["urgency_label"] = df["text"].apply(assign_urgency_label)
    return df[["text", "condition_label", "urgency_label"]]


def prepare_processed_datasets(raw_dir: Path, processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        raw_csv = raw_dir / f"medical_tc_{split}.csv"
        labeled = build_labeled_dataset(raw_csv)
        labeled.to_csv(processed_dir / f"triage_{split}.csv", index=False, encoding="utf-8")
