import zipfile

import pandas as pd

from ml.data import build_labeled_dataset, extract_raw_dataset, prepare_processed_datasets


def test_build_labeled_dataset_adds_urgency_column(sample_medical_tc_csv):
    df = build_labeled_dataset(sample_medical_tc_csv)

    assert list(df.columns) == ["text", "condition_label", "urgency_label"]
    assert set(df["urgency_label"].unique()) <= {"normal", "atenção", "urgente"}
    assert len(df) == 18


def test_extract_raw_dataset_extracts_expected_members(tmp_path, sample_medical_tc_csv):
    zip_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(sample_medical_tc_csv, arcname="medical_tc_train.csv")
        archive.write(sample_medical_tc_csv, arcname="medical_tc_test.csv")
        archive.write(sample_medical_tc_csv, arcname="medical_tc_labels.csv")

    raw_dir = tmp_path / "raw"
    extract_raw_dataset(zip_path, raw_dir)

    assert (raw_dir / "medical_tc_train.csv").exists()
    assert (raw_dir / "medical_tc_test.csv").exists()
    assert (raw_dir / "medical_tc_labels.csv").exists()


def test_prepare_processed_datasets_writes_train_and_test_csv(tmp_path, sample_medical_tc_csv):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for split in ("train", "test"):
        pd.read_csv(sample_medical_tc_csv).to_csv(raw_dir / f"medical_tc_{split}.csv", index=False)

    processed_dir = tmp_path / "processed"
    prepare_processed_datasets(raw_dir, processed_dir)

    assert (processed_dir / "triage_train.csv").exists()
    assert (processed_dir / "triage_test.csv").exists()
    train_df = pd.read_csv(processed_dir / "triage_train.csv")
    assert "urgency_label" in train_df.columns
