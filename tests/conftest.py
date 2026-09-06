from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_medical_tc_csv() -> Path:
    return FIXTURES_DIR / "medical_tc_sample.csv"
