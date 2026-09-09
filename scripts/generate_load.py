import argparse
import random
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.samples import SAMPLE_MEDICAL_TEXTS as VALID_TEXTS  # noqa: E402


def pick_payload(rng: random.Random, error_rate: float) -> dict:
    if rng.random() < error_rate:
        return {"text": ""}
    return {"text": rng.choice(VALID_TEXTS)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--error-rate", type=float, default=0.05)
    parser.add_argument("--interval-seconds", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    with httpx.Client(base_url=args.base_url, timeout=10.0) as client:
        for _ in range(args.count):
            payload = pick_payload(rng, args.error_rate)
            client.post("/predict", json=payload)
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
