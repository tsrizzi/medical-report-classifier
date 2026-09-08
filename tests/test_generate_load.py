import random

from scripts.generate_load import pick_payload


def test_pick_payload_always_valid_when_error_rate_zero():
    rng = random.Random(1)
    for _ in range(50):
        payload = pick_payload(rng, error_rate=0.0)
        assert payload["text"] != ""


def test_pick_payload_always_invalid_when_error_rate_one():
    rng = random.Random(1)
    for _ in range(50):
        payload = pick_payload(rng, error_rate=1.0)
        assert payload["text"] == ""
