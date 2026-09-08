import json
from pathlib import Path

DASHBOARD_PATH = (
    Path(__file__).parent.parent / "grafana" / "dashboards" / "triagem-api-dashboard.json"
)

EXPECTED_METRIC_SUBSTRINGS = (
    "triage_requests_total",
    "triage_request_latency_seconds_bucket",
    "triage_errors_total",
)


def test_dashboard_has_at_least_three_panels():
    dashboard = json.loads(DASHBOARD_PATH.read_text(encoding="utf-8"))
    assert len(dashboard["panels"]) >= 3


def test_dashboard_panels_reference_expected_metrics():
    dashboard = json.loads(DASHBOARD_PATH.read_text(encoding="utf-8"))
    all_exprs = " ".join(
        target["expr"] for panel in dashboard["panels"] for target in panel.get("targets", [])
    )
    for metric in EXPECTED_METRIC_SUBSTRINGS:
        assert metric in all_exprs
