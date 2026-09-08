import time
from collections.abc import Callable

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.requests import Request
from starlette.responses import Response

REQUEST_COUNT = Counter(
    "triage_requests_total",
    "Total de requisicoes recebidas pela API",
    ["method", "path", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "triage_request_latency_seconds",
    "Latencia das requisicoes em segundos",
    ["method", "path"],
)
ERROR_COUNT = Counter(
    "triage_errors_total",
    "Total de erros retornados pela API",
    ["method", "path", "status_code"],
)


async def metrics_middleware(request: Request, call_next: Callable) -> Response:
    start_time = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start_time
    path = request.url.path
    REQUEST_LATENCY.labels(method=request.method, path=path).observe(duration)
    REQUEST_COUNT.labels(
        method=request.method, path=path, status_code=response.status_code
    ).inc()
    if response.status_code >= 400:
        ERROR_COUNT.labels(
            method=request.method, path=path, status_code=response.status_code
        ).inc()
    return response


def metrics_endpoint() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
