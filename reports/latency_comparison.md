# Comparativo de latencia: sklearn vs ONNX Runtime

| Backend | p50 (ms) | p95 (ms) | p99 (ms) | media (ms) | amostras |
|---|---|---|---|---|---|
| sklearn (RandomForest puro) | 19.55 | 24.79 | 42.50 | 20.63 | 600 |
| onnx (classificador convertido) | 0.39 | 0.50 | 0.97 | 0.74 | 600 |
