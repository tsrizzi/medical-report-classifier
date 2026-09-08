# Comparativo de latencia: sklearn vs ONNX Runtime

| Backend | p50 (ms) | p95 (ms) | p99 (ms) | media (ms) | amostras |
|---|---|---|---|---|---|
| sklearn (RandomForest puro) | 39.45 | 52.56 | 69.22 | 41.51 | 600 |
| onnx (classificador convertido) | 0.42 | 1.14 | 16.44 | 0.95 | 600 |
