# Comparativo de latencia: sklearn vs ONNX Runtime

| Backend | p50 (ms) | p95 (ms) | p99 (ms) | media (ms) | amostras |
|---|---|---|---|---|---|
| sklearn (RandomForest puro) | 43.91 | 52.56 | 82.76 | 45.16 | 600 |
| onnx (classificador convertido) | 1.19 | 2.30 | 5.49 | 1.57 | 600 |
