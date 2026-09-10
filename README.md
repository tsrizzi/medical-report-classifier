# Sistema de Triagem Automática de Laudos Médicos

Classificador de urgência (normal / atenção / urgente) para laudos médicos
em texto livre, desenvolvido para o Tech Challenge da Fase 3 do curso de
Machine Learning Engineering da FIAP / Pós Tech.

Stack: Scikit-Learn (TF-IDF + Random Forest), FastAPI, prometheus_client,
Docker, GitHub Actions, Apache Airflow, Prometheus, Grafana e ONNX Runtime.

## Decisão Arquitetural de Nuvem

O cenário descrito — um hospital que envia o texto de um laudo e precisa da
classificação de urgência imediatamente, para priorizar o atendimento — é
inerentemente **real-time**, não batch: a decisão de triagem perde o
valor se não estiver disponível em segundos, no momento em que o laudo é
produzido. Por isso a arquitetura proposta é uma API REST síncrona, e não
um job periódico de processamento em lote.

**Cloud escolhida: AWS.**

Proposta de deploy em produção (fora do escopo de execução deste desafio,
que roda localmente via Docker Compose, mas documentada como decisão
arquitetural):

- **Amazon ECR** para armazenar a imagem Docker da API construída pelo
  pipeline de CI/CD.
- **Amazon ECS com Fargate** para rodar os containers da API sem gerenciar
  servidores, com auto scaling horizontal baseado em CPU/latência.
- **Application Load Balancer** na frente do serviço ECS, expondo o
  endpoint `/predict` para os sistemas clínicos consumidores.
- **Amazon S3** para versionar os artefatos do modelo
  (`triage_model.joblib`, `triage_classifier.onnx`) gerados pela DAG de
  treino do Airflow, com o container da API baixando a versão mais
  recente na inicialização.
- **Amazon Managed Workflows for Apache Airflow (MWAA)** para rodar a DAG
  de retreino periódico em produção.
- **Amazon Managed Service for Prometheus + Amazon Managed Grafana** como
  equivalente gerenciado da stack de observabilidade usada localmente
  neste repositório.

Um pipeline batch (ex.: AWS Batch ou Step Functions processando um lote
de laudos por noite) poderia complementar esse desenho para casos de
reprocessamento em massa (ex.: re-triagem após atualização do modelo),
mas não substitui o caminho real-time, que é o requisito clínico central.

## Como executar a API localmente

```bash
docker build -t triage-api .
docker run --rm -p 8000:8000 -v "$(pwd)/models:/app/models:ro" triage-api
```

Os artefatos do modelo (`models/triage_model.joblib`,
`models/triage_classifier.onnx`) já estão versionados no repositório —
não é necessário treinar nada para rodar a API.

Testar (o modelo é treinado no Medical Abstracts TC Corpus, em inglês —
use texto em inglês para obter uma predição significativa):

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Acute abdominal pain required emergency surgical intervention.\"}"
```

A API expõe documentação interativa (Swagger UI) em
`http://localhost:8000/docs`.

### Retreinar o modelo (opcional)

Só é necessário se você quiser regenerar os artefatos a partir do
dataset. Requer o arquivo `archive (1).zip` (Medical Abstracts TC Corpus —
fonte original: https://github.com/sebischair/Medical-Abstracts-TC-Corpus,
também disponível no Kaggle) na raiz do projeto:

```bash
uv run python scripts/prepare_dataset.py
uv run python scripts/train_model.py
```

## Sobre o rótulo de urgência

O dataset público usado (Medical Abstracts TC Corpus, Kaggle) classifica
**categoria de doença**, não urgência. Como não existe um dataset público
de triagem hospitalar real com esse rótulo, o alvo `normal / atenção /
urgente` é derivado de forma determinística por `ml/labeling.py`, a
partir de palavras-chave clínicas no próprio texto do abstract (ex.:
"acute", "shock", "arrest" → urgente; "moderate", "chronic", "unstable" →
atenção; ausência de ambas → normal). A distribuição resultante e a
justificativa de hiperparâmetros (ex.: `class_weight="balanced"`) estão
detalhadas em `notebooks/01_eda.ipynb`. É uma heurística assumida para
fins didáticos de MLOps, não uma classificação clinicamente validada — o
foco do desafio é o ciclo de vida do modelo (CI/CD, orquestração,
observabilidade, latência), não a acurácia clínica do classificador.

Como o rótulo é uma função determinística de palavras-chave presentes no
próprio texto, o modelo tende a **aprender a heurística**, não um padrão
clínico independente — por isso uma acurácia alta no conjunto de teste é
esperada e não deve ser lida como validação clínica.

`uv run python scripts/train_model.py` também avalia o modelo treinado contra o
split de teste (`data/processed/triage_test.csv`) e grava o relatório em
`reports/model_evaluation.md`. Na última execução, a acurácia obtida foi
**0.9498** (ver detalhamento por classe no arquivo do relatório).

## CI/CD (GitHub Actions)

O workflow em `.github/workflows/ci.yml` roda em todo push/PR, em 3 jobs
sequenciais: `lint` (ruff) → `test` (pytest, excluindo testes marcados
`airflow`) → `build` (build da imagem Docker da API). Isso cobre a
exigência de "pelo menos 2 automações" com folga.

## Orquestração de treino (Airflow)

A DAG `triage_training_pipeline` (`dags/training_dag.py`) tem duas tasks
via TaskFlow API:

1. `load_data` — extrai/processa o dataset (`ml.data.prepare_processed_datasets`).
2. `train_and_save` — treina o pipeline TF-IDF + RandomForest e salva o
   `.joblib` (`ml.train.train_from_csv` / `save_model`).

Para rodar localmente via Docker:

```bash
docker compose -f docker-compose.airflow.yml up -d
# abrir http://localhost:8080, logar com o usuario/senha do log do container
# habilitar e disparar a DAG "triage_training_pipeline"
docker compose -f docker-compose.airflow.yml down
```

Nota: o Airflow **não** roda na CI do GitHub Actions — é uma dependência
pesada e sensível a versão (exige um arquivo de constraints oficial para
instalar via pip). O teste automatizado da DAG
(`tests/test_training_dag.py`) existe e roda localmente quando o
desenvolvedor tem `apache-airflow` instalado (`requirements-airflow.txt`,
idealmente em WSL/Linux), mas é excluído da CI via marcador pytest
(`pytest -m "not airflow"`). A validação funcional "de verdade" é feita
via `docker-compose.airflow.yml`, demonstrada no vídeo do projeto.

## Monitoramento (Prometheus + Grafana)

A API expõe métricas Prometheus em `/metrics` via `prometheus_client`:

- `triage_requests_total{method,path,status_code}` — contagem de requisicoes.
- `triage_request_latency_seconds{method,path}` — histograma de latencia.
- `triage_errors_total{method,path,status_code}` — contagem de erros (status >= 400).

Subir a stack completa:

```bash
docker compose up -d
uv run python scripts/generate_load.py --count 300 --error-rate 0.05
```

- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (login `admin`/`admin`, dashboard
  "Triagem de Laudos - API" provisionado automaticamente na pasta
  "Triagem", com 3 paineis: taxa de requisicoes por rota, latencia p95 e
  taxa de erros).

```bash
docker compose down
```

### Latência baseline medida

Gerada por `uv run python scripts/measure_latency.py` contra a API rodando com o
backend sklearn (ver `reports/baseline_latency.md`). Esta é uma medição
**fim a fim via HTTP** (rede + FastAPI + container Docker + inferência) —
por isso os valores são maiores do que os da tabela de comparação de
backends logo abaixo, que mede **apenas a chamada de inferência em
processo**, sem a camada HTTP/Docker por cima:

- Requisições: 200
- p50: 58.87 ms
- p95: 64.65 ms
- p99: 102.82 ms
- média: 59.66 ms

## Otimização de latência (ONNX Runtime)

O classificador `RandomForestClassifier` (etapa mais custosa da inferência,
com `n_estimators=100`/`max_depth=20`) foi convertido para ONNX Runtime via
`skl2onnx` (`ml/export_onnx.py`). O `TfidfVectorizer` permanece em
scikit-learn/Python — apenas a árvore de decisão roda via ONNX Runtime, o
que evita a fragilidade conhecida da conversão de pipelines de texto
inteiros para ONNX.

Resultado da comparação sklearn vs onnx (gerado por
`uv run python scripts/compare_latency.py`, medindo apenas a chamada
`TriageModel.predict()` em processo — sem rede/HTTP/Docker; ver
`reports/latency_comparison.md` para os números atuais, reproduzidos
abaixo). Os dois backends chamam `predict_proba` uma única vez e obtêm o
rótulo por `argmax`, então a comparação isola o custo real da inferência,
sem chamadas redundantes de um lado só:

| Backend | p50 (ms) | p95 (ms) | p99 (ms) | média (ms) | amostras |
|---|---|---|---|---|---|
| sklearn (RandomForest puro) | 43.91 | 52.56 | 82.76 | 45.16 | 600 |
| onnx (classificador convertido) | 1.19 | 2.30 | 5.49 | 1.57 | 600 |

Ganho de ~37x no p50 (43.91 ms → 1.19 ms) só trocando a execução do
classificador para o ONNX Runtime, sem alterar o vetorizador nem a
acurácia (paridade de classificação verificada em `tests/test_export_onnx.py`
e `tests/test_model_onnx_backend.py`). Valores absolutos de latência variam
com a carga da máquina onde o benchmark roda — o que se mantém estável
entre execuções é a ordem de grandeza do ganho relativo do ONNX.

Ganho de ~50x no p50 (19.55 ms → 0.39 ms) só trocando a execução do
classificador para o ONNX Runtime, sem alterar o vetorizador nem a
acurácia (paridade de classificação verificada em `tests/test_export_onnx.py`
e `tests/test_model_onnx_backend.py`).

Para alternar entre os dois backends em runtime, definir a variável de
ambiente `MODEL_BACKEND=sklearn` ou `MODEL_BACKEND=onnx` (o
`docker-compose.yml` já usa `onnx` por padrão).

## Vídeo (metodo STAR)

Link do vídeo gravado: **[ADICIONAR LINK APOS A GRAVACAO]**

## Rodando os testes e o lint

O projeto usa [uv](https://docs.astral.sh/uv/) para gerenciar dependências
e o ambiente virtual (`pyproject.toml` + `uv.lock`). Instalar o uv
([instruções oficiais](https://docs.astral.sh/uv/getting-started/installation/))
e então:

```bash
uv sync
uv run ruff check .
uv run pytest -m "not airflow"
```

`uv run` sincroniza o ambiente automaticamente antes de cada execução, então
`uv sync` isolado só é necessário se quiser inspecionar o `.venv` diretamente.
