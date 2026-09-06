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
python scripts/prepare_dataset.py
python scripts/train_model.py
docker build -t triage-api .
docker run --rm -p 8000:8000 -v "$(pwd)/models:/app/models:ro" triage-api
```

Testar:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Paciente com dor toracica aguda e sudorese.\"}"
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
