# Trabalho Intermediário — Análise e Clusterização

Este repositório contém um pipeline simples para processamento, visualização e clusterização de uma base de dados sintética, usado no contexto do Trabalho Intermediário.

## Estrutura do repositório

- `requirements.txt` — dependências do projeto.
- `data/` — dados de entrada (ex.: `base_sintetica_media.csv`).
- `imgs/` — imagens geradas durante a análise.
- `output/` — saídas do pipeline (bases processadas, resultados).
- `src/` — scripts do pipeline:
  - `step1_normalizacao.py` — normaliza e padroniza a base.
  - `step2_vizualizacao.py` — gera visualizações exploratórias.
  - `step3_separacao.py` — separa conjuntos de treino/validação/teste.
  - `step4_clusterizacao.py` — executa algoritmos de clusterização e salva resultados.

## Requisitos

Instale as dependências em um ambiente virtual (recomendado):

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Uso

Execute os passos do pipeline na ordem:

```bash
python src/step1_normalizacao.py
python src/step2_vizualizacao.py
python src/step3_separacao.py
python src/step4_clusterizacao.py
```
