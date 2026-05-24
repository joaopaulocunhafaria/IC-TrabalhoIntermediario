#!/bin/bash

# Script para execução automática do pipeline de Inteligência Computacional
# Este script executa os passos de análise, processamento, clusterização e modelagem.

set -e # Interrompe a execução em caso de erro


echo -e "\n[Passo Inicial] Instalando dependências..."
pip install -r requirements.txt --quiet

echo -e "\n[Passo 0] Analisando distribuição das classes..."
python3 src/step0_analise_classes.py

echo -e "\n[Passo 1] Normalizando os dados..."
python3 src/step1_normalizacao.py

echo -e "\n[Passo 2] Separando bases de treino e validação..."
python3 src/step2_separacao.py

echo -e "\n[Passo 3] Executando Clusterização (FCM)..."
python3 src/step3_clusterizacao.py

echo -e "\n[Passo 5] Executando Método de Mamdani..."
python3 src/step5_metodoMandami.py

echo -e "\n=========================================================="
echo "Pipeline concluído com sucesso!"
echo "As imagens geradas podem ser encontradas na pasta 'imgs/'."
echo "=========================================================="
