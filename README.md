# Trabalho Intermediário de Inteligência Computacional

## Introdução
Este projeto consiste na implementação e comparação de modelos de inferência fuzzy para um problema de classificação baseado em uma base de dados sintética. O trabalho aborda desde a análise exploratória e pré-processamento dos dados até a aplicação de técnicas de clusterização **Fuzzy C-Means (FCM)** e a construção de modelos de lógica fuzzy do tipo **Mamdani**.

O objetivo principal é extrair regras de inferência a partir da topologia dos dados (clusters) e avaliar a capacidade desses modelos em classificar corretamente amostras de validação.

## Estrutura do Projeto
A organização do repositório reflete as etapas do pipeline de desenvolvimento:

```text
/
├── data/               # Conjunto de dados original (CSV)
├── output/             # Arquivos gerados (bases processadas, centros, coeficientes)
├── imgs/               # Gráficos e visualizações geradas (TSK, Mamdani, FCM)
├── src/                # Código-fonte organizado por etapas
│   ├── step0_...       # Análise de classes
│   ├── step1_...       # Normalização
│   ├── step2_...       # Separação (Treino/Validação)
│   ├── step3_...       # Clusterização e definição de centros
│   ├── step4_...       # Implementação do Modelo TSK
│   ├── step5_...       # Implementação do Método Mamdani
│   └── graficos/       # Scripts auxiliares para geração de visualizações
├── requirements.txt    # Dependências do projeto
├── run.sh              # Script de execução automatizada
└── README.md           # Documentação do projeto
```

## Requisitos
Para executar o projeto, você precisará de Python 3.10+ e das seguintes bibliotecas:
- `pandas`
- `numpy`
- `scikit-fuzzy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `scipy`

Você pode instalar todas as dependências via pip:
```bash
pip install -r requirements.txt
```
Caso prefira, pode instalar as dependências também rodando o próximo passo. 
## Como Rodar

### Usando o script automatizado
A forma mais simples de rodar todo o pipeline (da instalação das dependências à inferência) é através do script `run.sh`:
```bash
chmod +x run.sh
./run.sh
```

### Rodando arquivos em separado
Caso deseje executar ou testar etapas específicas, você pode rodar os arquivos individualmente a partir da raiz do projeto:

1. **Análise:** `python3 src/step0_analise_classes.py`
2. **Normalização:** `python3 src/step1_normalizacao.py`
3. **Separação:** `python3 src/step2_separacao.py`
4. **Clusterização:** `python3 src/step3_clusterizacao.py`
5. **Modelo TSK:** `python3 src/step4_modeloTSK.py`
6. **Modelo Mamdani:** `python3 src/step5_metodoMandami.py`

*Nota: Certifique-se de seguir a ordem numérica para garantir que os arquivos necessários no diretório `output/` sejam gerados corretamente.*


# Autores

<p>
  João Paulo da Cunha Faria - Graduando em Engenharia da Computação pelo <a href="https://www.cefetmg.br" target="_blank">CEFET-MG</a>. Contato: (<a href="mailto:joao.cruz@aluno.cefetmg.br">joao@aluno.cefetmg.br</a>)
</p>

<p>
  Joaquim Cézar Santana da Cruz - Graduando em Engenharia da Computação pelo <a href="https://www.cefetmg.br" target="_blank">CEFET-MG</a>. Contato: (<a href="mailto:joaquim.cruz@aluno.cefetmg.br">joaquim.cruz@aluno.cefetmg.br</a>)
</p>
