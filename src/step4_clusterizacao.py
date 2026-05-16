import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skfuzzy as fuzzy

# ==============================================================================
# PARÂMETROS DE CONFIGURAÇÃO (Altere aqui para tunar a performance)
# ==============================================================================
N_CLUSTERS = 4          # Número de grupos (c). Inicialmente 4 (número de classes)
FUZZIFIER_M = 2.0       # Grau de fuzificação (Valores entre 1.5 e 2.5)
ERROR_TOLERANCE = 0.005 # Critério de parada por mudança mínima
MAX_ITER = 1000         # Limite máximo de iterações do algoritmo
# ==============================================================================

print("1. Lendo as planilhas geradas no passo anterior...")
df_treino = pd.read_csv('output/base_treino.csv')

# Carrega os dados de validação separados (20% dos dados contendo a Classe Real)
df_validacao = pd.read_csv('output/base_teste_validacao.csv')

# Preparação das matrizes no formato exigido pela biblioteca (dimensões x amostras)
atributos_selecionados = ['atributo_2', 'atributo_4', 'atributo_5']
X_train = df_treino[atributos_selecionados].values
X_train_transposto = X_train.T 

print(f"2. Executando o Fuzzy C-Means nos dados de treino (c={N_CLUSTERS})...")
# Treinamento do FCM usando estritamente a planilha de treino
cntr, u, u0, d, jm, p, fpc = fuzzy.cluster.cmeans(
    X_train_transposto, 
    c=N_CLUSTERS, 
    m=FUZZIFIER_M, 
    error=ERROR_TOLERANCE, 
    maxiter=MAX_ITER, 
    init=None
)

# Partição rígida dos dados de treino apenas para fins de exibição no gráfico 3D
cluster_treino_predito = np.argmax(u, axis=0)

print("\n" + "="*60)
print("             MÉTRICAS E ANÁLISE DE CONFIABILIDADE")
print("="*60)

# --- AVALIAÇÃO INTERNA (Métrica Fuzzy Pura sobre o Treino) ---
print(f"Coeficiente de Partição Fuzzy (FPC): {fpc:.4f}")
print("-" * 60)
print("ANÁLISE DO FPC:")
if fpc > 0.70:
    print("-> RESULTADO EXCELENTE: Os clusters estão altamente espaçados e definidos.")
elif fpc > 0.50:
    print("-> RESULTADO BOM/MODERADO: Há intersecção suave entre as fronteiras dos grupos.\n"
          "   Isso valida o uso da abordagem Fuzzy para mapear as incertezas desse dataset.")
else:
    print("-> ALERTA: O FPC está muito baixo. Os grupos estão muito misturados no espaço 3D.")

print("-" * 60)

# --- AVALIAÇÃO EXTERNA (Mapeamento de Identidade usando a base de validação) ---
print("3. Validando consistência externa com a planilha de validação (20%)...")
X_val = df_validacao[atributos_selecionados].values
X_val_transposto = X_val.T

u_val, u0_val, d_val, jm_val, p_val, fpc_val = fuzzy.cluster.cmeans_predict(
    X_val_transposto, 
    cntr, 
    m=FUZZIFIER_M, 
    error=ERROR_TOLERANCE, 
    maxiter=MAX_ITER
)
cluster_val_predito = np.argmax(u_val, axis=0)

print("\nTABELA DE CONVENÇÃO (Cluster Predito vs Classe Real de Validação):")
df_analise_val = pd.DataFrame({
    'Cluster_FCM': cluster_val_predito,
    'Classe_Real': df_validacao['classe'].values
})

# Matriz cruzada para validação da utilidade do cluster para a lógica fuzzy
matriz_cruzada = pd.crosstab(df_analise_val['Cluster_FCM'], df_analise_val['Classe_Real'])
print(matriz_cruzada)
print("-" * 60)
print("COMO USAR ESSE PRINTER PARA CRIAR SUAS REGRAS (Trabalho Intelectual):")
print("-> Observe cada linha (Cluster). A coluna que possuir o maior número indica a classe\n"
      "   que aquele cluster representa matematicamente.\n"
      "   Exemplo: Se a linha do 'Cluster 0' tiver a grande maioria de seus pontos concentrados\n"
      "   na 'Classe 4', sua regra fuzzy associada ao Centro 0 terá o consequente: 'ENTÃO classe é 4'.")
print("="*60 + "\n")

print("4. Renderizando gráfico 3D da distribuição de treino...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Exibe as amostras de treino coloridas de acordo com o agrupamento cego realizado
scatter = ax.scatter(
    X_train[:, 0], X_train[:, 1], X_train[:, 2], 
    c=cluster_treino_predito, cmap='viridis', s=15, alpha=0.5, label='Amostras (Treino)'
)

# Desenha os centros fixados pelo modelo
ax.scatter(
    cntr[:, 0], cntr[:, 1], cntr[:, 2], 
    marker='X', s=200, color='red', edgecolor='black', linewidth=2, label='Centros Calculados'
)

ax.set_title(f'Visualização 3D - Clusters FCM Treinados (c={N_CLUSTERS})', fontsize=12)
ax.set_xlabel('Atributo 2 (Padronizado)')
ax.set_ylabel('Atributo 4 (Padronizado)')
ax.set_zlabel('Atributo 5 (Padronizado)')
ax.legend()

plt.show()