import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzzy
import os
from sklearn.metrics import silhouette_score

# Configurações
INPUT_FILE = 'output/base_treino.csv'
OUTPUT_DIR = 'imgs'
ATRIBUTOS = ['atributo_2', 'atributo_4', 'atributo_5']

# Parâmetros FCM
FUZZIFIER_M = 2.0
ERROR_TOLERANCE = 0.005
MAX_ITER = 1000

def generate_graphics():
    # 1. Carregar dados
    if not os.path.exists(INPUT_FILE):
        print(f"Erro: Arquivo {INPUT_FILE} não encontrado. Execute os steps 1, 2 e 3 primeiro.")
        return
    
    df = pd.read_csv(INPUT_FILE)
    X = df[ATRIBUTOS].values
    data = X.T

    range_n_clusters = range(2, 7)
    fpcs = []
    silhouette_avg = []

    print("Calculando métricas para diferentes números de clusters...")
    for n_clusters in range_n_clusters:
        # Executar FCM
        cntr, u, u0, d, jm, p, fpc = fuzzy.cluster.cmeans(
            data, n_clusters, FUZZIFIER_M, ERROR_TOLERANCE, MAX_ITER, init=None
        )
        fpcs.append(fpc)
        
        cluster_labels = np.argmax(u, axis=0)
        
        # Calcular Silhouette Score
        score = silhouette_score(X, cluster_labels)
        silhouette_avg.append(score)

    # Curva FPC
    plt.figure(figsize=(8, 5))
    plt.plot(range_n_clusters, fpcs, marker='o', linestyle='--', color='b')
    plt.title('Justificativa do Número de Clusters (FPC)')
    plt.xlabel('Número de Clusters (c)')
    plt.ylabel('Fuzzy Partition Coefficient (FPC)')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'curva_fpc.png'))
    plt.close()
    print("-> Gráfico 'curva_fpc.png' salvo em imgs/")

    # Gráfico do Método da Silhouette 
    plt.figure(figsize=(8, 5))
    plt.plot(range_n_clusters, silhouette_avg, marker='s', linestyle='-', color='green')
    plt.title('Método da Silhouette (Validação do Agrupamento)')
    plt.xlabel('Número de Clusters (c)')
    plt.ylabel('Silhouette Score Médio')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'metodo_silhouette.png'))
    plt.close()
    print("-> Gráfico 'metodo_silhouette.png' salvo em imgs/")

    # Gráfico de Perfil dos Centros (para c=4) 
    print("Gerando Perfil dos Centros (para c=4)...")
    N_CLUSTERS = 4
    cntr, u, u0, d, jm, p, fpc = fuzzy.cluster.cmeans(
        data, N_CLUSTERS, FUZZIFIER_M, ERROR_TOLERANCE, MAX_ITER, init=None
    )

    df_centers = pd.DataFrame(cntr, columns=ATRIBUTOS)
    df_centers.index.name = 'Cluster'
    
    ax = df_centers.plot(kind='bar', figsize=(10, 6))
    plt.title(f'Perfil dos Centros dos Clusters (c={N_CLUSTERS})')
    plt.xlabel('Clusters')
    plt.ylabel('Valor Padronizado (Z-score)')
    plt.xticks(rotation=0)
    plt.legend(title='Atributos')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.8)
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'perfil_centros.png'))
    plt.close()
    print("-> Gráfico 'perfil_centros.png' salvo em imgs/")

if __name__ == "__main__":
    generate_graphics()
