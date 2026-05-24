import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzzy
from scipy.spatial.distance import cdist

# constantes
N_CLUSTERS = 4           
FUZZIFIER_M = 2   
ERROR_TOLERANCE = 0.005 
MAX_ITER = 1000         



df_treino = pd.read_csv('output/base_treino.csv')
df_validacao = pd.read_csv('output/base_teste_validacao.csv')


atributos_selecionados = ['atributo_1', 'atributo_2', 'atributo_3', 'atributo_4', 'atributo_5', 'atributo_6']
X_train = df_treino[atributos_selecionados].values
X_train_transposto = X_train.T 

print("Calculando Centros Iniciais")
centros_iniciais = df_treino.groupby('classe')[atributos_selecionados].mean().values

print("\n Gerando matriz de pertinência inicial (u0) baseada em distâncias...")

distancias = cdist(X_train, centros_iniciais, metric='euclidean')
distancias = np.fmax(distancias, np.finfo(np.float64).eps)

prep_u0 = distancias ** (-2 / (FUZZIFIER_M - 1))
u0_inicial = (prep_u0.T / prep_u0.sum(axis=1)).T
u0_transposto = u0_inicial.T

print(f"\nExecutando o Fuzzy C-Means ")
cntr, u, u0_final, d, jm, p, fpc = fuzzy.cluster.cmeans(
    X_train_transposto, 
    c=N_CLUSTERS, 
    m=FUZZIFIER_M, 
    error=ERROR_TOLERANCE, 
    maxiter=MAX_ITER, 
    init=u0_transposto
)

# Salvando os centros para uso no próximo passo
np.save('output/centros.npy', cntr)
print("Centros salvos em 'output/centros.npy'")

cluster_treino_predito = np.argmax(u, axis=0)

print("Validando consistência externa com a planilha de validação (20%)...")
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

df_analise_val = pd.DataFrame({
    'Cluster_FCM': cluster_val_predito + 1,
    'Classe_Real': df_validacao['classe'].values
})

print("\nMATRIZ DE CONFUSÃO (Cluster vs Classe Real):")
matriz_cruzada = pd.crosstab(df_analise_val['Cluster_FCM'], df_analise_val['Classe_Real'])
print(matriz_cruzada)

from sklearn.metrics import accuracy_score, classification_report
acuracia = accuracy_score(df_analise_val['Classe_Real'], df_analise_val['Cluster_FCM'])
print(f"\nConsistência entre clusters e classes reais: {acuracia:.4f}")


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Mostrando apenas 3 atributos para visualização
scatter = ax.scatter(
    X_train[:, 0], X_train[:, 1], X_train[:, 2], 
    c=cluster_treino_predito, cmap='viridis', s=15, alpha=0.3, label='Clusters FCM'
)

ax.scatter(
    cntr[:, 0], cntr[:, 1], cntr[:, 2], 
    marker='X', s=200, color='red', edgecolor='black', linewidth=2, label='Centros Finais'
)

ax.set_title(f'FCM Semi-Supervisionado (Todos Atributos) (c={N_CLUSTERS})', fontsize=12)
ax.set_xlabel('Atributo 1')
ax.set_ylabel('Atributo 2')
ax.set_zlabel('Atributo 3')
ax.legend()

plt.show()
