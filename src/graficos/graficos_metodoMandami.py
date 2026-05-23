import pandas as pd
import numpy as np
import skfuzzy as fuzzy
import matplotlib
matplotlib.use('Agg') # Necessário para rodar em ambientes sem interface gráfica
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# --- Configurações Iniciais ---
# Caminhos dos arquivos gerados nos passos anteriores
caminho_centros = 'output/centros.npy'
caminho_treino = 'output/base_treino.csv'
caminho_validacao = 'output/base_teste_validacao.csv'

# Carregando os dados
print("Carregando bases e centros...")
centros = np.load(caminho_centros)
treino = pd.read_csv(caminho_treino)
validacao = pd.read_csv(caminho_validacao)

colunas_entrada = ['atributo_1', 'atributo_2', 'atributo_3', 'atributo_4', 'atributo_5', 'atributo_6']
X_treino = treino[colunas_entrada].values
y_treino = treino['classe'].values
X_val = validacao[colunas_entrada].values
y_val = validacao['classe'].values

# Parâmetros do Fuzzy C-Means
m_fuzzy = 1.5
tol_erro = 0.005
iter_max = 1000

# --- 1. Construção da Base de Regras (Mamdani) ---
print("Mapeando Clusters para as Classes (Base de Regras)...")
# Obtemos as pertinências do treino para saber qual classe domina cada cluster
u_treino, _, _, _, _, _ = fuzzy.cluster.cmeans_predict(
    X_treino.T, centros, m=m_fuzzy, error=tol_erro, maxiter=iter_max
)
pesos_treino = u_treino.T

n_clusters = centros.shape[0]
classes_unicas = np.unique(y_treino)
mapa_regras = {}

for i in range(n_clusters):
    # Identifica quais amostras pertencem majoritariamente a este cluster
    indices_cluster = np.argmax(pesos_treino, axis=1) == i
    if np.any(indices_cluster):
        # A classe mais frequente no cluster define o consequente da regra
        classe_voto = np.bincount(y_treino[indices_cluster]).argmax()
        mapa_regras[i] = classe_voto
    else:
        mapa_regras[i] = classes_unicas[0]

# Inferência no Conjunto de Validação 
print("Realizando inferência Mamdani na validação...")
u_val, _, _, _, _, _ = fuzzy.cluster.cmeans_predict(
    X_val.T, centros, m=m_fuzzy, error=tol_erro, maxiter=iter_max
)
pesos_val = u_val.T # Força de disparo das regras

# Matriz para guardar a pertinência final de cada amostra às classes 
pertinencia_classes = np.zeros((X_val.shape[0], np.max(classes_unicas) + 1))

for i in range(n_clusters):
    classe_consequente = mapa_regras[i]
    forca_regra = pesos_val[:, i]
    
    # Agregação via S-Norma Máximo 
    pertinencia_classes[:, classe_consequente] = np.maximum(
        pertinencia_classes[:, classe_consequente], 
        forca_regra
    )

# Defuzzificação por Máximo: Escolhe a classe com maior pertinência acumulada
predicoes_mamdani = np.argmax(pertinencia_classes[:, 1:], axis=1) + 1


print("Gerando visualização 3D...")
# Índices dos Atributos 2, 4 e 5 (colunas 1, 3 e 4)
idx2, idx4, idx5 = 1, 3, 4

fig = plt.figure(figsize=(15, 7))

# Gráfico da Esquerda: Dados Reais
ax1 = fig.add_subplot(121, projection='3d')
scatter1 = ax1.scatter(X_val[:, idx2], X_val[:, idx4], X_val[:, idx5], 
                      c=y_val, cmap='viridis', s=25, alpha=0.6)
ax1.set_title('Classes Reais (Validação)')
ax1.set_xlabel('Atributo 2')
ax1.set_ylabel('Atributo 4')
ax1.set_zlabel('Atributo 5')
fig.colorbar(scatter1, ax=ax1, label='Classe Real', shrink=0.5)

# Gráfico da Direita: Predição Mamdani
ax2 = fig.add_subplot(122, projection='3d')
scatter2 = ax2.scatter(X_val[:, idx2], X_val[:, idx4], X_val[:, idx5], 
                      c=predicoes_mamdani, cmap='viridis', s=25, alpha=0.6)
ax2.set_title('Predições do Método Mamdani')
ax2.set_xlabel('Atributo 2')
ax2.set_ylabel('Atributo 4')
ax2.set_zlabel('Atributo 5')
fig.colorbar(scatter2, ax=ax2, label='Classe Predita', shrink=0.5)

# Cálculo da acurácia para exibir no título
acuracia = accuracy_score(y_val, predicoes_mamdani)
plt.suptitle(f'Desempenho do Modelo Mamdani (Atributos 2, 4 e 5)\nAcurácia Global: {acuracia:.4f}', fontsize=14)

plt.tight_layout()
caminho_saida = 'imgs/ajuste_modelo_mamdani.png'
plt.savefig(caminho_saida)
print(f"Gráfico salvo com sucesso em: {caminho_saida}")
