import pandas as pd
import numpy as np
import skfuzzy as fuzzy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error

cntr = np.load('output/centros.npy')
df_val = pd.read_csv('output/base_teste_validacao.csv')
theta = np.load('output/coeficientes_ts.npy')

atributos_selecionados = ['atributo_1', 'atributo_2', 'atributo_3', 'atributo_4', 'atributo_5', 'atributo_6']
X_val = df_val[atributos_selecionados].values
y_val = df_val['classe'].values

m = 1.5
error = 0.005
maxiter = 1000

# Realizando inferência na validação usando os coeficientes carregados
u_val, _, _, _, _, _ = fuzzy.cluster.cmeans_predict(
    X_val.T, cntr, m=m, error=error, maxiter=maxiter
)
w_val = u_val.T

n_clusters = cntr.shape[0]
n_features = len(atributos_selecionados)

y_pred_cont = []
for k in range(X_val.shape[0]):
    y_k = 0
    param_idx = 0
    for i in range(n_clusters):
        y_ik = theta[param_idx]
        param_idx += 1
        for f in range(n_features):
            y_ik += theta[param_idx] * X_val[k, f]
            param_idx += 1
        y_k += w_val[k, i] * y_ik
    y_pred_cont.append(y_k)

y_pred_cont = np.array(y_pred_cont)
y_pred_final = np.round(y_pred_cont).astype(int)
y_pred_final = np.clip(y_pred_final, 1, 4)

# 3. Visualização
print("Gerando gráficos...")
# Atributos 2, 4 e 5 (índices 1, 3, 4)
idx2, idx4, idx5 = 1, 3, 4

fig = plt.figure(figsize=(15, 7))

# Subplot 1: Classes Reais
ax1 = fig.add_subplot(121, projection='3d')
scatter1 = ax1.scatter(X_val[:, idx2], X_val[:, idx4], X_val[:, idx5], 
                      c=y_val, cmap='viridis', s=20, alpha=0.6)
ax1.set_title('Classes Reais (Conjunto de Validação)')
ax1.set_xlabel('Atributo 2')
ax1.set_ylabel('Atributo 4')
ax1.set_zlabel('Atributo 5')
fig.colorbar(scatter1, ax=ax1, label='Classe', shrink=0.5)

# Subplot 2: Predições do Modelo TS
ax2 = fig.add_subplot(122, projection='3d')
scatter2 = ax2.scatter(X_val[:, idx2], X_val[:, idx4], X_val[:, idx5], 
                      c=y_pred_cont, cmap='viridis', s=20, alpha=0.6)
ax2.set_title('Predições Contínuas do Modelo TS Ordem 1')
ax2.set_xlabel('Atributo 2')
ax2.set_ylabel('Atributo 4')
ax2.set_zlabel('Atributo 5')
fig.colorbar(scatter2, ax=ax2, label='Saída TS (Contínua)', shrink=0.5)

accuracy = accuracy_score(y_val, y_pred_final)
mse = mean_squared_error(y_val, y_pred_cont)
plt.suptitle(f'Ajuste do Modelo Takagi-Sugeno de Ordem 1 (LSE)\nAcurácia: {accuracy:.4f} | MSE: {mse:.4f}', fontsize=14)

plt.tight_layout()
plt.savefig('imgs/ajuste_modelo_ts.png')
print("Gráfico atualizado e salvo em 'imgs/ajuste_modelo_ts.png'")
