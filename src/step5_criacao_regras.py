import pandas as pd
import numpy as np
import skfuzzy as fuzzy
from sklearn.metrics import accuracy_score, classification_report


print("1. Carregando centros e dados de validação...")
cntr = np.load('output/centros.npy')
df_validacao = pd.read_csv('output/base_teste_validacao.csv')

atributos_selecionados = ['atributo_1', 'atributo_2', 'atributo_3', 'atributo_4', 'atributo_5', 'atributo_6']
X_val = df_validacao[atributos_selecionados].values
y_real = df_validacao['classe'].values

# Definir os consequentes de ordem zero (k_i)
# Cada cluster mapeado para sua respectiva classe
consequentes = np.array([1, 2, 3, 4]) 

print(f"\n2. Definindo Regras Takagi-Sugeno de Ordem Zero:")
for i, center in enumerate(cntr):
    regra = f"REGRA {i+1}: IF (x is near Center_{i+1}) THEN y = {consequentes[i]}"
    print(regra)
    print(f"   Centro: {np.round(center, 2)}")

# Implementar a Inferência Takagi-Sugeno
print("\n3. Executando Inferência Takagi-Sugeno (Ordem Zero) na validação...")

m = 1.5
error = 0.005
maxiter = 1000

u_val, _, _, _, _, _ = fuzzy.cluster.cmeans_predict(
    X_val.T, cntr, m=m, error=error, maxiter=maxiter
)

pesos = u_val.T 
# Saída Fuzzy (Média ponderada dos consequentes constantes)
y_pred_fuzz = np.dot(pesos, consequentes)

# Para classificação: Classe com maior pertinência
y_pred_final = np.argmax(pesos, axis=1) + 1

# 4. Avaliação
print("\n" + "="*60)
print("             AVALIAÇÃO DO MODELO TAKAGI-SUGENO (ORDEM ZERO)")
print("="*60)

acuracia = accuracy_score(y_real, y_pred_final)
print(f"Acurácia: {acuracia:.4f}")

print("\nRelatório de Classificação:")
print(classification_report(y_real, y_pred_final))

# Exemplo de uma inferência individual
amostra_idx = 6
print(f"\nExemplo de Inferência para a amostra {amostra_idx}:")
print(f"Atributos: {X_val[amostra_idx]}")
print(f"Pertinências (Pesos w_i): {np.round(pesos[amostra_idx], 4)}")
print(f"Saída Fuzzy (y_pred_fuzz): {y_pred_fuzz[amostra_idx]:.4f}")
print(f"Classe Predita: {y_pred_final[amostra_idx]}")
print(f"Classe Real: {y_real[amostra_idx]}")
