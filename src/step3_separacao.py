import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('output/base_sintetica_padronizada.csv')

atributos_selecionados = ['atributo_2', 'atributo_4', 'atributo_5']

df_reduzido = df[atributos_selecionados + ['classe']]

# Divisão em Treino (80%) e Teste (20%)
df_treino, df_teste = train_test_split(df_reduzido, test_size=0.20, random_state=42)

# Conjunto 1: Apenas os atributos preditores de treino 
# Este conjunto será injetado diretamente no algoritmo de agrupamento (K-Means / FCM)
X_treino_clustering = df_treino[atributos_selecionados]

# Conjunto 2: O conjunto de 20% isolado com a classe para validação 
X_y_teste_validacao = df_teste.copy()

X_treino_clustering.to_csv('output/base_treino.csv', index=False)
X_y_teste_validacao.to_csv('output/base_teste_validacao.csv', index=False)

print("Processamento concluído!")
print(f"-> Conjunto 1 (Treino para Clustering - Sem Classe): {X_treino_clustering.shape[0]} amostras e {X_treino_clustering.shape[1]} atributos.")
print(f"-> Conjunto 2 (Validação - 20% com Classe): {X_y_teste_validacao.shape[0]} amostras e {X_y_teste_validacao.shape[1]} colunas.")