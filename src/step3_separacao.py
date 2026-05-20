import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('output/base_sintetica_padronizada.csv')

# Usando todos os atributos normalizados
atributos_selecionados = ['atributo_1', 'atributo_2', 'atributo_3', 'atributo_4', 'atributo_5', 'atributo_6']

df_reduzido = df[atributos_selecionados + ['classe']]

# Divisão em Treino (80%) e Teste (20%)
df_treino, df_teste = train_test_split(df_reduzido, test_size=0.20, random_state=42)

# Conjunto 1: Treino com a classe para sementes de conhecimento (Centros Iniciais)
df_treino.to_csv('output/base_treino.csv', index=False)

# Conjunto 2: O conjunto de 20% isolado com a classe para validação 
df_teste.to_csv('output/base_teste_validacao.csv', index=False)

print("Processamento concluído com todos os atributos!")
print(f"-> Conjunto 1 (Treino com Classe): {df_treino.shape[0]} amostras.")
print(f"-> Conjunto 2 (Validação - 20% com Classe): {df_teste.shape[0]} amostras.")
