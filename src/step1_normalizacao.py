import pandas as pd
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('data/base_sintetica_media.csv')

# Tratamento de valores ausentes (Imputação pela média)
df_filled = df.fillna(df.mean())

atributos_col = ['atributo_1', 'atributo_2', 'atributo_3', 'atributo_4', 'atributo_5', 'atributo_6']
df_processado = df_filled.copy()

# Aplicação do StandardScaler em todos os atributos
scaler = StandardScaler()
df_processado[atributos_col] = scaler.fit_transform(df_filled[atributos_col])


df_processado.to_csv('output/base_sintetica_padronizada.csv', index=False)

print("Padronização (Z-score) de todos os atributos concluída.")
print("\nEstatísticas dos atributos selecionados:")
print(df_processado[atributos_col].describe().loc[['mean', 'std', 'min', 'max']])