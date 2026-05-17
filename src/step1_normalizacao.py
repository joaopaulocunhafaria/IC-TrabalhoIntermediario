import pandas as pd
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('data/base_sintetica_media.csv')

# Tratamento de valores ausentes (Imputação pela média)
df_filled = df.fillna(df.mean())

# Seleção dos atributos de entrada
atributos_col = ['atributo_2', 'atributo_4', 'atributo_5']
df_processado = df_filled.copy()

# Aplicação do StandardScaler
scaler = StandardScaler()
df_processado[atributos_col] = scaler.fit_transform(df_filled[atributos_col])


df_processado.to_csv('output/base_sintetica_padronizada.csv', index=False)

print("Padronização (Z-score) concluída.")
print("\nNovas estatísticas (Média aproximada de 0 e Desvio Padrão de 1):")
print(df_processado[atributos_col].describe().loc[['mean', 'std', 'min', 'max']])