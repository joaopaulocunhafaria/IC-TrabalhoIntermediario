import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Carregamento dos dados
df = pd.read_csv('data/base_sintetica_media.csv')

# 2. Tratamento de valores ausentes (Imputação pela média)
# Fundamental para não gerar erros no cálculo do desvio padrão
df_filled = df.fillna(df.mean())

# 3. Seleção dos atributos de entrada
# A coluna 'classe' deve ser preservada sem alteração para validação
atributos_col = [ 'atributo_2', 'atributo_3', 'atributo_5', 'atributo_6']
df_processado = df_filled.copy()

# 4. Aplicação do StandardScaler
# Transforma os dados para média 0 e variância 1
scaler = StandardScaler()
df_processado[atributos_col] = scaler.fit_transform(df_filled[atributos_col])

# 5. Exportação da base padronizada
df_processado.to_csv('output/base_sintetica_padronizada.csv', index=False)

print("Padronização (Z-score) concluída.")
print("\nNovas estatísticas (Média aproximada de 0 e Desvio Padrão de 1):")
print(df_processado[atributos_col].describe().loc[['mean', 'std', 'min', 'max']])