import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Carregar a base (certifique-se de usar a base padronizada do passo anterior)
df = pd.read_csv('data/base_sintetica_media.csv')

# 2. Matriz de Correlação
# Ajuda a entender quais atributos "andam juntos"
plt.figure(figsize=(10, 8))
correlation_matrix = df.drop('classe', axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação entre Atributos')
plt.show()

# 3. Gráfico de Dispersão Pareado (Pairplot)
# Este é o mais importante para o seu trabalho. 
# Ele mostra a dispersão de cada par de atributos colorindo pela 'classe'.
print("Gerando gráficos de dispersão... Isso pode levar alguns segundos devido ao tamanho da base.")
sns.set(style="ticks")
g = sns.pairplot(df, hue="classe", palette="bright", diag_kind="kde", plot_kws={'alpha':0.5})
g.fig.suptitle("Dispersão dos Atributos por Classe", y=1.02)
plt.show()