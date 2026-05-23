import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/base_sintetica_media.csv')

#  Matriz de Correlação
plt.figure(figsize=(10, 8))
correlation_matrix = df.drop('classe', axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação entre Atributos')
plt.show()

# Gráfico de Dispersão Pareado 
print("Gerando gráficos de dispersão... Isso pode levar alguns segundos devido ao tamanho da base.")
sns.set(style="ticks")
g = sns.pairplot(df, hue="classe", palette="bright", diag_kind="kde", plot_kws={'alpha':0.5})
g.fig.suptitle("Dispersão dos Atributos por Classe", y=1.02)
plt.show()