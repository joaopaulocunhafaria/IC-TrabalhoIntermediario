import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#  Carregamento dos dados
df = pd.read_csv('data/base_sintetica_media.csv')

# Análise de classes
classes = sorted(df['classe'].unique())
num_classes = len(classes)
total_registros = len(df)
dimensoes = df.shape

distribuicao = df['classe'].value_counts().sort_index()
proporcoes = (distribuicao / total_registros) * 100

print("=" * 60)
print("ANÁLISE DE CLASSES - BASE DE DADOS")
print("=" * 60)
print(f"\nTotal de classes encontradas: {num_classes}")
print(f"Classes presentes: {classes}")

print("\nDistribuição de registros por classe:")
print("-" * 60)
for classe, count in distribuicao.items():
    percentual = (count / total_registros) * 100
    print(f"Classe {classe}: {count:,} registros ({percentual:.2f}%)")

print("-" * 60)
print(f"Total de registros: {total_registros:,}")
print(f"Tamanho do dataset: {dimensoes}")
print("=" * 60)

plt.rcParams['font.family'] = 'sans-serif'
fig, (ax_table, ax_chart) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1, 1.2]})
fig.patch.set_facecolor('#f0f4f7')

# Título Principal
fig.suptitle('RESUMO ESTATÍSTICO E DISTRIBUIÇÃO DO CONJUNTO DE DADOS', 
             fontsize=20, fontweight='bold', color='#003366', y=0.96)


ax_table.axis('off')


resumo_texto = (
    f"Registros Totais: {total_registros:,}\n"
    f"Número de Classes: {num_classes}\n"
    f"Dimensões do Dataset: {dimensoes}"
)
ax_table.text(0.05, 0.95, resumo_texto, fontsize=14, verticalalignment='top', fontweight='bold',
              bbox=dict(facecolor='white', edgecolor='#003366', boxstyle='round,pad=1'))

ax_table.text(0.05, 0.72, "TABELA I. DISTRIBUIÇÃO DETALHADA POR CLASSE", 
              fontsize=13, fontweight='bold', color='#003366', verticalalignment='top')

# Preparação dos dados da tabela
tabela_data = []
for i, classe in enumerate(classes):
    count = distribuicao[classe]
    prop = proporcoes[classe]
    tabela_data.append([i+1, int(classe), f"{count:,}", f"{prop:.2f}%"])
tabela_data.append(["", "Total (Soma):", f"{total_registros:,}", "100.00%"])

col_labels = ["nº", "Identificador\nda Classe", "Quantidade\nde Registros", "Proporção\n(%)"]

table = ax_table.table(cellText=tabela_data, colLabels=col_labels, cellLoc='center',
                       bbox=[0.05, 0.1, 0.9, 0.55]) 
table.auto_set_font_size(False)
table.set_fontsize(11)

# Estilização da tabela para combinar com a imagem
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='black')
        cell.set_facecolor('#d9e1f2')
    if row == len(tabela_data):
        cell.set_text_props(weight='bold')


colors = ['#2b577a' if prop > 15 else '#7b8da1' for prop in proporcoes.values]

sns.barplot(x=distribuicao.index.astype(int), y=proporcoes.values, palette=colors, ax=ax_chart, edgecolor='black')

for i, p in enumerate(ax_chart.patches):
    ax_chart.annotate(f'{proporcoes.values[i]:.2f}%', 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha='center', va='center', xytext=(0, 12), 
                      textcoords='offset points', fontsize=13, fontweight='bold', color='#003366')

ax_chart.set_title('FIGURA 1. VISUALIZAÇÃO PERCENTUAL POR CLASSE', fontsize=14, fontweight='bold', color='#003366', pad=20)
ax_chart.set_xlabel('Identificador da Classe (1–4)', fontsize=12, fontweight='bold')
ax_chart.set_ylabel('Porcentagem (%)', fontsize=12, fontweight='bold')
ax_chart.set_ylim(0, 40)
ax_chart.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))
ax_chart.grid(axis='y', linestyle='--', alpha=0.5)
ax_chart.set_facecolor('white')

plt.figtext(0.05, 0.03, "*Observação: Os dados foram processados de forma consolidada e analítica.*", 
            fontsize=10, style='italic', color='#333333')

plt.tight_layout(rect=[0, 0.05, 1, 0.92])

if not os.path.exists('imgs'):
    os.makedirs('imgs')

output_path = 'imgs/descrica_conjunto_dados_gerada.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nVisualização consolidada salva com sucesso em: {output_path}")

plt.close()
