import pandas as pd
import numpy as np
import skfuzzy as fuzzy
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def s_norm_maximo(a, b):
    """
    S-Norma máximo. 
    Preserva a força da regra que disparou com maior intensidade para a classe
    """
    return np.maximum(a, b)


# Carregamento de dados
centros = np.load('output/centros.npy')
treino = pd.read_csv('output/base_treino.csv')
validacao = pd.read_csv('output/base_teste_validacao.csv')

colunas_entrada = ['atributo_1', 'atributo_2', 'atributo_3', 'atributo_4', 'atributo_5', 'atributo_6']
X_treino = treino[colunas_entrada].values
y_treino = treino['classe'].values

X_val = validacao[colunas_entrada].values
y_val = validacao['classe'].values

# constantes
m_fuzzy = 2.0 # Ajustado para 2.0 conforme mencionado na Metodologia do main.tex
tol_erro = 0.005
iter_max = 1000


print("\n" + "="*60)
print("EXTRAÇÃO E CONSTRUÇÃO DA BASE DE REGRAS FUZZY")
print("="*60)
# Obtemos as pertinências para o conjunto de treino
u_treino, _, _, _, _, _ = fuzzy.cluster.cmeans_predict(
    X_treino.T, centros, m=m_fuzzy, error=tol_erro, maxiter=iter_max
)
pesos_treino = u_treino.T  # Pertinências das amostras aos clusters

n_clusters = centros.shape[0]
classes_unicas = sorted(np.unique(y_treino))

# Dicionário que mapeará cada Cluster (Regra) para uma Classe Consequente
regra_para_classe = {}

print("\nRegras Inferidas via Topologia de Clusters:")
print("-" * 60)
for i in range(n_clusters):
    # Encontra as amostras que têm maior pertinência a este cluster
    amostras_do_cluster = np.argmax(pesos_treino, axis=1) == i
    
    if np.any(amostras_do_cluster):
        # A classe que mais aparece dentro deste cluster define o consequente da regra
        classes_no_cluster = y_treino[amostras_do_cluster]
        classe_predominante = np.bincount(classes_no_cluster).argmax()
        regra_para_classe[i] = classe_predominante
    else:
        regra_para_classe[i] = classes_unicas[0]
    
    # Demonstração detalhada da regra para documentação
    centroide_info = ", ".join([f"{col}: {val:.2f}" for col, val in zip(colunas_entrada, centros[i])])
    print(f"REGRA {i+1}:")
    print(f"  IF (Amostra próxima ao Centroide {i+1})")
    print(f"     [Centro: {centroide_info}]")
    print(f"  THEN (Classe é {regra_para_classe[i]})")
    print("-" * 60)



# composição max-min e inferência na validação 
print("\n Executando Inferência Mamdani no conjunto de validação...")
# Obtemos as pertinências para o conjunto de validação
u_val, _, _, _, _, _ = fuzzy.cluster.cmeans_predict(
    X_val.T, centros, m=m_fuzzy, error=tol_erro, maxiter=iter_max
)
pesos_val = u_val.T # Força de disparo de cada regra para as amostras de validação

n_amostras_val = X_val.shape[0]

# Matriz para armazenar o grau de pertinência final de cada amostra a cada classe
pertinencia_final_classes = np.zeros((n_amostras_val, np.max(classes_unicas) + 1))

# Processo de Implicação e Agregação
for i in range(n_clusters):
    classe_consequente = regra_para_classe[i]
    forca_disparo_regra = pesos_val[:, i] # \alpha_i (Composição Min já implícita via FCM)
    
    # Agregação via S-Norma máximo
    pertinencia_atual = pertinencia_final_classes[:, classe_consequente]
    
    pertinencia_final_classes[:, classe_consequente] = s_norm_maximo(
        pertinencia_atual, 
        forca_disparo_regra
    )


# Defuzzificação por Máximo
predicoes_finais = np.argmax(pertinencia_final_classes[:, 1:], axis=1) + 1


print("\n" + "="*60)
print("AVALIAÇÃO DO MODELO MAMDANI (S-NORMA MÁXIMO)")
print("="*60)

acuracia = accuracy_score(y_val, predicoes_finais)
print(f"Acurácia Global: {acuracia:.4f}")

print("\nRelatório de Classificação:")
print(classification_report(y_val, predicoes_finais))

# --- Geração da Matriz de Confusão Visual ---
print("\nGerando Matriz de Confusão visual...")
cm = confusion_matrix(y_val, predicoes_finais)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes_unicas, 
            yticklabels=classes_unicas)
plt.title(f'Matriz de Confusão - Modelo Mamdani\n(Acurácia: {acuracia:.4f})', fontsize=15, fontweight='bold')
plt.xlabel('Classe Predita', fontsize=12, fontweight='bold')
plt.ylabel('Classe Real', fontsize=12, fontweight='bold')

# Garantir que o diretório existe
os.makedirs('imgs/Mamdani', exist_ok=True)
output_cm_path = 'imgs/Mamdani/matriz_confusao_mamdani.png'
plt.savefig(output_cm_path, dpi=300, bbox_inches='tight')
print(f"-> Matriz de Confusão salva em: {output_cm_path}")

plt.close()

# Exemplo detalhado de composição para a primeira amostra
amostra_idx = 0
print(f"\nExemplo de Composição para a amostra {amostra_idx}:")
for i in range(n_clusters):
    print(f"  Força da Regra {i+1} (-> Classe {regra_para_classe[i]}): {pesos_val[amostra_idx, i]:.4f}")
print(f"Pertinências Finais Agregadas (Classes 1 a 4): {np.round(pertinencia_final_classes[amostra_idx, 1:], 4)}")
print(f"Classe Predita (Defuzzificação Max): {predicoes_finais[amostra_idx]}")
print(f"Classe Real: {y_val[amostra_idx]}")
